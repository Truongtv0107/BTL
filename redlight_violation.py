import sys
import os
import cv2
import csv
import time
import datetime
import numpy as np
import tkinter as tk
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QLabel, QFileDialog, QApplication, QMessageBox
)
from ultralytics import YOLO

# ---------- Cấu hình chung ----------
TARGET_W, TARGET_H = 1280, 720

ROI_LIGHT_LEFT = (21 - 15, 108 - 35, 21 + 15, 108 + 35)   # (x1,y1,x2,y2)
ROI_LIGHT_RIGHT = (1242, 34, 1272, 104)                  # (x1,y1,x2,y2)

LINE_THICKNESS = 12

# Tọa độ Vạch Dừng (Stop Line)
STOP_LINE_X1 = 89
STOP_LINE_X2 = 459
STOP_LINE_X3 = 1086
STOP_LINE_Y_HEIGHT = 400  # y (pixel) của vạch dừng
LINE_Y = STOP_LINE_Y_HEIGHT

LINE3_X1, LINE3_Y1 = 73, 401
LINE3_X2, LINE3_Y2 = 352, 83

LINE_S4_X1, LINE_S4_Y1 = 1123, 370
LINE_S4_X2, LINE_S4_Y2 = 1005, 81

# Bản đồ màu (BGR)
COLOR_MAP = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "YELLOW": (0, 255, 255),
    "UNKNOWN": (255, 255, 255)
}

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO indices, tuỳ model)

# Thư mục lưu vi phạm + báo cáo
VIOLATION_DIR = "violations"
REPORT_CSV = os.path.join(VIOLATION_DIR, "report.csv")

def ensure_violation_dir():
    os.makedirs(VIOLATION_DIR, exist_ok=True)
    # nếu chưa có file csv, tạo và viết header
    if not os.path.exists(REPORT_CSV):
        with open(REPORT_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "timestamp", "image_path",
                "x1", "y1", "x2", "y2", "cx", "bottom_y",
                "lane", "light_right", "light_left"
            ])

def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()

def clamp_roi(x1, y1, x2, y2, w, h):
    x1c = max(0, min(w - 1, int(round(x1))))
    y1c = max(0, min(h - 1, int(round(y1))))
    x2c = max(0, min(w, int(round(x2))))
    y2c = max(0, min(h, int(round(y2))))
    if x2c <= x1c or y2c <= y1c:
        return None
    return x1c, y1c, x2c, y2c

def bgr_mean_to_hsv_color(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = roi_hsv[..., 0].mean(), roi_hsv[..., 1].mean(), roi_hsv[..., 2].mean()
    return h, s, v

def decide_light_from_hsv(h, s, v, is_right=False):
    if h is None:
        return "UNKNOWN"
    if v < 50 or s < 50:
        return "UNKNOWN"
    if (h < 10 or h > 165) and v > 80:
        return "RED"
    if 20 <= h <= 40 and v > 90 and s > 80:
        return "YELLOW"
    if 35 <= h <= 90 and v > 80 and s > 80:
        return "GREEN"
    if is_right:
        if 35 <= h <= 90 and v > 80 and s > 70:
            return "GREEN"
        return "RED"
    return "UNKNOWN"

class DetectWorker(QThread):
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    # optional: can emit new violation info for GUI
    new_violation_signal = pyqtSignal(dict)

    def __init__(self, source=0, model_path="yolov8m.pt"):
        super().__init__()
        self.source = source
        self.model_path = model_path
        self._running = False
        self.model = None
        self.violation_counter = 0  # để đặt id file

        ensure_violation_dir()
        # khởi tạo counter từ CSV hiện tại (nếu có) để tránh trùng id
        try:
            if os.path.exists(REPORT_CSV):
                with open(REPORT_CSV, newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) > 1:
                        # last id
                        last = rows[-1][0]
                        try:
                            self.violation_counter = int(last)
                        except Exception:
                            self.violation_counter = 0
        except Exception:
            self.violation_counter = 0

    def stop(self):
        self._running = False

    def save_violation(self, crop_img, bbox, cx, bottom_y, lane, light_right, light_left):
        """
        Lưu ảnh crop và ghi vào report CSV.
        crop_img: numpy array (BGR)
        bbox: (x1,y1,x2,y2)
        lane: str/int
        """
        self.violation_counter += 1
        vid = self.violation_counter
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violation_{timestamp}_{vid}.jpg"
        path = os.path.join(VIOLATION_DIR, filename)
        # bảo đảm crop không rỗng
        try:
            if crop_img is None or crop_img.size == 0:
                # fallback: không lưu ảnh, nhưng vẫn ghi report với empty path
                img_path = ""
            else:
                # nén và lưu
                cv2.imwrite(path, crop_img)
                img_path = path
        except Exception:
            img_path = ""
        # ghi CSV
        try:
            with open(REPORT_CSV, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([vid, datetime.datetime.now().isoformat(), img_path,
                                 bbox[0], bbox[1], bbox[2], bbox[3], cx, bottom_y,
                                 lane, light_right, light_left])
        except Exception as e:
            # nếu có lỗi ghi file, phát tín hiệu trạng thái
            self.status_signal.emit(f"Lỗi ghi báo cáo: {e}")

        # gửi signal cho GUI nếu cần hiển thị ngay
        violation_info = {
            "id": vid, "timestamp": datetime.datetime.now().isoformat(), "image_path": img_path,
            "bbox": bbox, "cx": cx, "bottom_y": bottom_y, "lane": lane,
            "light_right": light_right, "light_left": light_left
        }
        self.new_violation_signal.emit(violation_info)

    def run(self):
        # tải model trong thread
        try:
            self.status_signal.emit("Đang tải model YOLO...")
            self.model = YOLO(self.model_path)
            self.status_signal.emit("Model YOLO sẵn sàng.")
        except Exception as e:
            self.status_signal.emit(f"Lỗi tải model: {e}")
            self.finished_signal.emit()
            return

        # mở nguồn
        cap = None
        if isinstance(self.source, str) and os.path.exists(self.source):
            cap = cv2.VideoCapture(self.source)
        else:
            try:
                idx = int(self.source)
            except Exception:
                idx = 0
            cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            self.status_signal.emit("❌ Không thể mở nguồn video/camera.")
            self.finished_signal.emit()
            return

        self._running = True
        screen_w, screen_h = get_screen_size()

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (TARGET_W, TARGET_H))
                fh, fw = frame.shape[:2]

                roi_l_coords = clamp_roi(*ROI_LIGHT_LEFT, fw, fh)
                roi_r_coords = clamp_roi(*ROI_LIGHT_RIGHT, fw, fh)

                roi_l = None
                roi_r = None
                if roi_l_coords:
                    x1_l, y1_l, x2_l, y2_l = roi_l_coords
                    roi_l = frame[y1_l:y2_l, x1_l:x2_l]
                if roi_r_coords:
                    x1_r, y1_r, x2_r, y2_r = roi_r_coords
                    roi_r = frame[y1_r:y2_r, x1_r:x2_r]

                left_hsv = bgr_mean_to_hsv_color(roi_l) if roi_l is not None else (None, None, None)
                right_hsv = bgr_mean_to_hsv_color(roi_r) if roi_r is not None else (None, None, None)

                light_left = decide_light_from_hsv(*left_hsv, is_right=False)
                light_right = decide_light_from_hsv(*right_hsv, is_right=True)

                # Vẽ ROI
                if roi_l_coords:
                    cv2.rectangle(frame, (x1_l, y1_l), (x2_l, y2_l), COLOR_MAP.get(light_left, (255,255,255)), 2)
                    cv2.putText(frame, f"LEFT: {light_left}", (x1_l, y1_l - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAP.get(light_left), 2)
                if roi_r_coords:
                    cv2.rectangle(frame, (x1_r, y1_r), (x2_r, y2_r), COLOR_MAP.get(light_right, (255,255,255)), 2)
                    cv2.putText(frame, f"RIGHT: {light_right}", (max(0, x1_r - 50), y1_r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAP.get(light_right), 2)

                # Vẽ vạch
                color_vach = COLOR_MAP.get(light_right, COLOR_MAP["UNKNOWN"])
                cv2.line(frame, (STOP_LINE_X1, LINE_Y), (STOP_LINE_X2, LINE_Y), color_vach, LINE_THICKNESS)
                cv2.line(frame, (STOP_LINE_X2 + 1, LINE_Y), (STOP_LINE_X3, LINE_Y), color_vach, LINE_THICKNESS)

                color_v3 = COLOR_MAP.get(light_left, COLOR_MAP["UNKNOWN"])
                cv2.line(frame, (LINE3_X1, LINE3_Y1), (LINE3_X2, LINE3_Y2), color_v3, 3)
                cv2.putText(frame, "Vach 3", (LINE3_X1 + 5, LINE3_Y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_v3, 2)

                color_s4 = COLOR_MAP.get(light_right, COLOR_MAP["UNKNOWN"])
                cv2.line(frame, (LINE_S4_X1, LINE_S4_Y1), (LINE_S4_X2, LINE_S4_Y2), color_s4, 3)
                cv2.putText(frame, "Vach S4", (LINE_S4_X1 - 80, LINE_S4_Y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_s4, 2)

                # Chạy YOLO
                try:
                    results = self.model(frame, verbose=False)
                except Exception as e:
                    self.status_signal.emit(f"Lỗi model trên frame: {e}")
                    results = None

                if results is not None:
                    for box in results[0].boxes:
                        try:
                            cls = int(box.cls)
                        except Exception:
                            continue
                        if cls not in VEHICLE_CLASSES:
                            continue

                        x1_obj, y1_obj, x2_obj, y2_obj = map(int, box.xyxy[0].tolist())
                        bottom_y = y2_obj
                        cx = (x1_obj + x2_obj) // 2

                        is_violating = False
                        label_text = "hop le"
                        color_box = (0, 255, 0)
                        lane = "unknown"

                        # Điều kiện Vi Phạm: đèn phải RED + xe vượt vạch (bottom_y > LINE_Y)
                        # với tâm cx ở làn thứ 2 (STOP_LINE_X2 < cx <= STOP_LINE_X3)
                        if light_right == "RED" and (STOP_LINE_X2 < cx <= STOP_LINE_X3) and bottom_y < LINE_Y:
                            is_violating = True
                            label_text = "VI PHAM !"
                            color_box = (0, 0, 255)
                            lane = "lane_2"

                        # Vẽ bbox và label
                        cv2.rectangle(frame, (x1_obj, y1_obj), (x2_obj, y2_obj), color_box, 2)
                        cv2.putText(frame, label_text, (x1_obj, max(0, y1_obj - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)

                        # Nếu vi phạm -> crop & lưu ảnh + ghi report
                        if is_violating:
                            # mở rộng bbox một chút để ảnh dễ nhìn (padding)
                            pad_x = int((x2_obj - x1_obj) * 0.1)  # 10% padding
                            pad_y = int((y2_obj - y1_obj) * 0.1)
                            cx1 = max(0, x1_obj - pad_x)
                            cy1 = max(0, y1_obj - pad_y)
                            cx2 = min(fw, x2_obj + pad_x)
                            cy2 = min(fh, y2_obj + pad_y)
                            crop = frame[cy1:cy2, cx1:cx2].copy() if (cy2>cy1 and cx2>cx1) else frame[y1_obj:y2_obj, x1_obj:x2_obj].copy()
                            # Lưu thông tin
                            try:
                                self.save_violation(
                                    crop_img=crop,
                                    bbox=(x1_obj, y1_obj, x2_obj, y2_obj),
                                    cx=cx,
                                    bottom_y=bottom_y,
                                    lane=lane,
                                    light_right=light_right,
                                    light_left=light_left
                                )
                                # cập nhật status ngắn để GUI hiển thị
                                self.status_signal.emit(f"Phát hiện vi phạm: id {self.violation_counter}")
                            except Exception as e:
                                self.status_signal.emit(f"Lỗi lưu vi phạm: {e}")

                # Cập nhật status
                status_text = (f"Đèn Trái: {light_left} | Đèn Phải: {light_right} | Vi phạm đã lưu: {self.violation_counter}")
                self.status_signal.emit(status_text)

                # Hiển thị khung OpenCV
                cv2.imshow("Red Light Detection", frame)
                win_w, win_h = frame.shape[1], frame.shape[0]
                if win_w > screen_w or win_h > screen_h:
                    scale = min(screen_w / win_w, screen_h / win_h) * 0.7
                    try:
                        cv2.resizeWindow("Red Light Detection", int(win_w * scale), int(win_h * scale))
                    except Exception:
                        pass

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._running = False
                    break

        except Exception as e:
            self.status_signal.emit(f"Lỗi xử lý video: {e}")
        finally:
            try:
                cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()
            self.finished_signal.emit()

class RedLight_violationDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚦 Nhận Diện Vượt Đèn Đỏ & Lưu Báo Cáo")
        self.setFixedSize(640, 320)

        screen_w, screen_h = get_screen_size()
        self.move((screen_w - self.width()) // 2, (screen_h - self.height()) // 2)

        layout = QVBoxLayout()
        self.label = QLabel("Hệ thống đã sẵn sàng.")
        self.btn_start = QPushButton("▶ Bắt đầu (Camera)")
        self.btn_video = QPushButton("📂 Chọn video")
        self.btn_stop = QPushButton("⏹ Dừng")
        self.btn_report = QPushButton("📋 Xem báo cáo")

        layout.addWidget(self.label)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_video)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_report)
        self.setLayout(layout)

        # worker thread
        self.worker = None

        # Kết nối các nút
        self.btn_start.clicked.connect(self.start_detect_camera)
        self.btn_video.clicked.connect(self.start_detect_video)
        self.btn_stop.clicked.connect(self.stop_detect)
        self.btn_report.clicked.connect(self.show_report)

        # đảm bảo thư mục report tồn tại
        ensure_violation_dir()

    def update_status(self, text):
        # cập nhật nhãn (giữ ngắn gọn)
        self.label.setText(text)

    def start_detect_camera(self):
        if self.worker is not None and self.worker.isRunning():
            self.update_status("Đang chạy rồi.")
            return
        self.worker = DetectWorker(source=0, model_path="yolov8m.pt")
        self.worker.status_signal.connect(self.update_status)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.new_violation_signal.connect(self.on_new_violation)
        self.worker.start()
        self.update_status("Bắt đầu camera...")

    def start_detect_video(self):
        if self.worker is not None and self.worker.isRunning():
            self.update_status("Đang chạy rồi.")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path and os.path.exists(file_path):
            self.worker = DetectWorker(source=file_path, model_path="yolov8m.pt")
            self.worker.status_signal.connect(self.update_status)
            self.worker.finished_signal.connect(self.on_finished)
            self.worker.new_violation_signal.connect(self.on_new_violation)
            self.worker.start()
            self.update_status(f"Bắt đầu phát hiện trên: {os.path.basename(file_path)}")
        else:
            self.update_status("Chưa chọn file hoặc file không tồn tại.")

    def stop_detect(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.update_status("Đang dừng... (chờ thread kết thúc)")
        else:
            self.update_status("Không có quá trình nào đang chạy.")

    def on_finished(self):
        self.update_status("Đã dừng phát hiện.")

    def on_new_violation(self, info):
        # info là dict chứa thông tin vi phạm mới
        self.update_status(f"Vi phạm mới: id {info.get('id')} - {info.get('timestamp')}")

    def show_report(self):
        # đọc CSV và hiển thị 10 dòng cuối
        if not os.path.exists(REPORT_CSV):
            QMessageBox.information(self, "Báo cáo", "Chưa có báo cáo vi phạm nào.")
            return
        try:
            with open(REPORT_CSV, newline='', encoding='utf-8') as f:
                reader = list(csv.reader(f))
                if len(reader) <= 1:
                    QMessageBox.information(self, "Báo cáo", "Chưa có mục vi phạm.")
                    return
                rows = reader[1:]  # skip header
                last_rows = rows[-10:] if len(rows) > 10 else rows
                # build message
                msg_lines = []
                for r in reversed(last_rows):
                    # r: id, timestamp, image_path, x1,y1,x2,y2,cx,bottom_y,lane,light_right,light_left
                    img = os.path.basename(r[2]) if r[2] else "no-image"
                    msg_lines.append(f"ID {r[0]} | {r[1]} | {img} | lane:{r[9]} | lightR:{r[10]} | lightL:{r[11]}")
                msg = "\n".join(msg_lines)
                QMessageBox.information(self, "10 vi phạm gần nhất", msg)
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể đọc báo cáo: {e}")

def main():
    app = QApplication(sys.argv)
    dlg = RedLight_violationDialog()
    dlg.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
