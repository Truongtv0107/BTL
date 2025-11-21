import sys
import os
import cv2
import csv
import datetime
import tkinter as tk
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QApplication
)
from ultralytics import YOLO

# ---------- Cấu hình chung ----------
TARGET_W, TARGET_H = 1280, 720

# ROI đèn giao thông (x1, y1, x2, y2)
ROI_LIGHT_LEFT = (21 - 15, 108 - 15, 21 + 15, 150 + 35)
ROI_LIGHT_RIGHT = (1242, 30, 1272, 125)

LINE_THICKNESS = 12

# Tọa độ Vạch Dừng (Stop Line)
STOP_LINE_X1 = 89
STOP_LINE_X2 = 459
STOP_LINE_X3 = 1086
STOP_LINE_Y_HEIGHT = 400  # y (pixel) của vạch dừng
LINE_Y = STOP_LINE_Y_HEIGHT

# Vạch chéo / phụ
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

# Lớp xe trong COCO (tùy theo model)
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Thư mục lưu vi phạm + báo cáo
VIOLATION_DIR = "violations"
REPORT_CSV = os.path.join(VIOLATION_DIR, "report.csv")   # chi tiết bbox, ảnh, đèn
STATUS_CSV = os.path.join(VIOLATION_DIR, "status.csv")   # bảng: stt, ngày, loại, tình trạng


# ---------- Hàm tiện ích ----------
def ensure_violation_dir():
    """Tạo thư mục và 2 file CSV (report, status) nếu chưa có."""
    os.makedirs(VIOLATION_DIR, exist_ok=True)

    # report.csv – chi tiết vi phạm
    if not os.path.exists(REPORT_CSV):
        with open(REPORT_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "timestamp", "image_path",
                "x1", "y1", "x2", "y2", "cx", "bottom_y",
                "lane", "light_right", "light_left"
            ])

    # status.csv – bảng tóm tắt xử lý
    if not os.path.exists(STATUS_CSV):
        with open(STATUS_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "stt", "ngay_vi_pham", "loai_vi_pham", "tinh_trang"
            ])


def get_screen_size():
    """Lấy kích thước màn hình (dùng Tkinter cho đơn giản)."""
    root = tk.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()


def clamp_roi(x1, y1, x2, y2, w, h):
    """Giới hạn ROI trong khung hình."""
    x1c = max(0, min(w - 1, int(round(x1))))
    y1c = max(0, min(h - 1, int(round(y1))))
    x2c = max(0, min(w,      int(round(x2))))
    y2c = max(0, min(h,      int(round(y2))))
    if x2c <= x1c or y2c <= y1c:
        return None
    return x1c, y1c, x2c, y2c


def detect_light_color(roi_bgr):
    """
    Nhận diện màu đèn từ ROI bằng HSV.
    Đếm số pixel RED / YELLOW / GREEN và chọn nhiều nhất nếu vượt ngưỡng.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Điều kiện chung: S, V phải đủ lớn
    sat_mask = s > 80
    val_mask = v > 80
    sv_mask = cv2.bitwise_and(sat_mask.astype('uint8'), val_mask.astype('uint8'))

    # RED có 2 dải H: [0,10] và [160,180]
    red_mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # YELLOW
    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))

    # GREEN
    green_mask = cv2.inRange(hsv, (40, 80, 80), (85, 255, 255))

    # Áp thêm mask S,V
    red_mask = cv2.bitwise_and(red_mask, red_mask, mask=sv_mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, yellow_mask, mask=sv_mask)
    green_mask = cv2.bitwise_and(green_mask, green_mask, mask=sv_mask)

    red_count = int(cv2.countNonZero(red_mask))
    yellow_count = int(cv2.countNonZero(yellow_mask))
    green_count = int(cv2.countNonZero(green_mask))

    total_pixels = roi_bgr.shape[0] * roi_bgr.shape[1]
    if total_pixels == 0:
        return "UNKNOWN"

    # Tỉ lệ tối thiểu để coi là có đèn (ví dụ 1% ROI)
    min_ratio = 0.01
    max_count = max(red_count, yellow_count, green_count)
    if max_count < total_pixels * min_ratio:
        return "UNKNOWN"

    if max_count == red_count:
        return "RED"
    elif max_count == yellow_count:
        return "YELLOW"
    elif max_count == green_count:
        return "GREEN"
    return "UNKNOWN"


def detect_left_light(roi_bgr):
    """
    Đèn bên trái CHỈ có 2 trạng thái: RED hoặc GREEN.
    Nếu không rõ (YELLOW/UNKNOWN) -> ép về RED (an toàn).
    """
    base = detect_light_color(roi_bgr)
    if base == "GREEN":
        return "GREEN"
    return "RED"


# ---------- Simple Tracker (centroid tracking) ----------
class SimpleTracker:
    """
    Tracker đơn giản theo dõi xe bằng tâm (cx, bottom_y).
    - Gán ID cho mỗi xe mới.
    - Ghép detection mới với object cũ bằng khoảng cách tâm.
    """
    def __init__(self, dist_thresh=80, max_lost=10):
        self.next_id = 1
        self.objects = {}  # id -> {'cx','bottom_y','bbox','lost'}
        self.dist_thresh = dist_thresh
        self.max_lost = max_lost

    def update(self, detections):
        """
        detections: list[{'cx', 'bottom_y', 'bbox'}]
        return: list[{'id', 'cx', 'bottom_y', 'bbox'}]
        """
        results = []

        # Đánh dấu tất cả object là chưa cập nhật
        for obj in self.objects.values():
            obj["updated"] = False

        # Gán detection -> object cũ nếu khoảng cách < dist_thresh
        for det in detections:
            cx = det["cx"]
            by = det["bottom_y"]

            best_id = None
            best_dist = None

            for obj_id, obj in self.objects.items():
                dx = cx - obj["cx"]
                dy = by - obj["bottom_y"]
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist <= self.dist_thresh and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_id = obj_id

            if best_id is not None:
                # Cập nhật object cũ
                obj = self.objects[best_id]
                obj["cx"] = cx
                obj["bottom_y"] = by
                obj["bbox"] = det["bbox"]
                obj["lost"] = 0
                obj["updated"] = True

                det_with_id = det.copy()
                det_with_id["id"] = best_id
                results.append(det_with_id)
            else:
                # Tạo object mới
                new_id = self.next_id
                self.next_id += 1
                self.objects[new_id] = {
                    "cx": cx,
                    "bottom_y": by,
                    "bbox": det["bbox"],
                    "lost": 0,
                    "updated": True,
                }
                det_with_id = det.copy()
                det_with_id["id"] = new_id
                results.append(det_with_id)

        # Tăng lost cho các object không được cập nhật
        to_delete = []
        for obj_id, obj in self.objects.items():
            if not obj.get("updated", False):
                obj["lost"] += 1
                if obj["lost"] > self.max_lost:
                    to_delete.append(obj_id)

        for obj_id in to_delete:
            del self.objects[obj_id]

        return results


# ---------- Worker YOLO ----------
class DetectWorker(QThread):
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    new_violation_signal = pyqtSignal(dict)  # nếu cần dùng sau

    def __init__(self, source=0, model_path="yolov8m.pt"):
        super().__init__()
        self.source = source
        self.model_path = model_path
        self._running = False
        self.model = None
        self.violation_counter = 0  # để đặt stt/id

        ensure_violation_dir()
        self._init_violation_counter_from_csv()

        # Tracker & danh sách ID đã vi phạm
        self.tracker = SimpleTracker(dist_thresh=80, max_lost=10)
        self.violated_track_ids = set()

        # Lưu các vi phạm gần đây (dùng để khử trùng không gian + thời gian)
        # mỗi phần tử: {"track_id", "cx", "bottom_y", "bbox", "time"}
        self.recent_violations = []

    # ---------- Khởi tạo id ban đầu ----------
    def _init_violation_counter_from_csv(self):
        """Lấy id cuối cùng trong report.csv để tiếp tục đếm, tránh trùng id."""
        try:
            if os.path.exists(REPORT_CSV):
                with open(REPORT_CSV, newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) > 1:
                        last = rows[-1][0]  # id dòng cuối
                        try:
                            self.violation_counter = int(last)
                        except Exception:
                            self.violation_counter = 0
        except Exception:
            self.violation_counter = 0

    # ---------- Điều khiển ----------
    def stop(self):
        self._running = False

    # ---------- Khử trùng vi phạm ----------
    def _cleanup_recent_violations(self, max_age_sec=5.0):
        """Xóa các entry vi phạm quá cũ (mặc định > 5 giây)."""
        now = datetime.datetime.now()
        self.recent_violations = [
            v for v in self.recent_violations
            if (now - v["time"]).total_seconds() < max_age_sec
        ]

    @staticmethod
    def _bbox_iou(b1, b2):
        """Tính IoU giữa 2 bbox (x1,y1,x2,y2)."""
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0

        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = max(1e-6, area1 + area2 - inter)
        return inter / union

    def _recently_captured(self, cx, bottom_y, bbox,
                           pos_threshold=80, iou_threshold=0.3):
        """
        Kiểm tra xem vi phạm này có trùng với 1 vi phạm đã lưu gần đây không.
        - Gần về tâm (cx, bottom_y)
        - IoU bbox đủ lớn
        """
        for v in self.recent_violations:
            if abs(cx - v["cx"]) < pos_threshold and abs(bottom_y - v["bottom_y"]) < pos_threshold:
                iou = self._bbox_iou(bbox, v["bbox"])
                if iou > iou_threshold:
                    return True
        return False

    def _add_recent_violation(self, track_id, cx, bottom_y, bbox):
        self.recent_violations.append({
            "track_id": track_id,
            "cx": cx,
            "bottom_y": bottom_y,
            "bbox": bbox,
            "time": datetime.datetime.now(),
        })

    # ---------- Lưu vi phạm ----------
    def save_violation(self, crop_img, bbox, cx, bottom_y, lane, light_right, light_left):
        """
        Lưu ảnh crop + ghi vào report.csv và status.csv.
        crop_img: numpy array (BGR)
        bbox: (x1,y1,x2,y2)
        lane: str/int
        """
        self.violation_counter += 1
        vid = self.violation_counter
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        filename = f"violation_{timestamp_str}_{vid}.jpg"
        path = os.path.join(VIOLATION_DIR, filename)

        # Lưu ảnh
        try:
            if crop_img is None or crop_img.size == 0:
                img_path = ""
            else:
                cv2.imwrite(path, crop_img)
                img_path = path
        except Exception:
            img_path = ""

        # Ghi chi tiết vào REPORT_CSV
        try:
            with open(REPORT_CSV, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    vid,
                    now.isoformat(),
                    img_path,
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    cx, bottom_y,
                    lane, light_right, light_left
                ])
        except Exception as e:
            self.status_signal.emit(f"Lỗi ghi báo cáo: {e}")

        # Ghi bảng tóm tắt vào STATUS_CSV
        try:
            ngay_vi_pham = now.strftime("%d/%m/%Y")
            loai_vi_pham = "Vượt đèn đỏ"
            tinh_trang = "Chờ xử lý"
            with open(STATUS_CSV, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    vid, ngay_vi_pham, loai_vi_pham, tinh_trang
                ])
        except Exception as e:
            self.status_signal.emit(f"Lỗi ghi status: {e}")

        # Gửi signal cho GUI nếu muốn dùng
        violation_info = {
            "id": vid,
            "timestamp": now.isoformat(),
            "image_path": img_path,
            "bbox": bbox,
            "cx": cx,
            "bottom_y": bottom_y,
            "lane": lane,
            "light_right": light_right,
            "light_left": light_left
        }
        self.new_violation_signal.emit(violation_info)

    # ---------- Luồng chính ----------
    def run(self):
        # Tải model YOLO
        try:
            self.status_signal.emit("Đang tải model YOLO...")
            self.model = YOLO(self.model_path)
            self.status_signal.emit("Model YOLO sẵn sàng.")
        except Exception as e:
            self.status_signal.emit(f"Lỗi tải model: {e}")
            self.finished_signal.emit()
            return

        # Mở nguồn video/camera
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

                # Dọn vi phạm cũ trong bộ nhớ (khử trùng theo thời gian)
                self._cleanup_recent_violations()

                # Lấy ROI đèn
                roi_l_coords = clamp_roi(*ROI_LIGHT_LEFT, fw, fh)
                roi_r_coords = clamp_roi(*ROI_LIGHT_RIGHT, fw, fh)
                roi_l = frame[roi_l_coords[1]:roi_l_coords[3], roi_l_coords[0]:roi_l_coords[2]] if roi_l_coords else None
                roi_r = frame[roi_r_coords[1]:roi_r_coords[3], roi_r_coords[0]:roi_r_coords[2]] if roi_r_coords else None

                # Nhận diện màu đèn
                light_left = detect_left_light(roi_l)          # ĐÈN TRÁI: RED / GREEN
                light_right = detect_light_color(roi_r)        # ĐÈN PHẢI: RED / YELLOW / GREEN / UNKNOWN

                # Vẽ ROI & text đèn trái
                if roi_l_coords:
                    x1_l, y1_l, x2_l, y2_l = roi_l_coords
                    cv2.rectangle(
                        frame, (x1_l, y1_l), (x2_l, y2_l),
                        COLOR_MAP.get(light_left, (255, 255, 255)), 2
                    )
                    cv2.putText(
                        frame, f"LEFT: {light_left}",
                        (x1_l, y1_l - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        COLOR_MAP.get(light_left), 2
                    )

                # Vẽ ROI & text đèn phải
                if roi_r_coords:
                    x1_r, y1_r, x2_r, y2_r = roi_r_coords
                    cv2.rectangle(
                        frame, (x1_r, y1_r), (x2_r, y2_r),
                        COLOR_MAP.get(light_right, (255, 255, 255)), 2
                    )
                    cv2.putText(
                        frame, f"RIGHT: {light_right}",
                        (max(0, x1_r - 50), y1_r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        COLOR_MAP.get(light_right), 2
                    )

                # Vẽ vạch dừng theo trạng thái đèn phải
                color_vach = COLOR_MAP.get(light_right, COLOR_MAP["UNKNOWN"])
                cv2.line(
                    frame, (STOP_LINE_X1, LINE_Y), (STOP_LINE_X2, LINE_Y),
                    color_vach, LINE_THICKNESS
                )
                cv2.line(
                    frame, (STOP_LINE_X2 + 1, LINE_Y), (STOP_LINE_X3, LINE_Y),
                    color_vach, LINE_THICKNESS
                )

                # Vạch 3 (trái)
                color_v3 = COLOR_MAP.get(light_left, COLOR_MAP["UNKNOWN"])
                cv2.line(
                    frame, (LINE3_X1, LINE3_Y1), (LINE3_X2, LINE3_Y2),
                    color_v3, 3
                )
                cv2.putText(
                    frame, "Vach 3",
                    (LINE3_X1 + 5, LINE3_Y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_v3, 2
                )

                # Vạch S4 (phải)
                color_s4 = COLOR_MAP.get(light_right, COLOR_MAP["UNKNOWN"])
                cv2.line(
                    frame, (LINE_S4_X1, LINE_S4_Y1), (LINE_S4_X2, LINE_S4_Y2),
                    color_s4, 3
                )
                cv2.putText(
                    frame, "Vach S4",
                    (LINE_S4_X1 - 80, LINE_S4_Y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_s4, 2
                )

                # Chạy YOLO
                try:
                    results = self.model(frame, verbose=False)
                except Exception as e:
                    self.status_signal.emit(f"Lỗi model trên frame: {e}")
                    results = None

                # ---- Thu thập detection cho tracker ----
                detections = []
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

                        detections.append({
                            "bbox": (x1_obj, y1_obj, x2_obj, y2_obj),
                            "cx": cx,
                            "bottom_y": bottom_y,
                        })

                # ---- Tracking: gán ID cho mỗi xe ----
                tracks = self.tracker.update(detections)

                # ---- Xử lý từng track ----
                for tr in tracks:
                    track_id = tr["id"]
                    x1_obj, y1_obj, x2_obj, y2_obj = tr["bbox"]
                    cx = tr["cx"]
                    bottom_y = tr["bottom_y"]
                    bbox = (x1_obj, y1_obj, x2_obj, y2_obj)

                    # Nếu xe này đã từng vi phạm -> KHÔNG THEO DÕI NỮA
                    if track_id in self.violated_track_ids:
                        continue

                    is_violating = False
                    label_text = f"ID {track_id}"
                    color_box = (0, 255, 0)
                    lane = "unknown"

                    # Điều kiện vi phạm (giữ logic cũ)
                    if (
                        light_right == "RED"
                        and (STOP_LINE_X2 < cx <= STOP_LINE_X3)
                        and bottom_y < LINE_Y    # điều kiện vượt vạch
                    ):
                        # Kiểm tra có trùng với vi phạm đã lưu gần đây (theo vị trí + IoU) không
                        if self._recently_captured(cx, bottom_y, bbox):
                            # Đánh dấu track này là đã có vi phạm tương ứng,
                            # nhưng KHÔNG lưu thêm report mới
                            self.violated_track_ids.add(track_id)
                            continue
                        else:
                            is_violating = True
                            lane = "lane_2"
                            label_text = f"VI PHAM ID {track_id}"
                            color_box = (0, 0, 255)

                    # Vẽ bbox + label cho các track chưa bị "bỏ theo dõi"
                    cv2.rectangle(
                        frame, (x1_obj, y1_obj), (x2_obj, y2_obj),
                        color_box, 2
                    )
                    cv2.putText(
                        frame, label_text,
                        (x1_obj, max(0, y1_obj - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2
                    )

                    # Nếu vi phạm mới -> lưu CSV + ảnh, đánh dấu track_id + thêm vào recent_violations
                    if is_violating:
                        self.violated_track_ids.add(track_id)
                        self._add_recent_violation(track_id, cx, bottom_y, bbox)

                        pad_x = int((x2_obj - x1_obj) * 0.1)
                        pad_y = int((y2_obj - y1_obj) * 0.1)
                        cx1 = max(0, x1_obj - pad_x)
                        cy1 = max(0, y1_obj - pad_y)
                        cx2 = min(fw, x2_obj + pad_x)
                        cy2 = min(fh, y2_obj + pad_y)
                        if cy2 > cy1 and cx2 > cx1:
                            crop = frame[cy1:cy2, cx1:cx2].copy()
                        else:
                            crop = frame[y1_obj:y2_obj, x1_obj:x2_obj].copy()

                        try:
                            self.save_violation(
                                crop_img=crop,
                                bbox=bbox,
                                cx=cx,
                                bottom_y=bottom_y,
                                lane=lane,
                                light_right=light_right,
                                light_left=light_left
                            )
                            self.status_signal.emit(
                                f"Phát hiện vi phạm mới: id {self.violation_counter} (track {track_id})"
                            )
                        except Exception as e:
                            self.status_signal.emit(f"Lỗi lưu vi phạm: {e}")

                # Cập nhật status
                status_text = (
                    f"Đèn Trái: {light_left} | Đèn Phải: {light_right} | "
                    f"Vi phạm đã lưu: {self.violation_counter}"
                )
                self.status_signal.emit(status_text)

                # Hiển thị khung OpenCV
                cv2.imshow("Red Light Detection", frame)
                win_w, win_h = frame.shape[1], frame.shape[0]
                if win_w > screen_w or win_h > screen_h:
                    scale = min(screen_w / win_w, screen_h / win_h) * 0.7
                    try:
                        cv2.resizeWindow(
                            "Red Light Detection",
                            int(win_w * scale), int(win_h * scale)
                        )
                    except Exception:
                        pass

                # Nhấn 'q' để dừng
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


# ---------- Dialog PyQt ----------
class RedLight_violationDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚦 Nhận Diện Vượt Đèn Đỏ & Lưu Báo Cáo")
        self.setFixedSize(640, 260)

        screen_w, screen_h = get_screen_size()
        self.move(
            (screen_w - self.width()) // 2,
            (screen_h - self.height()) // 2
        )

        layout = QVBoxLayout()
        self.label = QLabel("Hệ thống đã sẵn sàng.")
        self.btn_start = QPushButton("▶ Bắt đầu (Camera)")
        self.btn_video = QPushButton("📂 Chọn video")
        self.btn_stop = QPushButton("⏹ Dừng")

        layout.addWidget(self.label)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_video)
        layout.addWidget(self.btn_stop)
        self.setLayout(layout)

        self.worker = None

        self.btn_start.clicked.connect(self.start_detect_camera)
        self.btn_video.clicked.connect(self.start_detect_video)
        self.btn_stop.clicked.connect(self.stop_detect)

        ensure_violation_dir()

    def update_status(self, text):
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
        self.update_status("Bắt đầu camera... (nhấn Q để thoát cửa sổ OpenCV)")

    def start_detect_video(self):
        if self.worker is not None and self.worker.isRunning():
            self.update_status("Đang chạy !")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_path and os.path.exists(file_path):
            self.worker = DetectWorker(source=file_path, model_path="yolov8m.pt")
            self.worker.status_signal.connect(self.update_status)
            self.worker.finished_signal.connect(self.on_finished)
            self.worker.new_violation_signal.connect(self.on_new_violation)
            self.worker.start()
            self.update_status(
                f"Bắt đầu phát hiện trên: {os.path.basename(file_path)} "
                "(nhấn Q để thoát cửa sổ OpenCV)"
            )
        else:
            self.update_status("Chưa chọn file hoặc file không tồn tại.")

    def stop_detect(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.update_status("Đang dừng... (đóng cửa sổ OpenCV nếu còn mở)")
        else:
            self.update_status("Không có quá trình nào đang chạy.")

    def on_finished(self):
        self.update_status("Đã dừng phát hiện.")

    def on_new_violation(self, info):
        # chỉ cập nhật status ngắn gọn
        self.update_status(
            f"Vi phạm mới: id {info.get('id')} - {info.get('timestamp')}"
        )


def main():
    app = QApplication(sys.argv)
    dlg = RedLight_violationDialog()
    dlg.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
