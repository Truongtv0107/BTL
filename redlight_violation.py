import sys
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QFileDialog
from ultralytics import YOLO
import cv2
import os
import tkinter as tk
import numpy as np

# Cấu hình chung
TARGET_W, TARGET_H = 1280, 720

# Vùng Quan Tâm (ROI) cho Đèn Giao Thông
# ROI Đèn Trái (giữ nguyên phương pháp cộng trừ)
ROI_LIGHT_LEFT = (21 - 15, 108 - 35, 21 + 15, 108 + 35) 
# ROI Đèn Phải (chốt theo tọa độ cuối cùng bạn cung cấp: 1242, 34, 1272, 104)
ROI_LIGHT_RIGHT = (1242, 34, 1272, 104)

LINE_THICKNESS = 12

# Tọa độ Vạch Dừng (Stop Line)
STOP_LINE_X1 = 89
STOP_LINE_X2 = 459 
STOP_LINE_X3 = 1086
STOP_LINE_Y_HEIGHT = 400 # Đã điều chỉnh xuống 400 để khớp bánh xe

LINE_Y = STOP_LINE_Y_HEIGHT
X_DIV = STOP_LINE_X2

# Tọa độ các vạch kẻ thêm
LINE3_X1, LINE3_Y1 = 73, 401
LINE3_X2, LINE3_Y2 = 352, 83
COLOR_LINE3 = (255, 0, 255)

LINE_S4_X1, LINE_S4_Y1 = 1123, 370
LINE_S4_X2, LINE_S4_Y2 = 1005, 81

# Bản đồ màu sắc (BGR)
COLOR_MAP = {
    "RED": (0, 0, 255), 
    "GREEN": (0, 255, 0), 
    "YELLOW": (0, 255, 255),
    "UNKNOWN": (255, 255, 255)
}

def get_screen_size():
    """Lấy kích thước màn hình để đặt cửa sổ hiển thị"""
    root = tk.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()


class RedLight_violationDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚦 Nhận Diện Vượt Đèn Đỏ & Vẽ Vạch Cố Định")
        self.setFixedSize(600, 300)

        screen_w, screen_h = get_screen_size()
        self.move((screen_w - self.width()) // 2, (screen_h - self.height()) // 2)

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

        try:
             # CẬP NHẬT: Dùng mô hình YOLOv8m để tăng cường nhận diện
             self.model = YOLO("yolov8m.pt") 
        except Exception as e:
             self.label.setText(f"Lỗi tải YOLO model: {e}")
             self.model = None

        self.running = False
        self.light_state_left = "UNKNOWN"
        self.light_state_right = "UNKNOWN"
        
        self.btn_start.clicked.connect(self.start_detect_camera)
        self.btn_video.clicked.connect(self.start_detect_video)
        self.btn_stop.clicked.connect(self.stop_detect)
        
        self.update_status_label()

    def get_light_color_from_roi(self, roi, is_right_light=False):
        """Xác định màu đèn từ vùng ROI"""
        if roi is None or roi.size == 0:
            return "UNKNOWN"
        
        b, g, r = roi.mean(axis=(0, 1))
        
        # Logic cho Đèn Bên Phải (chỉ Xanh hoặc Đỏ, mặc định là Đỏ)
        if is_right_light:
            if g > r * 1.5 and g > 80:
                return "GREEN"
            else:
                return "RED" 
        
        # Logic cho Đèn Bên Trái (3 màu)
        else:
            if r > g * 1.5 and r > 80:
                return "RED"
            elif g > r * 1.5 and g > 80:
                return "GREEN"
            elif r > 100 and g > 100 and abs(r - g) < 60 and b < 80:
                return "YELLOW"
            else:
                return "UNKNOWN"


    def update_status_label(self):
        """Cập nhật trạng thái hiển thị"""
        text = (f"Trạng thái Đèn Trái: **{self.light_state_left}**\n"
                f"Trạng thái Đèn Phải: **{self.light_state_right}**\n\n"
                f"Tọa độ Vạch (Stop Line): V1({STOP_LINE_X1},{LINE_Y})->({STOP_LINE_X2},{LINE_Y}) (Theo Đèn PHẢI), V2({STOP_LINE_X2+1},{LINE_Y})->({STOP_LINE_X3},{LINE_Y}) (Theo Đèn PHẢI)\n"
                f"Tọa độ Vạch 3 (Kẻ thêm): ({LINE3_X1},{LINE3_Y1})->({LINE3_X2},{LINE3_Y2}) (Theo màu đèn trái)\n"
                f"Tọa độ Vạch S4 (Kẻ thêm): ({LINE_S4_X1},{LINE_S4_Y1})->({LINE_S4_X2},{LINE_S4_Y2}) (Theo màu đèn phải)"
                )
        self.label.setText(text)

    def detect(self, cap):
        """Vòng lặp phát hiện chính"""
        if not cap.isOpened():
            self.label.setText("❌ Lỗi: Không thể mở video/camera. Vui lòng kiểm tra kết nối thiết bị (index 0).")
            self.running = False
            return
        
        if self.model is None:
            self.label.setText("❌ Lỗi: Model YOLO chưa được tải.")
            self.running = False
            return

        screen_w, screen_h = get_screen_size()
        self.running = True

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (TARGET_W, TARGET_H))
                frame_h, frame_w = frame.shape[:2]

                # Nhận diện Đèn
                x1_l, y1_l, x2_l, y2_l = ROI_LIGHT_LEFT
                roi_l = frame[y1_l:y2_l, x1_l:x2_l]
                self.light_state_left = self.get_light_color_from_roi(roi_l, is_right_light=False)
                
                x1_r, y1_r, x2_r, y2_r = ROI_LIGHT_RIGHT
                roi_r = frame[y1_r:y2_r, x1_r:x2_r]
                self.light_state_right = self.get_light_color_from_roi(roi_r, is_right_light=True)

                # Vẽ đèn và vạch
                color_map = COLOR_MAP
                
                color_l = color_map.get(self.light_state_left)
                cv2.rectangle(frame, (x1_l, y1_l), (x2_l, y2_l), color_l, 2)
                cv2.putText(frame, f"LEFT: {self.light_state_left}", (x1_l, y1_l - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_l, 2)

                color_r = color_map.get(self.light_state_right)
                cv2.rectangle(frame, (x1_r, y1_r), (x2_r, y2_r), color_r, 2)
                cv2.putText(frame, f"RIGHT: {self.light_state_right}", (x1_r - 50, y1_r - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_r, 2)
                
                # Vẽ Vạch Dừng (Stop Line) theo đèn bên phải
                color_vach1 = COLOR_MAP.get(self.light_state_right) 
                if color_vach1:
                    cv2.line(frame, (STOP_LINE_X1, LINE_Y), (STOP_LINE_X2, LINE_Y), color_vach1, LINE_THICKNESS)
                    
                color_vach2 = COLOR_MAP.get(self.light_state_right)
                if color_vach2:
                    cv2.line(frame, (STOP_LINE_X2 + 1, LINE_Y), (STOP_LINE_X3, LINE_Y), color_vach2, LINE_THICKNESS)
                
                # Vẽ vạch 3 theo đèn bên trái
                color_vach3 = COLOR_MAP.get(self.light_state_left)
                if color_vach3:
                    cv2.line(frame, (LINE3_X1, LINE3_Y1), (LINE3_X2, LINE3_Y2), color_vach3, 3)
                    cv2.putText(frame, "Vach 3", (LINE3_X1 + 5, LINE3_Y1 - 5), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_vach3, 2)
                
                # Vẽ vạch S4 theo đèn bên phải
                color_vachS4 = COLOR_MAP.get(self.light_state_right)
                if color_vachS4:
                    cv2.line(frame, (LINE_S4_X1, LINE_S4_Y1), (LINE_S4_X2, LINE_S4_Y2), color_vachS4, 3)
                    cv2.putText(frame, "Vach S4", (LINE_S4_X1 - 80, LINE_S4_Y1 + 15), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_vachS4, 2)

                # Chạy mô hình YOLO và kiểm tra Vi phạm
                results = self.model(frame, verbose=False) 
                vehicle_classes = [2, 3, 5, 7] # car, motorbike, bus, truck

                for box in results[0].boxes:
                    cls = int(box.cls)
                    if cls in vehicle_classes:
                        x1_obj, y1_obj, x2_obj, y2_obj = map(int, box.xyxy[0].tolist())
                        
                        bottom_y = y2_obj # Cạnh dưới của phương tiện
                        cx = (x1_obj + x2_obj) // 2 # Tâm ngang của phương tiện
                        
                        label_text = "hop le" 
                        color_box = (0, 255, 0)
                        is_violating = False
                        
                        # Điều kiện Vi phạm Vượt Vạch: Đèn Đỏ VÀ bottom_y > LINE_Y (đã vượt qua vạch)
                        
                        # Làn 1 
                        # if self.light_state_right == "RED" and STOP_LINE_X1 <= cx <= STOP_LINE_X2 and bottom_y < LINE_Y:
                        #     is_violating = True
                        #     label_text = "VI PHAM !" 
                        
                        # Làn 2
                        if self.light_state_right == "RED" and STOP_LINE_X2 < cx <= STOP_LINE_X3 and bottom_y < LINE_Y:
                            is_violating = True
                            label_text = "VI PHAM !" 

                        if is_violating:
                            color_box = (0, 0, 255) # Màu đỏ nếu vi phạm

                        cv2.rectangle(frame, (x1_obj, y1_obj), (x2_obj, y2_obj), color_box, 2)
                        cv2.putText(frame, label_text, (x1_obj, y1_obj - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)
                        
                        
                self.update_status_label()

                cv2.imshow("Red Light Detection", frame)

                win_w, win_h = frame_w, frame_h
                if win_w > screen_w or win_h > screen_h:
                    scale = min(screen_w / win_w, screen_h / win_h) * 0.7 
                    win_w, win_h = int(win_w * scale), int(win_h * scale)
                cv2.resizeWindow("Red Light Detection", win_w, win_h)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Lỗi trong quá trình xử lý video/khung hình: {e}")
            self.label.setText(f"❌ Lỗi xử lý: {e}. Đã dừng phát hiện.")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.running = False
            self.light_state_left = "Stopped"
            self.light_state_right = "Stopped"
            self.update_status_label()

    def start_detect_camera(self):
        if not self.running:
            cap = cv2.VideoCapture(0)
            self.detect(cap)

    def start_detect_video(self):
        if not self.running:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Chọn video", "", "Video Files (*.mp4 *.avi *.mov)"
            )
            if file_path and os.path.exists(file_path):
                cap = cv2.VideoCapture(file_path)
                self.detect(cap)

    def stop_detect(self):
        self.running = False
        self.light_state_left = "Stopped"
        self.light_state_right = "Stopped"
        self.update_status_label()