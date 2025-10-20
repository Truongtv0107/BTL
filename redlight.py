import sys
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QFileDialog
from ultralytics import YOLO
import cv2
import os
import tkinter as tk
import numpy as np

# ==============================================================================
# THÔNG SỐ CỐ ĐỊNH VÀ ROI ĐÈN ĐÃ CẬP NHẬT
# ==============================================================================
TARGET_W, TARGET_H = 1280, 720  # Kích thước chuẩn hóa video

# ⚠️ TỌA ĐỘ ROI ĐÈN ĐÃ CẬP NHẬT (Xung quanh các điểm bạn cung cấp)
# Đèn bên trái: Trung tâm ~ (21, 108). Chọn vùng 30x70 pixel xung quanh.
# ROI [x1, y1, x2, y2]
ROI_LIGHT_LEFT = (21 - 15, 108 - 35, 21 + 15, 108 + 35) # (6, 73, 36, 143)

# Đèn bên phải: Trung tâm ~ (1261, 98). Chọn vùng 30x70 pixel xung quanh.
ROI_LIGHT_RIGHT = (1261 - 15, 98 - 35, 1261 + 15, 98 + 35) # (1246, 63, 1276, 133)
# Cần đảm bảo x2 không vượt quá 1280
ROI_LIGHT_RIGHT = (1246, 63, 1279, 133) 


def get_screen_size():
    """Lấy kích thước màn hình chính"""
    root = tk.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()


class RedLightDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚦 Nhận Diện Đèn Giao Thông")
        self.setFixedSize(600, 300)

        screen_w, screen_h = get_screen_size()
        self.move((screen_w - self.width()) // 2, (screen_h - self.height()) // 2)

        layout = QVBoxLayout()
        self.label = QLabel("Chức năng chỉ nhận diện và hiển thị trạng thái đèn.")
        self.btn_start = QPushButton("▶ Bắt đầu (Camera)")
        self.btn_video = QPushButton("📂 Chọn video")
        self.btn_stop = QPushButton("⏹ Dừng")
        
        layout.addWidget(self.label)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_video)
        layout.addWidget(self.btn_stop)
        
        self.setLayout(layout)

        # Load YOLO model
        self.model = YOLO("yolov8n.pt")

        # Trạng thái
        self.running = False
        self.light_state_left = "UNKNOWN"
        self.light_state_right = "UNKNOWN"
        
        self.btn_start.clicked.connect(self.start_detect_camera)
        self.btn_video.clicked.connect(self.start_detect_video)
        self.btn_stop.clicked.connect(self.stop_detect)
        
        self.update_status_label()

    def get_light_color_from_roi(self, roi):
        """Xác định màu của vùng ROI đèn giao thông dựa trên giá trị BGR trung bình"""
        if roi is None or roi.size == 0:
            return "UNKNOWN"
        
        b, g, r = roi.mean(axis=(0, 1))
        
        if r > g * 1.5 and r > 80:
            return "RED"
        elif g > r * 1.5 and g > 80:
            return "GREEN"
        elif r > 100 and g > 100 and abs(r - g) < 60:
            return "YELLOW"
        else:
            return "UNKNOWN"

    def update_status_label(self):
        """Cập nhật trạng thái đèn trên cửa sổ PyQt"""
        text = (f"Trạng thái Đèn Trái: **{self.light_state_left}**\n"
                f"Trạng thái Đèn Phải: **{self.light_state_right}**\n\n"
                f"Chọn nguồn để bắt đầu nhận diện.")
        self.label.setText(text)

    def detect(self, cap):
        if not cap.isOpened():
            self.label.setText("❌ Lỗi: Không thể mở video/camera")
            return

        screen_w, screen_h = get_screen_size()
        self.running = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Chuẩn hóa kích thước khung hình
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))
            frame_h, frame_w = frame.shape[:2]

            # 2. Lấy ROI và Xác định màu đèn (CHỈ SỬ DỤNG CÁC ROI CỐ ĐỊNH)
            
            # Đèn Trái
            x1_l, y1_l, x2_l, y2_l = ROI_LIGHT_LEFT
            roi_l = frame[y1_l:y2_l, x1_l:x2_l]
            self.light_state_left = self.get_light_color_from_roi(roi_l)
            
            # Đèn Phải
            x1_r, y1_r, x2_r, y2_r = ROI_LIGHT_RIGHT
            roi_r = frame[y1_r:y2_r, x1_r:x2_r]
            self.light_state_right = self.get_light_color_from_roi(roi_r)

            # 3. Vẽ khung ROI và Label lên khung hình (dùng màu cho dễ debug)
            color_map = {"RED": (0, 0, 255), "GREEN": (0, 255, 0), "YELLOW": (0, 255, 255), "UNKNOWN": (100, 100, 100)}
            
            # Vẽ đèn Trái
            color_l = color_map.get(self.light_state_left)
            cv2.rectangle(frame, (x1_l, y1_l), (x2_l, y2_l), color_l, 2)
            cv2.putText(frame, f"LEFT: {self.light_state_left}", (x1_l, y1_l - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_l, 2)

            # Vẽ đèn Phải
            color_r = color_map.get(self.light_state_right)
            cv2.rectangle(frame, (x1_r, y1_r), (x2_r, y2_r), color_r, 2)
            cv2.putText(frame, f"RIGHT: {self.light_state_right}", (x1_r - 50, y1_r - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_r, 2)


            # 4. Nhận diện các đối tượng khác bằng YOLO (nhưng không xử lý vi phạm)
            results = self.model(frame, verbose=False)
            
            # 5. Cập nhật trạng thái trên dialog
            self.update_status_label()

            # 6. Hiển thị video
            cv2.imshow("Traffic Light Detection", frame)

            # Điều chỉnh cửa sổ theo kích thước màn hình
            win_w, win_h = frame_w, frame_h
            if win_w > screen_w or win_h > screen_h:
                 scale = min(screen_w / win_w, screen_h / win_h) * 0.7 
                 win_w, win_h = int(win_w * scale), int(win_h * scale)
            cv2.resizeWindow("Traffic Light Detection", win_w, win_h)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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