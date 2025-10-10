from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QFileDialog, QLabel
from ultralytics import YOLO
import cv2
import tkinter as tk


class DetectDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚗 Nhận diện phương tiện")
        self.setFixedSize(400, 300)

        # Layout
        layout = QVBoxLayout()
        self.label = QLabel("Chọn nguồn để bắt đầu nhận diện")
        self.btn_camera = QPushButton("📷 Mở camera")
        self.btn_file = QPushButton("📁 Mở video từ file")

        layout.addWidget(self.label)
        layout.addWidget(self.btn_camera)
        layout.addWidget(self.btn_file)
        self.setLayout(layout)

        # Nạp mô hình YOLOv8
        self.model = YOLO("yolov8n.pt")

        # Gắn sự kiện nút
        self.btn_camera.clicked.connect(self.detect_camera)
        self.btn_file.clicked.connect(self.detect_file)

    def detect_camera(self):
        self.run_detect(0)  # 0 = webcam mặc định

    def detect_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn video",
            "",
            "Video Files (*.mp4 *.avi *.mov)"
        )
        if path:
            self.run_detect(path)

    def run_detect(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.label.setText("❌ Không thể mở nguồn video")
            return

        # Tên cửa sổ
        win_name = "Nhận diện phương tiện"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 800)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        # Lấy độ phân giải màn hình để canh giữa
        root = tk.Tk()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()

        # Tính tọa độ để đặt chính giữa
        x = (screen_w - 800) // 2
        y = (screen_h - 800) // 2
        cv2.moveWindow(win_name, x, y)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Nhận diện bằng YOLO
            results = self.model(frame)
            annotated_frame = results[0].plot()

            # Resize frame để vừa cửa sổ 800x800
            annotated_frame = cv2.resize(annotated_frame, (800, 800))

            cv2.imshow(win_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
