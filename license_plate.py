import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox,
    QApplication, QHBoxLayout, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont
from ultralytics import YOLO
import easyocr


class PlateDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔢 Nhận diện biển số xe")
        self.setFixedSize(1200, 650)

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("🔲 NHẬN DIỆN BIỂN SỐ XE (YOLO + OCR)")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        main_layout.addSpacing(10)

        # Hai vùng hiển thị song song
        hbox = QHBoxLayout()

        # Khung hiển thị ảnh/video
        self.label_original = QLabel()
        self.label_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_original.setStyleSheet("border: 2px dashed gray; padding: 10px; background-color: #000;")
        self.label_original.setFixedSize(550, 420)

        # Khung hiển thị kết quả text
        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        self.text_result.setStyleSheet("""
            QTextEdit {
                border: 2px dashed gray;
                color: red;
                background-color: #fff;
                padding: 10px;
            }
        """)
        self.text_result.setFixedSize(550, 420)
        self.text_result.setPlaceholderText("Kết quả biển số sẽ hiển thị ở đây...")

        hbox.addWidget(self.label_original)
        hbox.addWidget(self.text_result)
        main_layout.addLayout(hbox)

        # Nút mở file
        main_layout.addSpacing(20)
        self.btn_open = QPushButton("📂 Chọn ảnh / video chứa biển số")
        self.btn_open.clicked.connect(self.open_file)
        main_layout.addWidget(self.btn_open)

        self.setLayout(main_layout)

        # --- Khởi tạo YOLO & OCR ---
        # Bạn có thể thay bằng model chuyên biển số (nếu có): "yolov8n-license.pt"
        self.model = YOLO("license_plate_detector.pt")
        self.reader = easyocr.Reader(['en'])

        # Video
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.detected_plates = set()

    # --- Mở file ---
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file", "", "Ảnh / Video (*.mp4 *.avi *.jpg *.png)"
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[-1].lower()
        self.detected_plates.clear()
        self.text_result.clear()

        if ext in [".jpg", ".png"]:
            self.process_image(file_path)
        else:
            self.play_video(file_path)

    # --- Xử lý ảnh ---
    def process_image(self, file_path):
        img = cv2.imread(file_path)
        results = self.model(img)
        plates = []

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                crop = img[y1:y2, x1:x2]
                text = self.ocr_plate(crop)
                if text:
                    plates.append(text)
                    # vẽ khung và text
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)

        self.display_cv_image(img, self.label_original)

        if plates:
            self.text_result.setPlainText("\n".join(plates))
        else:
            self.text_result.setPlainText("Không phát hiện được biển số nào.")

    #  Xử lý video 
    def play_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Lỗi", "Không thể mở video.")
            return
        self.timer.start(30)

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        results = self.model(frame)
        new_plates = set()

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                text = self.ocr_plate(crop)
                if text:
                    new_plates.add(text)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)

        self.detected_plates |= new_plates
        self.text_result.setPlainText("\n".join(sorted(self.detected_plates)))

        self.display_cv_image(frame, self.label_original)

    # --- Hiển thị ảnh ---
    def display_cv_image(self, cv_img, label):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(pixmap)

    # --- OCR ---
    def ocr_plate(self, image):
        if image is None or image.size == 0:
            return None
        results = self.reader.readtext(image)
        if results:
            return " ".join([res[1] for res in results])
        return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = PlateDialog()
    dlg.show()
    sys.exit(app.exec())
