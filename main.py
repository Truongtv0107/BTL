import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Import các trang con
from detect import DetectDialog
from redlight import RedLightDialog
from speed import SpeedDialog
from helmet import HelmetDialog
from report import ReportDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Cấu hình cửa sổ chính
        self.setWindowTitle("🚦 Hệ thống nhận diện vi phạm giao thông")
        self.setFixedSize(600, 400)

        # Layout chính
        layout = QVBoxLayout()

        # Các nút chức năng
        self.btn_detect = self.create_button("🚗 Nhận diện phương tiện", self.open_detect)
        self.btn_redlight = self.create_button("🚦 Vượt đèn đỏ", self.open_redlight)
        self.btn_speed = self.create_button("💨 Tốc độ", self.open_speed)
        self.btn_helmet = self.create_button("🪖 Không đội mũ bảo hiểm", self.open_helmet)
        self.btn_report = self.create_button("📊 Báo cáo", self.open_report)

        # Thêm nút vào layout
        layout.addWidget(self.btn_detect)
        layout.addWidget(self.btn_redlight)
        layout.addWidget(self.btn_speed)
        layout.addWidget(self.btn_helmet)
        layout.addWidget(self.btn_report)

        # Container chính
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Thêm CSS
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f4f6f9;
            }
            QPushButton {
                background-color: #2e86de;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
                border-radius: 12px;
                margin: 8px;
            }
            QPushButton:hover {
                background-color: #1e5fab;
            }
            QPushButton:pressed {
                background-color: #163d73;
            }
        """)

    def create_button(self, text, slot):
        """Hàm tạo QPushButton kèm sự kiện."""
        btn = QPushButton(text)
        btn.clicked.connect(slot)
        return btn

    # ================== Các hàm mở dialog ==================
    def open_detect(self):
        DetectDialog().exec()

    def open_redlight(self):
        RedLightDialog().exec()

    def open_speed(self):
        SpeedDialog().exec()

    def open_helmet(self):
        HelmetDialog().exec()

    def open_report(self):
        ReportDialog().exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
