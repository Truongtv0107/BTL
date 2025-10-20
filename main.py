import sys

from PyQt6.QtWidgets import (

    QApplication,

    QMainWindow,

    QPushButton,

    QVBoxLayout,

    QWidget,

    QLabel,

)

from PyQt6.QtGui import QFont

from PyQt6.QtCore import Qt



# ======================

# Import các module con

# ======================

from detect import DetectDialog         # Nhận diện phương tiện

from redlight import RedLightDialog     # Nhận diện đèn báo + vượt đèn đỏ

from license_plate import PlateDialog   # Nhận diện biển số xe (bạn đặt tên file là license_plate.py)

from report import ReportDialog         # Báo cáo & lưu trữ kết quả
from redlight_violation import RedLight_violationDialog #  vượt đèn đỏ




class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()



        # ======================

        # Cấu hình cửa sổ chính

        # ======================

        self.setWindowTitle("🚦 HỆ THỐNG NHẬN DIỆN VI PHẠM GIAO THÔNG")

        self.setFixedSize(700, 480)

        self.setStyleSheet("QMainWindow { background-color: #f2f6fa; }")



        # ======================

        # Layout chính

        # ======================

        main_layout = QVBoxLayout()

        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)



        # Tiêu đề

        title = QLabel("HỆ THỐNG NHẬN DIỆN VI PHẠM GIAO THÔNG")

        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))

        main_layout.addWidget(title)

        main_layout.addSpacing(25)



        # ======================

        # Tạo các nút chức năng

        # ======================

        buttons = [

            ("🚗 Nhận diện phương tiện", self.open_detect),

            ("🚦 Nhận diện đèn báo giao thông", self.open_redlight),

            ("🔢 Nhận diện biển số xe", self.open_plate),

            ("❌ Phát hiện vượt đèn đỏ", self.open_redlight_violation),

            ("📊 Lưu trữ & Báo cáo kết quả", self.open_report),

        ]



        for text, func in buttons:

            btn = self.create_button(text, func)

            main_layout.addWidget(btn)



        # ======================

        # Thiết lập container

        # ======================

        container = QWidget()

        container.setLayout(main_layout)

        self.setCentralWidget(container)



        # ======================

        # CSS cho nút bấm

        # ======================

        self.setStyleSheet("""

            QPushButton {

                background-color: #0078D7;

                color: white;

                font-size: 16px;

                font-weight: bold;

                padding: 12px;

                border-radius: 10px;

                margin: 8px 60px;

            }

            QPushButton:hover {

                background-color: #005fa3;

            }

            QPushButton:pressed {

                background-color: #003f73;

            }

        """)



    # ======================

    # Hàm tạo nút bấm

    # ======================

    def create_button(self, text, slot):

        btn = QPushButton(text)

        btn.clicked.connect(slot)

        return btn



    # ======================

    # Các hàm mở dialogq

    # ======================

    def open_detect(self):

        """Mở chức năng nhận diện phương tiện."""

        dialog = DetectDialog()

        dialog.exec()



    def open_redlight(self):

        """Mở chức năng nhận diện đèn báo giao thông."""

        dialog = RedLightDialog()

        dialog.exec()



    def open_plate(self):

        """Mở chức năng nhận diện biển số xe."""

        dialog = PlateDialog()

        dialog.exec()



    def open_redlight_violation(self):

        """Mở chức năng phát hiện vi phạm giao thông."""

        dialog = RedLight_violationDialog()  # Có thể dùng cùng dialog với redlight hoặc tách riêng file redlight_violation.py

        dialog.exec()



    def open_report(self):

        """Mở chức năng lưu trữ & báo cáo kết quả."""

        dialog = ReportDialog()

        dialog.exec()





# ======================

# Chạy chương trình

# ======================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()

    window.show()

    sys.exit(app.exec())
