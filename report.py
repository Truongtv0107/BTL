# report_dialog.py
import os
import sys
import csv
from datetime import datetime
from pathlib import Path
import shutil

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QMessageBox, QTableWidget,
    QTableWidgetItem, QFileDialog, QHBoxLayout, QLineEdit, QHeaderView, QDialogButtonBox,
    QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage

# Thư mục lưu ảnh vi phạm & file CSV
VIOLATIONS_DIR = Path("violations")
VIOLATIONS_DIR.mkdir(exist_ok=True)

# Những tên CSV có thể xuất hiện (Path objects)
POSSIBLE_CSVS = [
    VIOLATIONS_DIR / "violations_log.csv",
    VIOLATIONS_DIR / "report.csv",
    VIOLATIONS_DIR / "violations_report.csv",
    VIOLATIONS_DIR / "report_old.csv",
]

# Chọn file CSV tồn tại đầu tiên, nếu không có thì mặc định là report.csv trong thư mục violations
CSV_PATH = next((p for p in POSSIBLE_CSVS if p.exists()), VIOLATIONS_DIR / "report.csv")


class ImagePreviewDialog(QDialog):
    """Dialog nhỏ để xem ảnh thu phóng vừa phải"""
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Xem ảnh vi phạm")
        self.setMinimumSize(600, 400)
        layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.label)

        layout.addWidget(scroll)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self.setLayout(layout)
        self._load_image(image_path)

    def _load_image(self, path: str):
        """Tải ảnh từ đường dẫn. Nếu QPixmap(path) fail, đọc bằng cv2 và chuyển sang QImage."""
        if not path or not Path(path).exists():
            self.label.setText("Không tìm thấy ảnh.")
            return

        # Thử tạo QPixmap trực tiếp (tốt nhất)
        pix = QPixmap(path)
        if not pix.isNull():
            self.label.setPixmap(pix.scaled(
                self.label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            return

        # Fallback: đọc bằng OpenCV -> chuyển sang QImage -> QPixmap
        try:
            arr = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                self.label.setText("Không thể đọc ảnh.")
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.label.setPixmap(pix.scaled(
                self.label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        except Exception:
            self.label.setText("Không thể hiển thị ảnh.")

    def resizeEvent(self, event):
        pm = self.label.pixmap()
        if pm:
            self.label.setPixmap(pm.scaled(
                self.label.width(),
                self.label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        super().resizeEvent(event)


class ReportDialog(QDialog):
    """
    Dialog quản lý báo cáo: lưu ảnh + ghi CSV + hiển thị bảng.
    Public API:
      - add_violation_threadsafe(image_bgr, plate_text, violation_type, status)
      - add_violation(image_bgr, plate_text, violation_type, status)  # GUI thread
    """
    # Signal để cập nhật GUI an toàn từ thread khác
    _add_violation_signal = pyqtSignal(object, str, str, str)  # (image_np, plate, violation_type, status)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊 Báo cáo & lưu trữ kết quả")
        self.setMinimumSize(980, 560)

        self._init_ui()
        self._connect_signals()

        # ensure CSV exists with header
        if not CSV_PATH.exists():
            try:
                with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "image_path", "plate", "violation_type", "status"])
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"Không thể tạo file CSV: {e}")

        # load existing content
        self._load_csv_into_table()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("📊 BÁO CÁO KẾT QUẢ NHẬN DIỆN VI PHẠM")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Top controls: refresh, export, open folder, search, clear
        controls = QHBoxLayout()
        btn_refresh = QPushButton("🔄 Cập nhật dữ liệu")
        btn_export = QPushButton("💾 Xuất báo cáo (CSV)")
        btn_open_folder = QPushButton("📁 Mở thư mục ảnh vi phạm")
        btn_clear_all = QPushButton("🗑️ Xóa toàn bộ dữ liệu")  # CLEAR button
        btn_clear_all.setToolTip("Xóa tất cả ảnh và file CSV (yêu cầu xác nhận).")

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Tìm kiếm theo biển số / loại vi phạm...")

        controls.addWidget(btn_refresh)
        controls.addWidget(btn_export)
        controls.addWidget(btn_open_folder)
        controls.addWidget(btn_clear_all)
        controls.addStretch(1)
        controls.addWidget(QLabel("Tìm:"))
        controls.addWidget(self.search_input)

        # Bảng dữ liệu
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Thời gian", "Hình ảnh", "Biển số xe", "Loại vi phạm", "Trạng thái", "Đường dẫn ảnh"]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        header.resizeSection(0, 160)
        header.resizeSection(1, 120)
        header.resizeSection(2, 120)
        header.resizeSection(3, 180)
        header.resizeSection(4, 100)
        header.resizeSection(5, 280)

        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(self.table.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(self.table.EditTrigger.NoEditTriggers)

        layout.addWidget(title)
        layout.addSpacing(8)
        layout.addLayout(controls)
        layout.addSpacing(6)
        layout.addWidget(self.table)
        self.setLayout(layout)

        # store references to buttons for connecting signals later
        self._btn_refresh = btn_refresh
        self._btn_export = btn_export
        self._btn_open_folder = btn_open_folder
        self._btn_clear_all = btn_clear_all

    def _connect_signals(self):
        # connect buttons
        self._btn_refresh.clicked.connect(self.refresh_data)
        self._btn_export.clicked.connect(self.export_report)
        self._btn_open_folder.clicked.connect(self.open_violations_folder)
        self._btn_clear_all.clicked.connect(self.clear_all_data)  # connect clear action

        # search box
        self.search_input.textChanged.connect(self._apply_search_filter)

        # double click -> open image or preview
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)

        # internal signal (thread-safe add)
        self._add_violation_signal.connect(self._handle_add_violation)

    # ---------- Public API ----------
    def add_violation(self, image_bgr, plate_text="", violation_type="Vượt đèn đỏ", status="Đã lưu"):
        """Gọi từ GUI thread: lưu ảnh & cập nhật bảng"""
        self._save_and_add_row(image_bgr, plate_text, violation_type, status)

    def add_violation_threadsafe(self, image_bgr, plate_text="", violation_type="Vượt đèn đỏ", status="Đã lưu"):
        """Gọi từ thread khác: phát tín hiệu để GUI cập nhật an toàn"""
        self._add_violation_signal.emit(image_bgr, plate_text, violation_type, status)

    # ---------- Internal handlers ----------
    def _handle_add_violation(self, image_bgr, plate_text, violation_type, status):
        try:
            self._save_and_add_row(image_bgr, plate_text, violation_type, status)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi lưu vi phạm", f"Lỗi khi lưu vi phạm: {e}")

    def _save_and_add_row(self, image_bgr, plate_text, violation_type, status):
        """
        Lưu ảnh vi phạm vào thư mục, ghi file CSV log, cập nhật table.
        image_bgr: numpy array (BGR)
        """
        # timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plate_safe = (plate_text or "unknown").replace("/", "_").replace("\\", "_").replace(" ", "")
        filename = f"{ts}_{plate_safe}.jpg"
        file_path = VIOLATIONS_DIR / filename

        # Lưu file ảnh (sử dụng imencode + tofile để hỗ trợ Unicode paths trên Windows)
        saved = False
        try:
            encoded = cv2.imencode(".jpg", image_bgr)[1]
            encoded.tofile(str(file_path))
            saved = True
        except Exception:
            try:
                saved = cv2.imwrite(str(file_path), image_bgr)
            except Exception:
                saved = False

        if not saved:
            raise IOError("Không thể lưu file ảnh vi phạm.")

        # Ghi log CSV (append) - nếu file chưa có header thì tạo header
        header_needed = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
        try:
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if header_needed:
                    writer.writerow(["timestamp", "image_path", "plate", "violation_type", "status"])
                writer.writerow([datetime.now().isoformat(sep=" "), str(file_path), plate_text, violation_type, status])
        except Exception as e:
            QMessageBox.warning(self, "Lỗi ghi CSV", f"Không thể ghi file CSV: {e}")

        # Cập nhật bảng GUI (thêm 1 dòng)
        row_idx = self.table.rowCount()
        self.table.insertRow(row_idx)

        self.table.setItem(row_idx, 0, QTableWidgetItem(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        # Hiển thị text "Xem ảnh" cho ảnh (double click sẽ mở)
        self.table.setItem(row_idx, 1, QTableWidgetItem("Xem ảnh"))
        self.table.setItem(row_idx, 2, QTableWidgetItem(plate_text))
        self.table.setItem(row_idx, 3, QTableWidgetItem(violation_type))
        self.table.setItem(row_idx, 4, QTableWidgetItem(status))
        self.table.setItem(row_idx, 5, QTableWidgetItem(str(file_path)))

        # chọn và tỏa sáng dòng mới
        self.table.selectRow(row_idx)

    # ---------- Other UI functions ----------
    def refresh_data(self):
        """Tải lại từ CSV"""
        if not CSV_PATH.exists():
            QMessageBox.information(self, "Cập nhật", "Chưa có dữ liệu vi phạm để tải.")
            return
        self._load_csv_into_table()
        QMessageBox.information(self, "Cập nhật", "Dữ liệu đã được tải lại thành công!")

    def _load_csv_into_table(self):
        """Đọc CSV và hiển thị lên bảng"""
        self.table.setRowCount(0)
        try:
            with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r = self.table.rowCount()
                    self.table.insertRow(r)
                    self.table.setItem(r, 0, QTableWidgetItem(row.get("timestamp", "")))
                    self.table.setItem(r, 1, QTableWidgetItem("Xem ảnh"))
                    self.table.setItem(r, 2, QTableWidgetItem(row.get("plate", "")))
                    self.table.setItem(r, 3, QTableWidgetItem(row.get("violation_type", "")))
                    self.table.setItem(r, 4, QTableWidgetItem(row.get("status", "")))
                    self.table.setItem(r, 5, QTableWidgetItem(row.get("image_path", "")))
        except Exception as e:
            QMessageBox.critical(self, "Lỗi đọc CSV", f"Không thể đọc file CSV: {e}")

    def export_report(self):
        """Xuất file CSV (cho Excel)"""
        if not CSV_PATH.exists():
            QMessageBox.information(self, "Xuất báo cáo", "Không có dữ liệu để xuất.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Lưu báo cáo CSV", "violations_report.csv", "CSV Files (*.csv)")
        if not save_path:
            return
        try:
            # copy CSV_PATH -> save_path (binary-safe)
            with open(CSV_PATH, "r", encoding="utf-8") as src, open(save_path, "w", encoding="utf-8", newline="") as dst:
                dst.write(src.read())
            QMessageBox.information(self, "Xuất báo cáo", f"Đã xuất báo cáo tới:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi xuất báo cáo", f"Lỗi khi xuất CSV: {e}")

    def open_violations_folder(self):
        """Mở thư mục chứa ảnh vi phạm bằng trình quản lý file OS"""
        folder = str(VIOLATIONS_DIR.resolve())
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)
            elif sys.platform == "darwin":
                # macOS
                os.system(f'open "{folder}"')
            else:
                # linux
                os.system(f'xdg-open "{folder}"')
        except Exception:
            QMessageBox.information(self, "Mở thư mục", f"Thư mục ảnh: {folder}")

    def clear_all_data(self):
        """
        Xóa toàn bộ ảnh trong VIOLATIONS_DIR và reset CSV_PATH.
        Yêu cầu xác nhận người dùng trước khi thao tác.
        """
        reply = QMessageBox.question(
            self,
            "Xác nhận xóa toàn bộ dữ liệu",
            "Bạn có chắc chắn muốn xóa tất cả ảnh vi phạm và báo cáo CSV không?\nHành động này không thể hoàn tác.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        errors = []
        # 1) Xóa tất cả file ảnh trong thư mục violations (giữ nguyên thư mục)
        try:
            for item in VIOLATIONS_DIR.iterdir():
                try:
                    if item.is_file():
                        # only delete image files and CSVs inside the folder
                        if item.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".csv", ".gif"]:
                            item.unlink()
                    # if it's a directory, skip
                except Exception as e:
                    errors.append(f"Không xóa được {item.name}: {e}")
        except Exception as e:
            errors.append(f"Lỗi truy cập thư mục {VIOLATIONS_DIR}: {e}")

        # 2) Reset CSV file (create empty with header)
        try:
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "image_path", "plate", "violation_type", "status"])
        except Exception as e:
            errors.append(f"Lỗi tạo file CSV: {e}")

        # 3) Refresh table UI (clear rows)
        try:
            self.table.setRowCount(0)
        except Exception as e:
            errors.append(f"Lỗi cập nhật bảng: {e}")

        # 4) Show result
        if errors:
            QMessageBox.warning(self, "Xóa dữ liệu - hoàn tất (có lỗi)", "Hoàn tất nhưng có một số lỗi:\n" + "\n".join(errors))
        else:
            QMessageBox.information(self, "Xóa dữ liệu", "Đã xóa tất cả ảnh và reset báo cáo CSV thành công.")

    def _apply_search_filter(self, text: str):
        """Lọc bảng theo chuỗi tìm kiếm (biển số, loại vi phạm hoặc đường dẫn)"""
        txt = text.strip().lower()
        for r in range(self.table.rowCount()):
            plate_item = self.table.item(r, 2)
            type_item = self.table.item(r, 3)
            path_item = self.table.item(r, 5)
            joined = " ".join([
                (plate_item.text() if plate_item else "").lower(),
                (type_item.text() if type_item else "").lower(),
                (path_item.text() if path_item else "").lower()
            ])
            match = txt in joined
            self.table.setRowHidden(r, not match)

    def _on_cell_double_clicked(self, row: int, col: int):
        """Double click: nếu có đường dẫn ảnh -> mở preview và/hoặc mở file bằng default viewer"""
        path_item = self.table.item(row, 5)
        if not path_item:
            return
        img_path = path_item.text().strip()
        if not img_path:
            QMessageBox.information(self, "Ảnh không tồn tại", "Dòng này không có đường dẫn ảnh.")
            return

        p = Path(img_path)
        if not p.exists():
            QMessageBox.warning(self, "Không tìm thấy ảnh", f"Không tìm thấy file ảnh:\n{img_path}")
            return

        # 1) Show in-app preview dialog
        try:
            dlg = ImagePreviewDialog(str(p), parent=self)
            dlg.exec()
        except Exception:
            # ignore preview errors, continue to open external viewer
            pass

        # 2) Try open with OS default viewer (user may expect this)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))
            elif sys.platform == "darwin":
                os.system(f'open "{p}"')
            else:
                os.system(f'xdg-open "{p}"')
        except Exception:
            QMessageBox.information(self, "Mở ảnh", f"Ảnh đã được xem trong ứng dụng.\nĐường dẫn: {img_path}")
