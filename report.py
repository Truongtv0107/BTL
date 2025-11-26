# report_dialog.py
import os
import csv
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QMessageBox,
    QTableWidget, QTableWidgetItem, QFileDialog, QHBoxLayout,
    QScrollArea, QWidget, QDialogButtonBox, QLineEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import cv2

# ---------------- Config ----------------
#by Truong Viet Tran , do not reup ,sdt:0877973723
VIOLATIONS_DIR = Path("violations")
VIOLATIONS_DIR.mkdir(parents=True, exist_ok=True)

# 2 file CSV giống với code YOLO mới
# status.csv: track_id, ngay_vi_pham, loai_vi_pham, tinh_trang
# report.csv: id, timestamp, image_path, ..., light_left, track_id
STATUS_CSV = VIOLATIONS_DIR / "status.csv"
REPORT_CSV = VIOLATIONS_DIR / "report.csv"


# ---------------- Edit Dialog (Thêm/Sửa) ----------------
class ViolationEditDialog(QDialog):
    """
    Dialog dùng chung cho THÊM / SỬA 1 dòng trong status.csv
    data: dict {"track_id", "ngay_vi_pham", "loai_vi_pham", "tinh_trang"}
    """
    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chỉnh sửa vi phạm")
        self.setMinimumSize(400, 200)

        self._data = data

        layout = QVBoxLayout(self)

        # TRACK_ID (chỉ hiển thị, không cho sửa)
        self.lbl_track_id = QLabel(f"Tracking ID: <b>{data.get('track_id', '')}</b>")
        layout.addWidget(self.lbl_track_id)

        # Ngày vi phạm
        layout.addWidget(QLabel("Ngày vi phạm:"))
        self.ed_date = QLineEdit(data.get("ngay_vi_pham", ""))
        layout.addWidget(self.ed_date)

        # Loại vi phạm
        layout.addWidget(QLabel("Loại vi phạm:"))
        self.ed_violation_type = QLineEdit(data.get("loai_vi_pham", "Vượt đèn đỏ"))
        layout.addWidget(self.ed_violation_type)

        # Tình trạng
        layout.addWidget(QLabel("Tình trạng:"))
        self.ed_status = QLineEdit(data.get("tinh_trang", "Chờ xử lý"))
        layout.addWidget(self.ed_status)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_row(self):
        """Trả về list [track_id, ngay_vi_pham, loai_vi_pham, tinh_trang]"""
        return [
            self._data.get("track_id", ""),
            self.ed_date.text().strip(),
            self.ed_violation_type.text().strip() or "Vượt đèn đỏ",
            self.ed_status.text().strip() or "Chờ xử lý",
        ]


# ---------------- Detail Dialog ----------------
class ViolationDetailDialog(QDialog):
    """Hiển thị chi tiết vi phạm + ảnh"""
    def __init__(self, info: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chi tiết vi phạm")
        self.setMinimumSize(600, 500)
        layout = QVBoxLayout(self)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll.setWidget(container)
        container_layout = QVBoxLayout(container)
        layout.addWidget(scroll)

        # Thông tin chi tiết
        info_text = (
            f"<b>Tracking ID:</b> {info.get('track_id','')}<br>"
            f"<b>Ngày vi phạm:</b> {info.get('date','')}<br>"
            f"<b>Loại vi phạm:</b> {info.get('violation_type','Vượt đèn đỏ')}<br>"
            f"<b>Tình trạng:</b> {info.get('note','Chờ xử lý')}<br>"
            f"<b>Đường dẫn ảnh:</b> {info.get('image_path','')}"
        )
        label = QLabel(info_text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignTop)
        container_layout.addWidget(label)

        # Hiển thị ảnh
        img_path = info.get('image_path', '')
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if img_path and Path(img_path).exists():
            pixmap = QPixmap()
            if pixmap.load(str(img_path)):
                img_label.setPixmap(
                    pixmap.scaled(
                        500,
                        400,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                img_label.setText("Không thể mở ảnh.")
        else:
            img_label.setText("Không có ảnh vi phạm.")
        container_layout.addWidget(img_label)

        # Close button
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        container_layout.addWidget(buttons)


# ---------------- Main Report Dialog ----------------
class ReportDialog(QDialog):
    """Dialog quản lý báo cáo vi phạm vượt đèn đỏ (đọc từ status.csv + report.csv)"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊 Báo cáo vi phạm - Vượt đèn đỏ")
        self.setMinimumSize(600, 400)
        self._init_ui()
        self._ensure_csv_files()
        self._load_status_into_table()

    # ---------- UI ----------
    def _init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("📊 BÁO CÁO VI PHẠM - Vượt đèn đỏ")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Controls (trên cùng)
        controls = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 Cập nhật")
        self.btn_export = QPushButton("💾 Xuất CSV (tóm tắt)")
        self.btn_clear_all = QPushButton("🗑️ Xóa toàn bộ")
        controls.addWidget(self.btn_refresh)
        controls.addWidget(self.btn_export)
        controls.addWidget(self.btn_clear_all)
        controls.addStretch(1)
        layout.addLayout(controls)

        # Controls cho thêm/sửa/xóa 1 dòng
        row_controls = QHBoxLayout()
        self.btn_add = QPushButton("➕ Thêm")
        self.btn_edit = QPushButton("✏️ Sửa")
        self.btn_delete = QPushButton("❌ Xóa dòng")
        row_controls.addWidget(self.btn_add)
        row_controls.addWidget(self.btn_edit)
        row_controls.addWidget(self.btn_delete)
        row_controls.addStretch(1)
        layout.addLayout(row_controls)

        # Table: lấy dữ liệu từ status.csv
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Tracking ID", "Ngày vi phạm", "Loại vi phạm", "Tình trạng"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        # Kết nối sự kiện
        self.btn_refresh.clicked.connect(self.refresh_data)
        self.btn_export.clicked.connect(self.export_report)
        self.btn_clear_all.clicked.connect(self.clear_all_data)
        self.table.cellDoubleClicked.connect(self.show_detail)

        self.btn_add.clicked.connect(self.add_row)
        self.btn_edit.clicked.connect(self.edit_row)
        self.btn_delete.clicked.connect(self.delete_row)

    # ---------- CSV setup ----------
    def _ensure_csv_files(self):
        """Đảm bảo tồn tại status.csv và report.csv với header chuẩn (giống code YOLO)."""
        # status.csv: bảng tóm tắt
        if not STATUS_CSV.exists():
            with open(STATUS_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["track_id", "ngay_vi_pham", "loai_vi_pham", "tinh_trang"])

        # report.csv: chi tiết YOLO (phiên bản có cột track_id ở cuối)
        if not REPORT_CSV.exists():
            with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "id",
                        "timestamp",
                        "image_path",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "cx",
                        "bottom_y",
                        "lane",
                        "light_right",
                        "light_left",
                        "track_id",
                    ]
                )

    # ---------- Helpers đọc/ghi status.csv ----------
    def _read_status_rows(self):
        """
        Trả về list các row dữ liệu (bỏ header).
        Mỗi row là list [track_id, ngay_vi_pham, loai_vi_pham, tinh_trang]
        """
        if not STATUS_CSV.exists():
            return []
        with open(STATUS_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) <= 1:
            return []
        return rows[1:]

    def _write_status_rows(self, data_rows):
        """
        Ghi lại header + data_rows vào status.csv
        data_rows: list[list[str]]
        """
        with open(STATUS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "ngay_vi_pham", "loai_vi_pham", "tinh_trang"])
            writer.writerows(data_rows)

    # ---------- Load data ----------
    def _load_status_into_table(self):
        """Đọc status.csv và hiển thị lên bảng."""
        self.table.setRowCount(0)
        if not STATUS_CSV.exists():
            return
        try:
            data_rows = self._read_status_rows()
            for r_idx, row in enumerate(data_rows):
                if len(row) < 4:
                    continue
                self.table.insertRow(r_idx)
                for c_idx in range(4):
                    self.table.setItem(r_idx, c_idx, QTableWidgetItem(row[c_idx]))
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể đọc status.csv: {e}")

    # ---------- UI actions ----------
    def refresh_data(self):
        self._load_status_into_table()

    def export_report(self):
        """Xuất bảng tóm tắt (status.csv) ra file CSV do người dùng chọn."""
        if not STATUS_CSV.exists():
            QMessageBox.information(self, "Thông báo", "Không có dữ liệu để xuất.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu báo cáo tóm tắt CSV",
            "violations_status.csv",
            "CSV Files (*.csv)",
        )
        if save_path:
            try:
                with open(STATUS_CSV, "r", encoding="utf-8") as src, open(
                    save_path, "w", encoding="utf-8", newline=""
                ) as dst:
                    dst.write(src.read())
                QMessageBox.information(
                    self, "Xuất báo cáo", f"Đã xuất báo cáo tới:\n{save_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Lỗi xuất báo cáo", f"Lỗi: {e}")

    def clear_all_data(self):
        """Xóa toàn bộ ảnh + reset status.csv và report.csv về header."""
        reply = QMessageBox.question(
            self,
            "Xác nhận xóa",
            "Xóa toàn bộ dữ liệu CSV và ảnh?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # Xóa toàn bộ file ảnh (jpg/png) trong thư mục violations
            for f in VIOLATIONS_DIR.iterdir():
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    try:
                        f.unlink()
                    except Exception:
                        pass

            # reset status.csv
            with open(STATUS_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["track_id", "ngay_vi_pham", "loai_vi_pham", "tinh_trang"])

            # reset report.csv
            with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "id",
                        "timestamp",
                        "image_path",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "cx",
                        "bottom_y",
                        "lane",
                        "light_right",
                        "light_left",
                        "track_id",
                    ]
                )

            self.table.setRowCount(0)

    # ---------- Thêm / Sửa / Xóa 1 dòng ----------
    def add_row(self):
        """Thêm 1 dòng mới vào status.csv (track_id tự tăng, chỉ dùng cho nhập tay)."""
        data_rows = self._read_status_rows()

        # Tự động sinh track_id mới (max + 1)
        next_id = 1
        for r in data_rows:
            try:
                v = int(r[0])
                if v >= next_id:
                    next_id = v + 1
            except Exception:
                pass

        init_data = {
            "track_id": str(next_id),
            "ngay_vi_pham": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "loai_vi_pham": "Vượt đèn đỏ",
            "tinh_trang": "Chờ xử lý",
        }

        dlg = ViolationEditDialog(init_data, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_row = dlg.get_row()
            data_rows.append(new_row)
            self._write_status_rows(data_rows)
            self.refresh_data()

    def edit_row(self):
        """Sửa dòng đang chọn trong bảng và lưu lại vào status.csv"""
        row_idx = self.table.currentRow()
        if row_idx < 0:
            QMessageBox.information(self, "Thông báo", "Hãy chọn một dòng để sửa.")
            return

        data_rows = self._read_status_rows()
        if row_idx >= len(data_rows):
            return

        row = data_rows[row_idx]
        init_data = {
            "track_id": row[0] if len(row) > 0 else "",
            "ngay_vi_pham": row[1] if len(row) > 1 else "",
            "loai_vi_pham": row[2] if len(row) > 2 else "Vượt đèn đỏ",
            "tinh_trang": row[3] if len(row) > 3 else "Chờ xử lý",
        }

        dlg = ViolationEditDialog(init_data, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            updated_row = dlg.get_row()
            data_rows[row_idx] = updated_row
            self._write_status_rows(data_rows)
            self.refresh_data()

    def delete_row(self):
        """Xóa 1 dòng đang chọn trong bảng (và xóa chi tiết tương ứng trong report.csv theo track_id)"""
        row_idx = self.table.currentRow()
        if row_idx < 0:
            QMessageBox.information(self, "Thông báo", "Hãy chọn một dòng để xóa.")
            return

        reply = QMessageBox.question(
            self,
            "Xác nhận xóa",
            "Bạn có chắc muốn xóa dòng đã chọn?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        data_rows = self._read_status_rows()
        if row_idx >= len(data_rows):
            return

        removed_row = data_rows.pop(row_idx)
        removed_track_id = removed_row[0] if removed_row else ""

        # Ghi lại status.csv
        self._write_status_rows(data_rows)

        # Xóa các dòng trong report.csv có track_id trùng
        if REPORT_CSV.exists() and removed_track_id:
            try:
                with open(REPORT_CSV, "r", encoding="utf-8") as f:
                    rows = list(csv.reader(f))
                if not rows:
                    self.refresh_data()
                    return

                header = rows[0]
                body = rows[1:]

                # Tìm index của cột track_id (nếu có)
                track_idx = None
                try:
                    track_idx = header.index("track_id")
                except ValueError:
                    # Nếu không có cột track_id thì thôi, giữ nguyên ^_^
                    track_idx = None

                if track_idx is not None:
                    new_rows = [
                        r for r in body
                        if not (len(r) > track_idx and r[track_idx] == removed_track_id)
                    ]
                else:
                    # Không có cột track_id, không đụng đến report.csv
                    new_rows = body

                with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(new_rows)
            except Exception:
                pass

        self.refresh_data()

    # ---------- Show detail ----------
    def show_detail(self, row, col):
        """
        Khi double-click 1 dòng:
        - Lấy track_id/ngày/loại/tình_trạng từ status.csv
        - Tra trong report.csv theo track_id để lấy image_path
        - Mở dialog chi tiết
        """
        if not STATUS_CSV.exists():
            return
        try:
            # Đọc toàn bộ status.csv
            with open(STATUS_CSV, "r", encoding="utf-8") as f:
                status_rows = list(csv.reader(f))
                if row + 1 >= len(status_rows):
                    return
                data_row = status_rows[row + 1]  # row 0 là header

            track_id = data_row[0] if len(data_row) > 0 else ""
            ngay_vi_pham = data_row[1] if len(data_row) > 1 else ""
            loai_vi_pham = data_row[2] if len(data_row) > 2 else "Vượt đèn đỏ"
            tinh_trang = data_row[3] if len(data_row) > 3 else "Chờ xử lý"

            # Tìm image_path trong report.csv theo track_id
            image_path = ""
            if REPORT_CSV.exists() and track_id:
                with open(REPORT_CSV, "r", encoding="utf-8") as f:
                    report_rows = list(csv.reader(f))
                    if len(report_rows) > 1:
                        header = report_rows[0]
                        body = report_rows[1:]
                        # tìm index cột image_path và track_id
                        try:
                            img_idx = header.index("image_path")
                        except ValueError:
                            img_idx = 2  # fallback: cột 2 như bản cũ

                        track_idx = None
                        try:
                            track_idx = header.index("track_id")
                        except ValueError:
                            track_idx = None

                        if track_idx is not None:
                            for r in body:
                                if len(r) > track_idx and r[track_idx] == track_id:
                                    if len(r) > img_idx:
                                        image_path = r[img_idx]
                                    break

            info = {
                "track_id": track_id,
                "date": ngay_vi_pham,
                "violation_type": loai_vi_pham,
                "note": tinh_trang,
                "image_path": image_path,
            }

            dlg = ViolationDetailDialog(info, parent=self)
            dlg.exec()

        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể mở chi tiết: {e}")
