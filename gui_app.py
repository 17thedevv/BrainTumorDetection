"""
Brain Tumor Detector — Desktop GUI (PyQt5)
==========================================
Run:
    .\\venv\\Scripts\\Activate.ps1
    python gui_app.py
"""

import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5.QtCore import (
    QObject, QRunnable, QSize, Qt, QThreadPool, pyqtSignal, pyqtSlot,
    QPropertyAnimation, QEasingCurve,
)
from PyQt5.QtGui import (
    QColor, QDragEnterEvent, QDropEvent, QFont, QFontDatabase,
    QIcon, QPainter, QPalette, QPixmap,
)
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGraphicsDropShadowEffect,
    QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── TensorFlow & Fallback ──────────────────────────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError as e:
    print("[WARNING] Cannot load TensorFlow. App will run in Demo (Mock) Mode.")
    TF_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH  = ROOT / "saved_model" / "brain_tumor_detector" / "best_model.h5"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_KEYS  = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_INFO = {
    "Glioma": {
        "icon":  "🔴", "color": "#ef4444", "bg": "#2d1515",
        "severity": "Nguy hiểm cao",
        "desc": (
            "U thần kinh đệm phát sinh từ tế bào thần kinh đệm (glial cells). "
            "Đây là loại u não phổ biến nhất và thường có tính ác tính cao. "
            "Cần thăm khám chuyên khoa thần kinh ngay."
        ),
    },
    "Meningioma": {
        "icon":  "🟠", "color": "#f97316", "bg": "#2d1e10",
        "severity": "Mức độ trung bình",
        "desc": (
            "U màng não phát sinh từ màng bao quanh não và tủy sống. "
            "Thường lành tính và phát triển chậm. "
            "Phần lớn có thể điều trị bằng phẫu thuật hoặc xạ trị."
        ),
    },
    "No Tumor": {
        "icon":  "🟢", "color": "#22c55e", "bg": "#0f2d1a",
        "severity": "Bình thường",
        "desc": (
            "Không phát hiện khối u trong ảnh MRI này. "
            "Hình ảnh trong giới hạn bình thường. "
            "Vẫn nên thực hiện kiểm tra định kỳ theo lịch của bác sĩ."
        ),
    },
    "Pituitary": {
        "icon":  "🟡", "color": "#eab308", "bg": "#2a260a",
        "severity": "Mức độ thấp",
        "desc": (
            "U tuyến yên phát sinh ở tuyến yên, thường lành tính và nhỏ. "
            "Có thể ảnh hưởng đến việc sản xuất hormone. "
            "Điều trị thường hiệu quả qua dùng thuốc hoặc phẫu thuật."
        ),
    },
}

BAR_COLORS = {
    "Glioma":     "#ef4444",
    "Meningioma": "#f97316",
    "No Tumor":   "#22c55e",
    "Pituitary":  "#eab308",
}

# ── Global QSS stylesheet ──────────────────────────────────────────────────────
QSS = """
QMainWindow, QWidget#central {
    background-color: #0a0f1e;
}
QWidget {
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #e2e8f0;
    font-size: 13px;
}

/* Cards */
QFrame#card {
    background-color: #111827;
    border: 1px solid #1e2d40;
    border-radius: 12px;
}
QFrame#inner {
    background-color: #0d1526;
    border: 1px solid #1e2d40;
    border-radius: 8px;
}

/* Primary button */
QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #3b82f6, stop:1 #6366f1);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: 700;
}
QPushButton#primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2563eb, stop:1 #4f46e5);
}
QPushButton#primary:disabled {
    background: #1e2d40;
    color: #4a5568;
}
QPushButton#primary:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1d4ed8, stop:1 #4338ca);
}

/* Section labels */
QLabel#section {
    color: #64748b;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#title {
    color: #e2e8f0;
    font-size: 22px;
    font-weight: 800;
}
QLabel#muted {
    color: #64748b;
    font-size: 12px;
}
QLabel#badge {
    background-color: #1e2d40;
    color: #60a5fa;
    border-radius: 12px;
    padding: 4px 12px;
    font-weight: 700;
    font-size: 12px;
}
QLabel#disclaimer {
    background-color: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 8px;
    color: #fbbf24;
    padding: 8px 14px;
    font-size: 11px;
}

/* Progress bars */
QProgressBar {
    background-color: #1e2d40;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: right;
}
QProgressBar::chunk {
    border-radius: 4px;
}

/* Scrollbar */
QScrollBar:vertical {
    background: #0a0f1e;
    width: 6px;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #1e2d40;
    border-radius: 3px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

/* Splitter */
QSplitter::handle {
    background-color: #1e2d40;
    width: 2px;
}
"""


# ── Worker signals ─────────────────────────────────────────────────────────────
class WorkerSignals(QObject):
    result   = pyqtSignal(object)   # ndarray of predictions
    error    = pyqtSignal(str)
    finished = pyqtSignal()


# ── Model loader runnable ──────────────────────────────────────────────────────
class ModelLoader(QRunnable):
    def __init__(self, path: Path):
        super().__init__()
        self.path    = path
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            if not TF_AVAILABLE:
                import time
                time.sleep(1.0) # Giả lập độ trễ tải model
                self.signals.result.emit("DEMO_MODEL_READY")
                return

            model = tf.keras.models.load_model(str(self.path))
            self.signals.result.emit(model)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


# ── Inference runnable ─────────────────────────────────────────────────────────
class InferenceRunner(QRunnable):
    def __init__(self, model, image_path: Path):
        super().__init__()
        self.model      = model
        self.image_path = image_path
        self.signals    = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            if not TF_AVAILABLE:
                # ── Chế độ Demo (Mock mode) ──
                import time
                time.sleep(1.5) # Giả lập thời gian chạy CNN
                
                # Tạo kết quả ngẫu nhiên hợp lý dựa vào kích thước ảnh để kết quả ổn định với 1 ảnh
                img = Image.open(self.image_path)
                seed = img.width * img.height
                np.random.seed(seed % 100000)
                
                mock_preds = np.random.dirichlet(np.ones(4))
                
                # Làm cho 1 class trội lên để kết quả trông thực tế
                max_idx = np.random.randint(0, 4)
                mock_preds[max_idx] += 1.5
                mock_preds = mock_preds / np.sum(mock_preds)
                
                self.signals.result.emit(mock_preds)
                return

            img = Image.open(self.image_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            arr = np.array(img, dtype=np.float32)
            arr = np.expand_dims(arr, 0)
            preds = self.model.predict(arr, verbose=0)[0]
            self.signals.result.emit(preds)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


# ── Drop-enabled image label ───────────────────────────────────────────────────
class DropLabel(QLabel):
    """A QLabel that accepts file drops and emits a signal."""
    file_dropped = pyqtSignal(Path)
    clicked      = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._default_style = (
            "border: 2px dashed #1e2d40;"
            "border-radius: 10px;"
            "background: #0d1526;"
            "padding: 30px;"
        )
        self._hover_style = (
            "border: 2px dashed #3b82f6;"
            "border-radius: 10px;"
            "background: rgba(59,130,246,0.07);"
            "padding: 30px;"
        )
        self.setStyleSheet(self._default_style)
        self.setCursor(Qt.PointingHandCursor)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self._hover_style)

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self._default_style)

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self._default_style)
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}:
                self.file_dropped.emit(path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

    def enterEvent(self, event):
        self.setStyleSheet(self._hover_style)

    def leaveEvent(self, event):
        self.setStyleSheet(self._default_style)


# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.model         = None
        self.selected_path = None
        self.pool          = QThreadPool.globalInstance()
        self._bar_widgets  = {}   # class_name → QProgressBar
        self._pct_labels   = {}   # class_name → QLabel

        self._setup_window()
        self._build_ui()
        self._load_model()

    # ── Window ────────────────────────────────────────────────────────────────
    def _setup_window(self):
        self.setWindowTitle("🧠  Brain Tumor Detector  |  AI Medical Imaging")
        self.setMinimumSize(1040, 680)
        self.resize(1100, 740)

        # Center
        screen = QApplication.desktop().screenGeometry()
        self.move(
            (screen.width()  - 1100) // 2,
            (screen.height() -  740) // 2,
        )

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

    # ── Full UI ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        root_layout = QVBoxLayout(self.centralWidget())
        root_layout.setSpacing(0)
        root_layout.setContentsMargins(0, 0, 0, 0)

        root_layout.addWidget(self._make_header())

        # Disclaimer
        disc = QLabel(
            "⚠️  Ứng dụng này chỉ dành cho mục đích nghiên cứu và học thuật. "
            "Kết quả không thay thế chẩn đoán y tế từ bác sĩ chuyên khoa."
        )
        disc.setObjectName("disclaimer")
        disc.setWordWrap(True)
        root_layout.addWidget(disc)
        m = root_layout.contentsMargins()
        root_layout.setContentsMargins(16, 10, 16, 10)

        # Body splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.addWidget(self._make_left_panel())
        splitter.addWidget(self._make_right_panel())
        splitter.setSizes([480, 560])
        root_layout.addWidget(splitter, 1)

        root_layout.addWidget(self._make_statusbar())

    # ── Header ────────────────────────────────────────────────────────────────
    def _make_header(self):
        header = QFrame()
        header.setObjectName("card")
        header.setStyleSheet(
            "QFrame { background: #111827; border: none; border-radius: 0; border-bottom: 1px solid #1e2d40; }"
        )
        h = QHBoxLayout(header)
        h.setContentsMargins(20, 14, 20, 14)

        # Logo
        logo = QHBoxLayout()
        logo.setSpacing(12)
        icon = QLabel("🧠")
        icon.setStyleSheet("font-size: 32px; padding: 2px;")
        logo.addWidget(icon)

        text_v = QVBoxLayout()
        text_v.setSpacing(2)
        t1 = QLabel("Brain Tumor Detector")
        t1.setStyleSheet("font-size: 17px; font-weight: 800; color: #e2e8f0;")
        t2 = QLabel("AI Medical Imaging  ·  Custom CNN  ·  TensorFlow 2.20")
        t2.setStyleSheet("font-size: 11px; color: #64748b;")
        text_v.addWidget(t1)
        text_v.addWidget(t2)
        logo.addLayout(text_v)

        h.addLayout(logo)
        h.addStretch()

        self._status_badge = QLabel("⏳  Đang tải model…")
        self._status_badge.setObjectName("badge")
        h.addWidget(self._status_badge)

        return header

    # ── Left panel ────────────────────────────────────────────────────────────
    def _make_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 8, 8, 0)

        # -- Upload card --
        upload_card = QFrame()
        upload_card.setObjectName("card")
        uc_layout = QVBoxLayout(upload_card)
        uc_layout.setContentsMargins(18, 14, 18, 16)
        uc_layout.setSpacing(8)

        sec = QLabel("📂   TẢI ẢNH MRI LÊN")
        sec.setObjectName("section")
        uc_layout.addWidget(sec)

        # Drop zone
        self._drop_lbl = DropLabel()
        self._drop_lbl.setAlignment(Qt.AlignCenter)
        drop_inner = QVBoxLayout()
        drop_inner.setSpacing(4)

        self._drop_icon = QLabel("🩻")
        self._drop_icon.setStyleSheet("font-size: 48px; border: none; background: transparent;")
        self._drop_icon.setAlignment(Qt.AlignCenter)

        drop_title = QLabel("Nhấn hoặc kéo & thả ảnh vào đây")
        drop_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #e2e8f0; border: none; background: transparent;")
        drop_title.setAlignment(Qt.AlignCenter)

        drop_sub = QLabel("JPG · PNG · BMP · WEBP  |  Tối đa 16 MB")
        drop_sub.setStyleSheet("font-size: 11px; color: #64748b; border: none; background: transparent;")
        drop_sub.setAlignment(Qt.AlignCenter)

        drop_inner.addStretch()
        drop_inner.addWidget(self._drop_icon)
        drop_inner.addWidget(drop_title)
        drop_inner.addWidget(drop_sub)
        drop_inner.addStretch()
        self._drop_lbl.setLayout(drop_inner)
        self._drop_lbl.setMinimumHeight(160)

        self._drop_lbl.clicked.connect(self._browse_file)
        self._drop_lbl.file_dropped.connect(self._load_image)
        uc_layout.addWidget(self._drop_lbl)

        # File info
        self._file_info = QLabel("Chưa chọn file")
        self._file_info.setObjectName("muted")
        self._file_info.setAlignment(Qt.AlignCenter)
        uc_layout.addWidget(self._file_info)

        layout.addWidget(upload_card)

        # -- Preview card --
        prev_card = QFrame()
        prev_card.setObjectName("card")
        prev_layout = QVBoxLayout(prev_card)
        prev_layout.setContentsMargins(18, 14, 18, 16)
        prev_layout.setSpacing(8)

        sec2 = QLabel("🖼️   XEM TRƯỚC")
        sec2.setObjectName("section")
        prev_layout.addWidget(sec2)

        self._preview_lbl = QLabel("Ảnh MRI sẽ hiển thị ở đây")
        self._preview_lbl.setObjectName("muted")
        self._preview_lbl.setAlignment(Qt.AlignCenter)
        self._preview_lbl.setMinimumHeight(220)
        self._preview_lbl.setStyleSheet(
            "background: #0d1526; border-radius: 8px; border: 1px solid #1e2d40;"
        )
        prev_layout.addWidget(self._preview_lbl)
        layout.addWidget(prev_card, 1)

        # -- Analyze button --
        self._analyze_btn = QPushButton("🔍   Phân tích ảnh")
        self._analyze_btn.setObjectName("primary")
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.setFixedHeight(52)
        self._analyze_btn.setCursor(Qt.PointingHandCursor)
        self._analyze_btn.clicked.connect(self._on_analyze)
        layout.addWidget(self._analyze_btn)

        return panel

    # ── Right panel ───────────────────────────────────────────────────────────
    def _make_right_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 8, 0, 0)

        # -- Result card --
        self._result_card = QFrame()
        self._result_card.setObjectName("card")
        self._result_layout = QVBoxLayout(self._result_card)
        self._result_layout.setContentsMargins(18, 14, 18, 16)
        self._result_layout.setSpacing(10)

        sec = QLabel("🩺   KẾT QUẢ CHẨN ĐOÁN")
        sec.setObjectName("section")
        self._result_layout.addWidget(sec)

        self._result_placeholder = QLabel("Tải ảnh lên và nhấn\n\"Phân tích ảnh\" để xem kết quả")
        self._result_placeholder.setObjectName("muted")
        self._result_placeholder.setAlignment(Qt.AlignCenter)
        self._result_placeholder.setMinimumHeight(80)
        self._result_layout.addWidget(self._result_placeholder)

        layout.addWidget(self._result_card)

        # -- Probability card --
        prob_card = QFrame()
        prob_card.setObjectName("card")
        prob_layout = QVBoxLayout(prob_card)
        prob_layout.setContentsMargins(18, 14, 18, 16)
        prob_layout.setSpacing(12)

        sec2 = QLabel("📊   XÁC SUẤT TỪNG NHÃN")
        sec2.setObjectName("section")
        prob_layout.addWidget(sec2)

        for name in CLASS_NAMES:
            color = BAR_COLORS[name]
            icon  = CLASS_INFO[name]["icon"]

            row = QHBoxLayout()
            row.setSpacing(10)

            lbl = QLabel(f"{icon}  {name}")
            lbl.setStyleSheet(f"color: {color}; font-weight: 700; font-size: 13px;")
            lbl.setFixedWidth(130)
            row.addWidget(lbl)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            bar.setStyleSheet(
                f"QProgressBar {{ background:#1e2d40; border-radius:4px; }}"
                f"QProgressBar::chunk {{ background:{color}; border-radius:4px; }}"
            )
            row.addWidget(bar, 1)

            pct = QLabel("0.0%")
            pct.setStyleSheet(f"color:{color}; font-family:'Courier New'; font-weight:700; font-size:12px;")
            pct.setFixedWidth(46)
            pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(pct)

            prob_layout.addLayout(row)
            self._bar_widgets[name] = bar
            self._pct_labels[name]  = pct

        layout.addWidget(prob_card)

        # -- Model info card --
        info_card = QFrame()
        info_card.setObjectName("card")
        info_layout = QHBoxLayout(info_card)
        info_layout.setContentsMargins(18, 14, 18, 14)
        info_layout.setSpacing(8)

        specs = [
            ("🏗️", "Kiến trúc", "Custom CNN"),
            ("📐", "Input size", "224 × 224 × 3"),
            ("🎯", "Classes", "4 loại u não"),
            ("🧠", "Optimizer", "Adam lr=1e-4"),
        ]
        for icon, label, val in specs:
            box = QFrame()
            box.setObjectName("inner")
            box_v = QVBoxLayout(box)
            box_v.setContentsMargins(10, 10, 10, 10)
            box_v.setSpacing(2)
            box_v.setAlignment(Qt.AlignCenter)

            QLabel(icon, box).setStyleSheet("font-size:22px; border:none; background:transparent;")
            v_lbl = QLabel(val)
            v_lbl.setStyleSheet("font-weight:700; font-size:12px; color:#e2e8f0; border:none; background:transparent;")
            v_lbl.setAlignment(Qt.AlignCenter)
            k_lbl = QLabel(label)
            k_lbl.setStyleSheet("font-size:10px; color:#64748b; border:none; background:transparent;")
            k_lbl.setAlignment(Qt.AlignCenter)

            for w in box.findChildren(QLabel):
                box_v.addWidget(w, 0, Qt.AlignCenter)

            info_layout.addWidget(box, 1)

        layout.addWidget(info_card)
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    # ── Status bar ────────────────────────────────────────────────────────────
    def _make_statusbar(self):
        sb = QFrame()
        sb.setStyleSheet("background:#111827; border-top:1px solid #1e2d40;")
        sb.setFixedHeight(28)
        h = QHBoxLayout(sb)
        h.setContentsMargins(14, 0, 14, 0)

        self._statusbar_lbl = QLabel(
            "⚠️  Ứng dụng chỉ dành cho mục đích nghiên cứu. Không thay thế chẩn đoán y tế."
        )
        self._statusbar_lbl.setStyleSheet("color:#f59e0b; font-size:10px;")
        h.addWidget(self._statusbar_lbl)
        return sb

    # ── Load model ────────────────────────────────────────────────────────────
    def _load_model(self):
        if not TF_AVAILABLE:
            self._status_badge.setText("⚠️  Demo Mode (Thiếu TF DLL)")
            self._status_badge.setStyleSheet(
                "background:#2d1e10; color:#f97316; border-radius:12px; padding:4px 12px; font-weight:700;"
            )
            loader = ModelLoader(MODEL_PATH)
            loader.signals.result.connect(self._on_model_loaded)
            loader.signals.error.connect(self._on_model_error)
            self.pool.start(loader)
            return

        if not MODEL_PATH.exists():
            self._status_badge.setText("❌  Model không tồn tại")
            self._status_badge.setStyleSheet(
                "background:#2d1515; color:#ef4444; border-radius:12px; padding:4px 12px; font-weight:700;"
            )
            return

        loader = ModelLoader(MODEL_PATH)
        loader.signals.result.connect(self._on_model_loaded)
        loader.signals.error.connect(self._on_model_error)
        self.pool.start(loader)

    def _on_model_loaded(self, model):
        self.model = model
        
        if TF_AVAILABLE:
            self._status_badge.setText("✅  Model sẵn sàng")
            self._status_badge.setStyleSheet(
                "background:#0f2d1a; color:#22c55e; border-radius:12px; padding:4px 12px; font-weight:700;"
            )

        if self.selected_path:
            self._analyze_btn.setEnabled(True)

    def _on_model_error(self, msg):
        self._status_badge.setText("❌  Lỗi tải model")
        self._status_badge.setStyleSheet(
            "background:#2d1515; color:#ef4444; border-radius:12px; padding:4px 12px; font-weight:700;"
        )
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(
            self, "Lỗi tải model",
            f"Không thể tải model:\n\n{msg}\n\n"
            "Vui lòng chạy huấn luyện trước:\n  python train.py --dataset-dir data/Dataset",
        )

    # ── Browse / Load image ───────────────────────────────────────────────────
    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh MRI", str(ROOT),
            "Ảnh (*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif);;Tất cả (*.*)",
        )
        if path:
            self._load_image(Path(path))

    def _load_image(self, path: Path):
        self.selected_path = path
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as exc:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Lỗi", f"Không thể đọc ảnh:\n{exc}")
            return

        # File info
        size_kb = path.stat().st_size // 1024
        self._file_info.setText(
            f"{path.name}  ·  {size_kb} KB  ·  {pil_img.size[0]}×{pil_img.size[1]} px"
        )
        self._file_info.setStyleSheet("color:#94a3b8; font-size:11px;")

        # Preview (fit into label)
        pil_img.thumbnail((440, 280))
        w, h = pil_img.size
        data = pil_img.tobytes("raw", "RGB")
        qimg = __import__("PyQt5.QtGui", fromlist=["QImage"]).QImage(data, w, h, w * 3, __import__("PyQt5.QtGui", fromlist=["QImage"]).QImage.Format_RGB888)
        self._preview_lbl.setPixmap(QPixmap.fromImage(qimg).scaled(
            self._preview_lbl.width(), self._preview_lbl.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        ))
        self._preview_lbl.setAlignment(Qt.AlignCenter)
        self._drop_icon.setText("✅")

        if self.model:
            self._analyze_btn.setEnabled(True)

    # ── Analyze ───────────────────────────────────────────────────────────────
    def _on_analyze(self):
        if not self.selected_path or not self.model:
            return

        self._analyze_btn.setEnabled(False)
        self._analyze_btn.setText("⏳   Đang phân tích…")
        self._statusbar_lbl.setText("🔄  Đang chạy inference CNN…")
        self._statusbar_lbl.setStyleSheet("color:#60a5fa; font-size:10px;")

        worker = InferenceRunner(self.model, self.selected_path)
        worker.signals.result.connect(self._show_result)
        worker.signals.error.connect(self._on_infer_error)
        self.pool.start(worker)

    def _on_infer_error(self, msg):
        self._analyze_btn.setEnabled(True)
        self._analyze_btn.setText("🔍   Phân tích ảnh")
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Lỗi inference", f"Lỗi khi chạy model:\n{msg}")

    # ── Show result ───────────────────────────────────────────────────────────
    def _show_result(self, preds: np.ndarray):
        idx        = int(np.argmax(preds))
        pred_name  = CLASS_NAMES[idx]
        confidence = float(preds[idx]) * 100
        info       = CLASS_INFO[pred_name]
        color      = info["color"]

        # ── Update bars ────────────────────────────────────────────────────────
        for i, name in enumerate(CLASS_NAMES):
            val = int(float(preds[i]) * 100)
            self._bar_widgets[name].setValue(val)
            self._pct_labels[name].setText(f"{val:.1f}%")

        # ── Rebuild result card ────────────────────────────────────────────────
        # Clear old widgets
        while self._result_layout.count():
            item = self._result_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Section header
        sec = QLabel("🩺   KẾT QUẢ CHẨN ĐOÁN")
        sec.setObjectName("section")
        self._result_layout.addWidget(sec)

        # Prediction row
        pred_row = QHBoxLayout()
        pred_row.setSpacing(14)

        icon_box = QLabel(info["icon"])
        icon_box.setStyleSheet(
            f"font-size:36px; background:{info['bg']}; border-radius:12px;"
            f" padding:8px 14px; border:1px solid {color}44;"
        )
        icon_box.setAlignment(Qt.AlignCenter)
        pred_row.addWidget(icon_box)

        name_v = QVBoxLayout()
        name_v.setSpacing(2)
        sub = QLabel("Phân loại phát hiện")
        sub.setStyleSheet("color:#64748b; font-size:10px;")
        name_v.addWidget(sub)
        pred_lbl = QLabel(pred_name)
        pred_lbl.setStyleSheet(f"color:{color}; font-size:24px; font-weight:800;")
        name_v.addWidget(pred_lbl)
        sev = QLabel(info["severity"])
        sev.setStyleSheet(f"color:{color}; font-size:11px; font-weight:600;")
        name_v.addWidget(sev)
        pred_row.addLayout(name_v, 1)
        self._result_layout.addLayout(pred_row)

        # Confidence bar
        conf_lbl = QLabel(f"Độ tin cậy: {confidence:.1f}%")
        conf_lbl.setStyleSheet(f"color:{color}; font-weight:700; font-size:13px;")
        self._result_layout.addWidget(conf_lbl)

        conf_bar = QProgressBar()
        conf_bar.setRange(0, 100)
        conf_bar.setValue(int(confidence))
        conf_bar.setTextVisible(False)
        conf_bar.setFixedHeight(10)
        conf_bar.setStyleSheet(
            f"QProgressBar {{ background:#1e2d40; border-radius:5px; }}"
            f"QProgressBar::chunk {{ background:{color}; border-radius:5px; }}"
        )
        self._result_layout.addWidget(conf_bar)

        # Description box
        desc_box = QFrame()
        desc_box.setStyleSheet(
            f"background:{info['bg']}; border-radius:8px; border:1px solid {color}33;"
        )
        desc_v = QVBoxLayout(desc_box)
        desc_v.setContentsMargins(14, 12, 14, 12)
        desc_lbl = QLabel(info["desc"])
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color:#cbd5e1; font-size:12px; background:transparent; border:none;")
        desc_v.addWidget(desc_lbl)
        self._result_layout.addWidget(desc_box)

        # Stats row
        sorted_preds = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)
        second_conf  = float(sorted_preds[1][1]) * 100
        margin       = confidence - second_conf

        stats_h = QHBoxLayout()
        stats_h.setSpacing(8)
        for label, val, col in [
            ("Confidence",  f"{confidence:.1f}%",  color),
            ("2nd Best",    f"{second_conf:.1f}%", "#64748b"),
            ("Margin",      f"{margin:.1f}%",       "#3b82f6"),
        ]:
            sf = QFrame()
            sf.setObjectName("inner")
            sv = QVBoxLayout(sf)
            sv.setContentsMargins(10, 10, 10, 10)
            sv.setSpacing(2)
            v = QLabel(val)
            v.setStyleSheet(f"color:{col}; font-family:'Courier New'; font-size:16px; font-weight:700; border:none; background:transparent;")
            v.setAlignment(Qt.AlignCenter)
            k = QLabel(label)
            k.setStyleSheet("color:#64748b; font-size:10px; border:none; background:transparent;")
            k.setAlignment(Qt.AlignCenter)
            sv.addWidget(v)
            sv.addWidget(k)
            stats_h.addWidget(sf, 1)
        self._result_layout.addLayout(stats_h)

        # ── Restore button ─────────────────────────────────────────────────────
        self._analyze_btn.setEnabled(True)
        self._analyze_btn.setText("🔍   Phân tích lại")
        self._statusbar_lbl.setText(
            f"✅  Kết quả: {pred_name}  ({confidence:.1f}% confidence)"
            "  ·  ⚠️  Chỉ dành cho nghiên cứu, không thay thế chẩn đoán y tế."
        )
        self._statusbar_lbl.setStyleSheet("color:#94a3b8; font-size:10px;")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)

    # High-DPI support
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
