"""
main_window.py - Main application window for Brain Tumor Detection Inference GUI.
All business logic is delegated to controller.py; this file manages layout only.
"""

import os
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStatusBar,
    QSizePolicy, QMessageBox, QSpinBox, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QIcon

from gui.controller import Controller
from gui.widgets import ConfidenceBar, PredictionResultWidget, SectionFrame

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0f1923;
    color: #ecf0f1;
    font-family: "Segoe UI";
}
QPushButton {
    background-color: #1a6faf;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2980b9;
}
QPushButton:pressed {
    background-color: #1a5276;
}
QPushButton:disabled {
    background-color: #2c3e50;
    color: #7f8c8d;
}
QStatusBar {
    background-color: #0a1017;
    color: #7f8c8d;
    font-size: 11px;
}
QSpinBox {
    background-color: #1e2d3d;
    border: 1px solid #34495e;
    border-radius: 5px;
    color: #ecf0f1;
    padding: 4px;
}
QLabel {
    background-color: transparent;
}
"""


class InferenceThread(QThread):
    """Run inference in background so GUI doesn't freeze."""
    result_ready = pyqtSignal(dict)

    def __init__(self, controller: Controller):
        super().__init__()
        self._controller = controller

    def run(self):
        result = self._controller.run_inference()
        self.result_ready.emit(result)


class MainWindow(QMainWindow):
    CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    def __init__(self):
        super().__init__()
        self.controller = Controller()
        self._inference_thread = None
        self._last_result = None

        self.setWindowTitle("Brain Tumor Detection — Inference GUI")
        self.setMinimumSize(960, 680)
        self.setStyleSheet(DARK_STYLESHEET)

        self._build_ui()
        self.statusBar().showMessage("Ready. Load a model to begin.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        root.addLayout(self._build_left_panel(), stretch=3)
        root.addLayout(self._build_right_panel(), stretch=2)

    def _build_left_panel(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Title
        title = QLabel("🧠 Brain Tumor Detection")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet("color: #3498db; padding-bottom: 4px;")
        layout.addWidget(title)

        # Model load section
        model_frame = SectionFrame("Model")
        self.model_status_lbl = QLabel("No model loaded")
        self.model_status_lbl.setStyleSheet("color: #e74c3c; border: none;")
        self.model_status_lbl.setFont(QFont("Segoe UI", 10))
        model_frame.add_widget(self.model_status_lbl)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(64, 512)
        self.image_size_spin.setValue(224)
        self.image_size_spin.setSuffix(" px")
        form.addRow("Image size:", self.image_size_spin)
        model_frame._layout.addLayout(form)

        load_model_btn = QPushButton("📂  Load Model (.pth)")
        load_model_btn.clicked.connect(self._on_load_model)
        model_frame.add_widget(load_model_btn)
        layout.addWidget(model_frame)

        # MRI image display section
        image_frame = SectionFrame("MRI Image")
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(280)
        self.image_label.setStyleSheet(
            "border: 2px dashed #34495e; border-radius: 8px; color: #7f8c8d; font-size: 13px;"
        )
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_frame.add_widget(self.image_label)

        select_image_btn = QPushButton("🖼️  Select MRI Image")
        select_image_btn.clicked.connect(self._on_select_image)
        image_frame.add_widget(select_image_btn)
        layout.addWidget(image_frame, stretch=1)

        # Action buttons
        btn_row = QHBoxLayout()
        self.predict_btn = QPushButton("⚡  Run Prediction")
        self.predict_btn.setEnabled(False)
        self.predict_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; } QPushButton:hover { background-color: #2ecc71; } QPushButton:disabled { background-color: #2c3e50; color: #7f8c8d; }"
        )
        self.predict_btn.clicked.connect(self._on_predict)

        self.export_btn = QPushButton("💾  Export PNG")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)

        btn_row.addWidget(self.predict_btn)
        btn_row.addWidget(self.export_btn)
        layout.addLayout(btn_row)

        return layout

    def _build_right_panel(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Prediction result
        result_frame = SectionFrame("Prediction")
        self.result_widget = PredictionResultWidget()
        result_frame.add_widget(self.result_widget)
        layout.addWidget(result_frame)

        # Confidence bars
        conf_frame = SectionFrame("Class Confidence")
        self.conf_bars = {}
        for cls in self.CLASS_NAMES:
            bar = ConfidenceBar(cls)
            conf_frame.add_widget(bar)
            self.conf_bars[cls] = bar
        layout.addWidget(conf_frame)

        # Grad-CAM Tumor Localization
        gradcam_frame = SectionFrame("Grad-CAM Tumor Localization")
        self.gradcam_label = QLabel("Run prediction to see tumor localization")
        self.gradcam_label.setAlignment(Qt.AlignCenter)
        self.gradcam_label.setMinimumHeight(240)
        self.gradcam_label.setStyleSheet(
            "border: 2px dashed #34495e; border-radius: 8px; color: #7f8c8d; font-size: 12px;"
        )
        self.gradcam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gradcam_frame.add_widget(self.gradcam_label)
        layout.addWidget(gradcam_frame)

        layout.addStretch()
        return layout

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model checkpoint", "saved_model", "PyTorch Model (*.pth)"
        )
        if not path:
            return

        image_size = self.image_size_spin.value()
        msg = self.controller.load_model(path, image_size=image_size)
        if msg.startswith("Error"):
            self.model_status_lbl.setText(msg)
            self.model_status_lbl.setStyleSheet("color: #e74c3c; border: none;")
        else:
            self.model_status_lbl.setText(msg)
            self.model_status_lbl.setStyleSheet("color: #2ecc71; border: none;")
        self._refresh_predict_btn()
        self.statusBar().showMessage(msg)

    def _on_select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MRI Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not path:
            return

        msg = self.controller.set_image(path)
        self.statusBar().showMessage(msg)
        self._display_image(path)
        self.result_widget.reset()
        for bar in self.conf_bars.values():
            bar.set_value(0.0)
        self.gradcam_label.clear()
        self.gradcam_label.setText("Run prediction to see tumor localization")
        self.gradcam_label.setStyleSheet(
            "border: 2px dashed #34495e; border-radius: 8px; color: #7f8c8d; font-size: 12px;"
        )
        self.export_btn.setEnabled(False)
        self._last_result = None
        self._refresh_predict_btn()

    def _on_predict(self):
        self.predict_btn.setEnabled(False)
        self.statusBar().showMessage("Running inference…")
        self._inference_thread = InferenceThread(self.controller)
        self._inference_thread.result_ready.connect(self._on_result_ready)
        self._inference_thread.start()

    def _on_result_ready(self, result: dict):
        self._last_result = result
        self.result_widget.update_result(result)

        if 'probabilities' in result:
            for cls_name, prob in result['probabilities']:
                if cls_name in self.conf_bars:
                    self.conf_bars[cls_name].set_value(prob)

        # Display Grad-CAM heatmap
        if 'gradcam_heatmap' in result and result['gradcam_heatmap'] is not None:
            self._display_gradcam(result['gradcam_heatmap'])

        self.predict_btn.setEnabled(True)
        self.export_btn.setEnabled('error' not in result)

        if 'error' not in result:
            self.statusBar().showMessage(
                f"Done — {result['class']} ({result['confidence']*100:.1f}%) in {result['inference_time_ms']:.1f} ms"
            )
        else:
            self.statusBar().showMessage(f"Error: {result['error']}")
            self.gradcam_label.setText("Error generating Grad-CAM")

    def _on_export(self):
        if not self._last_result or 'error' in self._last_result:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Grad-CAM Result", "gradcam_result.png", "PNG Image (*.png)"
        )
        if not save_path:
            return
        
        # Export the full-resolution Grad-CAM image directly from PIL
        if 'gradcam_heatmap' in self._last_result and self._last_result['gradcam_heatmap'] is not None:
            try:
                self._last_result['gradcam_heatmap'].save(save_path, "PNG")
                self.statusBar().showMessage(f"Exported Grad-CAM to {save_path}")
            except Exception as e:
                self.statusBar().showMessage(f"Export failed: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _display_image(self, path: str):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.image_label.setText("Could not load image.")
            return
        scaled = pixmap.scaled(
            self.image_label.width() or 400,
            self.image_label.height() or 280,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def _display_gradcam(self, pil_img):
        from PyQt5.QtGui import QImage
        im = pil_img.convert("RGBA")
        data = im.tobytes("raw", "RGBA")
        qim = QImage(data, im.size[0], im.size[1], QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qim)
        
        scaled = pixmap.scaled(
            self.gradcam_label.width() or 400,
            self.gradcam_label.height() or 240,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.gradcam_label.setPixmap(scaled)
        self.gradcam_label.setStyleSheet("border: 1px solid #34495e; border-radius: 8px;")

    def _refresh_predict_btn(self):
        ready = self.controller.is_model_loaded() and bool(self.controller.get_image_path())
        self.predict_btn.setEnabled(ready)

