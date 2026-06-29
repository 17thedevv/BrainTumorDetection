"""
widgets.py - Reusable PyQt5 widget components for the Brain Tumor Detection GUI.
No business logic here; all logic lives in controller.py.
"""

from PyQt5.QtWidgets import (
    QWidget, QLabel, QProgressBar, QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPalette


CLASS_COLORS = {
    'Glioma':     '#e74c3c',
    'Meningioma': '#e67e22',
    'No Tumor':   '#2ecc71',
    'Pituitary':  '#3498db',
}


class ConfidenceBar(QWidget):
    """A labeled horizontal progress bar showing per-class confidence."""

    def __init__(self, class_name: str, parent=None):
        super().__init__(parent)
        self.class_name = class_name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self.label = QLabel(class_name)
        self.label.setFixedWidth(100)
        self.label.setFont(QFont("Segoe UI", 10))
        self.label.setStyleSheet("color: #ecf0f1;")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        self.bar.setFixedHeight(20)
        color = CLASS_COLORS.get(class_name, '#95a5a6')
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 8px;
                background-color: #2c3e50;
                color: #ffffff;
                text-align: center;
            }}
            QProgressBar::chunk {{
                border-radius: 8px;
                background-color: {color};
            }}
        """)

        layout.addWidget(self.label)
        layout.addWidget(self.bar)

    def set_value(self, probability: float):
        """Update bar with a probability value in [0, 1]."""
        pct = int(probability * 100)
        self.bar.setValue(pct)
        self.bar.setFormat(f"{probability * 100:.1f}%")


class PredictionResultWidget(QWidget):
    """Shows class label, confidence text, and inference time."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        self.class_label = QLabel("—")
        self.class_label.setAlignment(Qt.AlignCenter)
        self.class_label.setFont(QFont("Segoe UI", 22, QFont.Bold))
        self.class_label.setStyleSheet("color: #2ecc71; padding: 6px 0;")

        self.conf_label = QLabel("Confidence: —")
        self.conf_label.setAlignment(Qt.AlignCenter)
        self.conf_label.setFont(QFont("Segoe UI", 11))
        self.conf_label.setStyleSheet("color: #bdc3c7;")

        self.time_label = QLabel("Inference time: —")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setFont(QFont("Segoe UI", 10))
        self.time_label.setStyleSheet("color: #7f8c8d; font-style: italic;")

        layout.addWidget(self.class_label)
        layout.addWidget(self.conf_label)
        layout.addWidget(self.time_label)

    def update_result(self, result: dict):
        if 'error' in result:
            self.class_label.setText("Error")
            self.class_label.setStyleSheet("color: #e74c3c; padding: 6px 0;")
            self.conf_label.setText(result['error'])
            self.time_label.setText("")
            return

        cls = result['class']
        color = CLASS_COLORS.get(cls, '#2ecc71')
        self.class_label.setText(cls)
        self.class_label.setStyleSheet(f"color: {color}; padding: 6px 0;")
        self.conf_label.setText(f"Confidence: {result['confidence'] * 100:.2f}%")
        self.time_label.setText(f"Inference time: {result['inference_time_ms']:.1f} ms")

    def reset(self):
        self.class_label.setText("—")
        self.class_label.setStyleSheet("color: #2ecc71; padding: 6px 0;")
        self.conf_label.setText("Confidence: —")
        self.time_label.setText("Inference time: —")


class SectionFrame(QFrame):
    """A styled container frame for grouping related widgets."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #1e2d3d;
                border: 1px solid #34495e;
                border-radius: 10px;
            }
        """)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(6)

        if title:
            lbl = QLabel(title.upper())
            lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
            lbl.setStyleSheet("color: #7f8c8d; border: none;")
            self._layout.addWidget(lbl)

    def add_widget(self, widget: QWidget):
        self._layout.addWidget(widget)
