"""
Entry point for the Brain Tumor Detection Inference GUI.

Usage:
    python gui_app.py
"""

import sys

# ---------------------------------------------------------------
# IMPORTANT: torch MUST be imported before PyQt5 on Windows to
# prevent WinError 1114 (DLL initialization failure).
# ---------------------------------------------------------------
try:
    import torch  # noqa: F401
except Exception:
    pass  # If torch is not available, the error will surface later in the UI

from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Brain Tumor Detection")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
