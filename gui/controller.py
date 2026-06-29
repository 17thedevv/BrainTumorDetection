"""
controller.py - Bridge between the GUI and the inference module.
All business logic lives here; widgets only call Controller methods.
"""

import os
from typing import Optional
from inference.predictor import Predictor


class Controller:
    """Connects MainWindow with the Predictor inference engine."""

    def __init__(self):
        self._predictor: Optional[Predictor] = None
        self._model_path: str = ""
        self._image_path: str = ""

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_model(self, model_path: str, image_size: int = 224) -> str:
        """Load a trained model checkpoint.

        Returns:
            A status message string (success or error).
        """
        if not os.path.exists(model_path):
            return f"Error: File not found — {model_path}"
        try:
            self._predictor = Predictor(model_path=model_path, image_size=image_size)
            self._model_path = model_path
            return f"Model loaded successfully: {os.path.basename(model_path)}"
        except Exception as exc:
            self._predictor = None
            return f"Error loading model: {exc}"

    def is_model_loaded(self) -> bool:
        return self._predictor is not None

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------

    def set_image(self, image_path: str) -> str:
        """Register the selected image path.

        Returns:
            Status message.
        """
        if not os.path.exists(image_path):
            return f"Error: Image not found — {image_path}"
        self._image_path = image_path
        return f"Image selected: {os.path.basename(image_path)}"

    def get_image_path(self) -> str:
        return self._image_path

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference(self) -> dict:
        """Run prediction on the currently selected image.

        Returns:
            Result dict from Predictor.predict(), or an error dict.
        """
        if not self.is_model_loaded():
            return {"error": "No model loaded. Please load a .pth file first."}
        if not self._image_path:
            return {"error": "No image selected. Please select an MRI image."}
        try:
            return self._predictor.predict(self._image_path)
        except Exception as exc:
            return {"error": str(exc)}
