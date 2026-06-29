import time
import os
from typing import Dict, List, Tuple

CLASS_NAMES: List[str] = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


class Predictor:
    """Handles model loading and single-image inference for the GUI.

    All heavy imports (torch, models) are deferred until the model is
    actually loaded, so the GUI can start up without triggering DLL
    initialization on import.

    Usage::

        predictor = Predictor(model_path="saved_model/best_model.pth")
        result = predictor.predict("path/to/mri.jpg")
        print(result['class'], result['confidence'], result['inference_time_ms'])
    """

    def __init__(self, model_path: str, image_size: int = 224, num_classes: int = 4):
        # Lazy import — deferred until model is actually loaded
        import torch
        from models.cnn import BaselineCNN

        self.image_size = image_size
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._BaselineCNN = BaselineCNN

        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        torch = self._torch
        BaselineCNN = self._BaselineCNN

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = BaselineCNN(num_classes=self.num_classes, pretrained=False)

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path: str) -> Dict:
        """Run inference on a single image.

        Args:
            image_path: Path to the MRI image file.

        Returns:
            A dict with keys:
                - class (str): Predicted class name.
                - class_idx (int): Index of predicted class.
                - confidence (float): Confidence of predicted class (0-1).
                - probabilities (List[Tuple[str, float]]): Probabilities per class.
                - inference_time_ms (float): Inference time in milliseconds.
        """
        import torch.nn.functional as F
        from inference.preprocessing import load_and_preprocess

        torch = self._torch
        tensor = load_and_preprocess(image_path, self.image_size).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000

        probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()
        class_idx = int(torch.argmax(torch.tensor(probs)).item())

        return {
            'class': CLASS_NAMES[class_idx],
            'class_idx': class_idx,
            'confidence': probs[class_idx],
            'probabilities': list(zip(CLASS_NAMES, probs)),
            'inference_time_ms': elapsed_ms,
            # Grad-CAM hook (Phase 3): attach here
            'gradcam_heatmap': None,
        }
