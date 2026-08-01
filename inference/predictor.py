import time
import os
from typing import Dict, List, Tuple

CLASS_NAMES: List[str] = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


class GradCAM:
    def __init__(self, model, target_layer, torch_module):
        self.model = model
        self.target_layer = target_layer
        self.torch = torch_module
        self.gradients = None
        self.activations = None
        
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        # Using register_full_backward_hook to avoid deprecation warnings
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
            
        loss = output[0, class_idx]
        loss.backward()
        
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Global average pooling of gradients
        weights = self.torch.mean(gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps
        grad_cam = self.torch.sum(weights * activations, dim=1).squeeze(0)
        
        # Apply ReLU
        grad_cam = self.torch.clamp(grad_cam, min=0)
        
        # Normalize
        grad_cam_max = grad_cam.max()
        if grad_cam_max > 0:
            grad_cam = grad_cam / grad_cam_max
            
        return grad_cam.cpu().numpy(), class_idx
        
    def release(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


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
        from models.cnn import BaselineCNN, ImprovedCNN

        self.image_size = image_size
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._BaselineCNN = BaselineCNN
        self._ImprovedCNN = ImprovedCNN

        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        torch = self._torch
        BaselineCNN = self._BaselineCNN
        ImprovedCNN = self._ImprovedCNN

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Detect architecture automatically from checkpoint state_dict keys
        is_improved = any(k.startswith('stem.') or k.startswith('layer1.') for k in state_dict.keys())
        if is_improved:
            model = ImprovedCNN(num_classes=self.num_classes, pretrained=False)
        else:
            model = BaselineCNN(num_classes=self.num_classes, pretrained=False)

        model.load_state_dict(state_dict)

        model.to(self.device)
        # Note: model must remain in eval mode, but we need gradients for Grad-CAM.
        # PyTorch requires requires_grad=True or we can temporarily enable it.
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
                - gradcam_heatmap (PIL.Image.Image): Grad-CAM visualization image.
        """
        import torch.nn.functional as F
        from inference.preprocessing import load_and_preprocess
        from PIL import Image
        import numpy as np
        import cv2

        torch = self._torch
        tensor = load_and_preprocess(image_path, self.image_size).to(self.device)
        
        # Ensure we can compute gradients for Grad-CAM
        tensor.requires_grad = True

        start = time.perf_counter()
        
        # 1. Run forward pass (with grad enabled for Grad-CAM)
        self.model.zero_grad()
        
        # Hook last conv layer for Grad-CAM (supports both ImprovedCNN and BaselineCNN)
        if hasattr(self.model, 'layer4'):
            target_layer = self.model.layer4[-1].conv2
        else:
            target_layer = self.model.features[-4]
        cam_extractor = GradCAM(self.model, target_layer, torch)
        
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()
        class_idx = int(torch.argmax(torch.tensor(probs)).item())
        
        # 2. Generate Grad-CAM heatmap
        heatmap_2d, _ = cam_extractor.generate(tensor, class_idx)
        cam_extractor.release()
        
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 3. Generate overlaid image
        # Load original image
        orig_img = Image.open(image_path).convert('RGB')
        orig_np = np.array(orig_img)
        h, w, _ = orig_np.shape

        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap_2d, (w, h))

        # Convert to uint8 and apply colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Superimpose the heatmap on input image
        alpha = 0.4
        overlaid_np = cv2.addWeighted(orig_np, 1 - alpha, heatmap_color, alpha, 0)

        # ── Draw red ellipse around the tumor region ──────────────────────
        # Only draw circle for classes that actually have a tumor
        if class_idx != 2:  # 2 = "No Tumor"
            # Threshold: keep only pixels with activation above 50% of max
            threshold = 0.5
            tumor_mask = (heatmap_resized >= threshold).astype(np.uint8)

            # Find contours of the high-activation region
            contours, _ = cv2.findContours(
                tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Pick the largest contour (main tumor area)
                largest = max(contours, key=cv2.contourArea)

                if len(largest) >= 5:
                    # Fit an ellipse to the contour
                    ellipse = cv2.fitEllipse(largest)
                    center, axes, angle = ellipse

                    # Expand ellipse slightly so it fully wraps the region
                    expand_ratio = 1.25
                    expanded_axes = (
                        max(int(axes[0] * expand_ratio / 2), 8),
                        max(int(axes[1] * expand_ratio / 2), 8),
                    )

                    center_int = (int(center[0]), int(center[1]))

                    # Draw outer glow (semi-transparent by blending) with a thick dark ring
                    cv2.ellipse(
                        overlaid_np,
                        center_int,
                        (expanded_axes[0] + 4, expanded_axes[1] + 4),
                        angle,
                        0, 360,
                        (180, 0, 0),  # dark red glow
                        thickness=6,
                    )
                    # Draw bright red ellipse border
                    cv2.ellipse(
                        overlaid_np,
                        center_int,
                        expanded_axes,
                        angle,
                        0, 360,
                        (255, 50, 50),  # bright red
                        thickness=3,
                    )
                else:
                    # Fallback: use bounding rect → draw a circle
                    x, y, bw, bh = cv2.boundingRect(largest)
                    cx = x + bw // 2
                    cy = y + bh // 2
                    radius = int(max(bw, bh) * 0.65)
                    cv2.circle(overlaid_np, (cx, cy), radius + 4, (180, 0, 0), 6)
                    cv2.circle(overlaid_np, (cx, cy), radius, (255, 50, 50), 3)
        # ── End of tumor circle ────────────────────────────────────────────

        gradcam_img = Image.fromarray(overlaid_np)

        return {
            'class': CLASS_NAMES[class_idx],
            'class_idx': class_idx,
            'confidence': probs[class_idx],
            'probabilities': list(zip(CLASS_NAMES, probs)),
            'inference_time_ms': elapsed_ms,
            'gradcam_heatmap': gradcam_img,
        }
