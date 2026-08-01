"""Semi-Supervised Learning Trainer — Pseudo-Labeling with Curriculum.

Pipeline 2 giai đoạn:
    Phase A (Warmup Supervised):
        Huấn luyện supervised trên labeled data (1000 ảnh có nhãn thật).
        Mục đích: khởi tạo model đủ tốt để tạo pseudo-label chất lượng.

    Phase B (SSL Pseudo-Labeling):
        Mỗi epoch:
        1. Generate pseudo-labels từ unlabeled data (confidence > threshold)
        2. Kết hợp labeled + pseudo-labeled → train
        3. Tăng threshold theo curriculum (0.85 → 0.95)
"""
import time
import logging
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm

from training.metrics import calculate_metrics
from datasets.brain_mri_dataset import PseudoLabelDataset


# ---------------------------------------------------------------------------
# Helper: Curriculum Threshold
# ---------------------------------------------------------------------------

def get_curriculum_threshold(
    ssl_epoch: int,
    ssl_epochs: int,
    start: float = 0.85,
    end: float = 0.95,
) -> float:
    """Tính ngưỡng confidence theo curriculum tuyến tính.

    ssl_epoch=1  → start (0.85)
    ssl_epoch=ssl_epochs → end (0.95)
    """
    progress = (ssl_epoch - 1) / max(ssl_epochs - 1, 1)
    return start + (end - start) * progress


# ---------------------------------------------------------------------------
# SSL Trainer
# ---------------------------------------------------------------------------

class SSLTrainer:
    """Trainer cho Semi-Supervised Learning với Pseudo-Labeling Curriculum.

    Attributes:
        model   : nn.Module (ImprovedCNN)
        device  : 'cuda' hoặc 'cpu'
        logger  : logging.Logger
    """

    def __init__(self, model: nn.Module, device: str, logger: logging.Logger):
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        self.num_classes = 4
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

        # Loss với class weights (sẽ cập nhật khi có pseudo-labels)
        self.criterion = nn.CrossEntropyLoss()

        # AMP — chỉ kích hoạt trên CUDA
        self.use_amp = (device == 'cuda')
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            self.logger.info("Automatic Mixed Precision (AMP) enabled.")

    # -----------------------------------------------------------------------
    # Core training methods
    # -----------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        phase_label: str = "Train",
    ) -> Dict[str, float]:
        """Train 1 epoch trên bất kỳ dataloader nào (labeled hoặc combined)."""
        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        start_time = time.time()

        pbar = tqdm(dataloader, desc=f"  [{phase_label}]", unit="batch", leave=False,
                    bar_format="{l_bar}{bar:30}{r_bar}")

        for inputs, labels in pbar:
            # Bỏ qua batch chỉ có pseudo-label nhưng labels = -1 (shouldn't happen)
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue
            inputs = inputs[valid_mask].to(self.device)
            labels = labels[valid_mask].to(self.device)

            optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        elapsed = time.time() - start_time
        metrics = calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / max(len(dataloader), 1)
        metrics['epoch_time'] = elapsed
        return metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Đánh giá model trên validation set."""
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        pbar = tqdm(dataloader, desc="  [Val]", unit="batch", leave=False,
                    bar_format="{l_bar}{bar:30}{r_bar}")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        metrics = calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / max(len(dataloader), 1)
        return metrics

    # -----------------------------------------------------------------------
    # Pseudo-Labeling
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        unlabeled_loader: DataLoader,
        threshold: float,
        max_per_class: int = 1500,
    ) -> Tuple[PseudoLabelDataset, Dict[str, Any]]:
        """Forward pass tren unlabeled data, giu anh co confidence >= threshold.

        Co them per-class quota de tranh mat can bang pseudo-labels.
        Moi class chi giu toi da `max_per_class` anh co confidence cao nhat.

        Returns:
            pseudo_dataset : PseudoLabelDataset
            stats          : dict thong ke
        """
        self.model.eval()

        # Thu thap candidates theo tung class: {label: [(confidence, tensor), ...]}
        candidates: Dict[int, List[Tuple[float, torch.Tensor]]] = defaultdict(list)
        total_unlabeled = 0

        for images, _ in tqdm(unlabeled_loader, desc="  [Pseudo-Label Gen]",
                               leave=False, bar_format="{l_bar}{bar:25}{r_bar}"):
            images = images.to(self.device)
            total_unlabeled += images.shape[0]

            if self.use_amp:
                with autocast():
                    logits = self.model(images)
            else:
                logits = self.model(images)

            probs = F.softmax(logits, dim=1)          # (B, C)
            max_probs, pseudo_cls = probs.max(dim=1)   # (B,)

            mask = max_probs >= threshold
            for i, keep in enumerate(mask):
                if keep:
                    lbl = pseudo_cls[i].item()
                    conf = max_probs[i].item()
                    candidates[lbl].append((conf, images[i].cpu()))

        # Per-class quota: sort giam dan theo confidence, lay toi da max_per_class
        selected_tensors: List[torch.Tensor] = []
        selected_labels: List[int] = []
        per_class_count: Dict[int, int] = {}

        for lbl, cands in candidates.items():
            cands.sort(key=lambda x: x[0], reverse=True)
            chosen = cands[:max_per_class]
            per_class_count[lbl] = len(chosen)
            for _, tensor in chosen:
                selected_tensors.append(tensor)
                selected_labels.append(lbl)

        pseudo_dataset = PseudoLabelDataset(selected_tensors, selected_labels)

        stats = {
            'total_unlabeled': total_unlabeled,
            'selected': len(selected_tensors),
            'selection_rate': len(selected_tensors) / max(total_unlabeled, 1),
            'threshold': threshold,
            'per_class_count': per_class_count,
        }
        return pseudo_dataset, stats

    def build_combined_loader(
        self,
        labeled_loader: DataLoader,
        pseudo_dataset: PseudoLabelDataset,
        batch_size: int,
        num_workers: int = 2,
    ) -> DataLoader:
        """Kết hợp labeled dataset + pseudo-labeled dataset thành 1 DataLoader.

        Lưu ý: chỉ gọi khi pseudo_dataset không rỗng.
        """
        labeled_base = labeled_loader.dataset  # Subset of BrainMRIDataset
        combined = ConcatDataset([labeled_base, pseudo_dataset])
        return DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    # -----------------------------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------------------------

    def save_checkpoint(self, state: dict, save_dir: str, filename: str):
        import os
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        torch.save(state, filepath)
        self.logger.info(f"Checkpoint saved → {filepath}")
