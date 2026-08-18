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

Cải tiến v2:
    - Label Smoothing (0.1) — giảm overfitting, cải thiện calibration
    - Gradient Clipping (max_norm=1.0) — ổn định khi train với pseudo-labels
    - Mixup Training (alpha=0.2) — regularization mạnh ở level data
    - Confidence-weighted pseudo-labels — sample weight theo confidence
"""
import time
import logging
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
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
# Helper: Mixup
# ---------------------------------------------------------------------------

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Mixup augmentation: trộn 2 sample theo tỷ lệ lambda.

    Tạo ra samples trung gian giữa 2 ảnh, giúp model học decision boundary
    mượt hơn và giảm overfitting đáng kể.

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor,
                    y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Loss cho mixup: kết hợp loss 2 targets theo tỷ lệ lambda."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# Weighted Pseudo-Label Dataset
# ---------------------------------------------------------------------------

class WeightedPseudoLabelDataset(Dataset):
    """Pseudo-label dataset với sample weight theo confidence.

    Thay vì coi mọi pseudo-label ngang nhau, weight = confidence score
    giúp model tin tưởng nhiều hơn vào pseudo-labels chất lượng cao.
    """

    def __init__(self, image_tensors: List[torch.Tensor],
                 pseudo_labels: List[int],
                 weights: List[float]):
        assert len(image_tensors) == len(pseudo_labels) == len(weights)
        self.image_tensors = image_tensors
        self.pseudo_labels = pseudo_labels
        self.weights = weights

    def __len__(self) -> int:
        return len(self.image_tensors)

    def __getitem__(self, idx) -> Tuple:
        return self.image_tensors[idx], self.pseudo_labels[idx], self.weights[idx]


# ---------------------------------------------------------------------------
# SSL Trainer
# ---------------------------------------------------------------------------

class SSLTrainer:
    """Trainer cho Semi-Supervised Learning với Pseudo-Labeling Curriculum.

    Cải tiến v2:
        - Label Smoothing 0.1
        - Gradient Clipping max_norm=1.0
        - Mixup Training alpha=0.2
        - Confidence-weighted pseudo-labels

    Attributes:
        model   : nn.Module (ImprovedCNN)
        device  : 'cuda' hoặc 'cpu'
        logger  : logging.Logger
    """

    def __init__(self, model: nn.Module, device: str, logger: logging.Logger,
                 use_mixup: bool = True, mixup_alpha: float = 0.2,
                 label_smoothing: float = 0.1, max_grad_norm: float = 1.0):
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        self.num_classes = 4
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

        # Mixup config
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        # Gradient clipping
        self.max_grad_norm = max_grad_norm

        # [B3] Label Smoothing — giảm overfitting, đặc biệt hữu ích với pseudo-labels
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.logger.info(f"Label smoothing: {label_smoothing} | Mixup: {use_mixup} (α={mixup_alpha}) | Grad clip: {max_grad_norm}")

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
        """Train 1 epoch trên bất kỳ dataloader nào (labeled hoặc combined).

        Cải tiến:
          - Mixup augmentation (nếu use_mixup=True)
          - Gradient clipping sau backward
        """
        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        start_time = time.time()

        pbar = tqdm(dataloader, desc=f"  [{phase_label}]", unit="batch", leave=False,
                    bar_format="{l_bar}{bar:30}{r_bar}")

        for batch in pbar:
            # Hỗ trợ cả 2-tuple (image, label) và 3-tuple (image, label, weight)
            if len(batch) == 3:
                inputs, labels, sample_weights = batch
            else:
                inputs, labels = batch
                sample_weights = None

            # Bỏ qua batch chỉ có pseudo-label nhưng labels = -1 (shouldn't happen)
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue
            inputs = inputs[valid_mask].to(self.device)
            labels = labels[valid_mask].to(self.device)
            if sample_weights is not None:
                sample_weights = sample_weights[valid_mask].to(self.device)

            optimizer.zero_grad()

            # [A2] Mixup training
            if self.use_mixup and sample_weights is None:
                # Chỉ mixup trên labeled data (không mixup pseudo-labeled để tránh nhiễu)
                mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, self.mixup_alpha)
                if self.use_amp:
                    with autocast():
                        outputs = self.model(mixed_inputs)
                        loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    # [C3] Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(mixed_inputs)
                    loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()

                # Accuracy tracking dùng unmixed predictions (gần đúng)
                with torch.no_grad():
                    unmixed_outputs = self.model(inputs)
                    _, preds = torch.max(unmixed_outputs, 1)
            else:
                # Standard training (cho pseudo-labeled hoặc khi mixup tắt)
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        if sample_weights is not None:
                            # Confidence-weighted loss cho pseudo-labels
                            per_sample_loss = F.cross_entropy(outputs, labels, reduction='none',
                                                              label_smoothing=0.1)
                            loss = (sample_weights * per_sample_loss).mean()
                        else:
                            loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    if sample_weights is not None:
                        per_sample_loss = F.cross_entropy(outputs, labels, reduction='none',
                                                          label_smoothing=0.1)
                        loss = (sample_weights * per_sample_loss).mean()
                    else:
                        loss = self.criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()

                _, preds = torch.max(outputs, 1)

            total_loss += loss.item()
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
    ) -> Tuple['WeightedPseudoLabelDataset', Dict[str, Any]]:
        """Forward pass tren unlabeled data, giu anh co confidence >= threshold.

        Cải tiến v2: lưu kèm confidence score làm sample weight.
        Co them per-class quota de tranh mat can bang pseudo-labels.
        Moi class chi giu toi da `max_per_class` anh co confidence cao nhat.

        Returns:
            pseudo_dataset : WeightedPseudoLabelDataset (có weight)
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
        selected_weights: List[float] = []
        per_class_count: Dict[int, int] = {}

        for lbl, cands in candidates.items():
            cands.sort(key=lambda x: x[0], reverse=True)
            chosen = cands[:max_per_class]
            per_class_count[lbl] = len(chosen)
            for conf, tensor in chosen:
                selected_tensors.append(tensor)
                selected_labels.append(lbl)
                selected_weights.append(conf)  # [D1] confidence as weight

        pseudo_dataset = WeightedPseudoLabelDataset(
            selected_tensors, selected_labels, selected_weights
        )

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
        pseudo_dataset: 'WeightedPseudoLabelDataset',
        batch_size: int,
        num_workers: int = 2,
    ) -> DataLoader:
        """Kết hợp labeled dataset + pseudo-labeled dataset thành 1 DataLoader.

        Lưu ý: chỉ gọi khi pseudo_dataset không rỗng.
        Labeled samples sẽ trả (image, label) 2-tuple.
        Pseudo samples sẽ trả (image, label, weight) 3-tuple.
        train_epoch() tự xử lý cả 2 format.
        """
        labeled_base = labeled_loader.dataset  # Subset of BrainMRIDataset
        combined = ConcatDataset([labeled_base, pseudo_dataset])

        def collate_fn(batch):
            """Collate cho mixed 2-tuple và 3-tuple samples."""
            images = []
            labels = []
            weights = []
            has_weights = False
            for item in batch:
                images.append(item[0])
                labels.append(item[1])
                if len(item) == 3:
                    weights.append(item[2])
                    has_weights = True
                else:
                    weights.append(1.0)  # labeled data: weight = 1.0

            images = torch.stack(images)
            labels = torch.tensor(labels, dtype=torch.long)
            if has_weights:
                weights = torch.tensor(weights, dtype=torch.float32)
                return images, labels, weights
            return images, labels

        return DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
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
