import os
import glob
import random
from collections import defaultdict
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from configs.config import Config
from datasets.brain_mri_dataset import BrainMRIDataset, UnlabeledDataset
from datasets.transforms import get_train_transforms, get_val_transforms


class DataModule:
    """Quản lý DataLoaders cho pipeline SSL.

    Phân chia Dataset1 Training thành 2 phần:
      - labeled_pool  : `labeled_per_class` ảnh mỗi class (stratified)
      - unlabeled_pool: phần còn lại của Dataset1 + toàn bộ Dataset2

    get_labeled_dataloader()   → 1000 ảnh có nhãn thật (dùng WeightedRandomSampler)
    get_unlabeled_dataloader() → ~7600 ảnh không nhãn (dùng cho pseudo-labeling)
    get_val_dataloader()       → Dataset1 Testing (1600 ảnh, giữ nguyên)
    """

    CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

    def __init__(self, config: Config):
        self.config = config
        self.data_cfg = config.data
        self.mode_cfg = config.active_mode
        self.ssl_cfg = config.ssl

        # Thực hiện split ngay khi khởi tạo để đảm bảo nhất quán
        self._labeled_indices, self._unlabeled_indices = self._split_labeled_unlabeled()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_labeled_unlabeled(self) -> Tuple[List[int], List[int]]:
        """Stratified split: lấy đúng `labeled_per_class` ảnh mỗi class làm labeled."""
        full_dataset = BrainMRIDataset(self.data_cfg.dataset1_train_path)

        class_indices: dict = defaultdict(list)
        for idx, label in enumerate(full_dataset.labels):
            class_indices[label].append(idx)

        labeled_indices: List[int] = []
        unlabeled_indices: List[int] = []

        n_per_class = self.ssl_cfg.labeled_per_class

        for label, indices in class_indices.items():
            random.shuffle(indices)
            labeled_indices.extend(indices[:n_per_class])
            unlabeled_indices.extend(indices[n_per_class:])

        return labeled_indices, unlabeled_indices

    def _collect_dataset2_paths(self) -> List[str]:
        """Thu thập đường dẫn ảnh từ Dataset2 (yes/ + no/)."""
        paths: List[str] = []
        ds2_root = self.data_cfg.dataset2_path
        for root, _, files in os.walk(ds2_root):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    paths.append(os.path.join(root, f))
        return paths

    # ------------------------------------------------------------------
    # Public DataLoaders
    # ------------------------------------------------------------------

    def get_labeled_dataloader(self) -> DataLoader:
        """1000 ảnh có nhãn thật (stratified, 250/class).

        Dùng WeightedRandomSampler để đảm bảo mỗi batch đều cân bằng class,
        đặc biệt hữu ích khi labeled set nhỏ.
        """
        transform = get_train_transforms(self.mode_cfg.image_size)
        full_dataset = BrainMRIDataset(self.data_cfg.dataset1_train_path, transform=transform)
        labeled_dataset = Subset(full_dataset, self._labeled_indices)

        # Tính class weights để cân bằng sampling
        labels = [full_dataset.labels[i] for i in self._labeled_indices]
        class_counts = defaultdict(int)
        for lbl in labels:
            class_counts[lbl] += 1
        weights = [1.0 / class_counts[lbl] for lbl in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        return DataLoader(
            labeled_dataset,
            batch_size=self.mode_cfg.batch_size,
            sampler=sampler,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
        )

    def get_unlabeled_dataloader(self) -> DataLoader:
        """~7600 ảnh không nhãn: phần dư Dataset1 + toàn bộ Dataset2."""
        transform = get_val_transforms(self.mode_cfg.image_size)

        # Phần dư từ Dataset1
        full_dataset_noxform = BrainMRIDataset(self.data_cfg.dataset1_train_path)
        ds1_unlabeled_paths = [full_dataset_noxform.image_paths[i] for i in self._unlabeled_indices]

        # Toàn bộ Dataset2
        ds2_paths = self._collect_dataset2_paths()

        all_unlabeled_paths = ds1_unlabeled_paths + ds2_paths
        unlabeled_dataset = UnlabeledDataset(all_unlabeled_paths, transform=transform)

        return DataLoader(
            unlabeled_dataset,
            batch_size=self.mode_cfg.batch_size * 2,  # Batch lớn hơn để pseudo-label nhanh
            shuffle=False,  # Không shuffle để lưu đúng index
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
        )

    def get_val_dataloader(self) -> DataLoader:
        """Dataset1 Testing — 1600 ảnh, không thay đổi."""
        transform = get_val_transforms(self.mode_cfg.image_size)
        dataset = BrainMRIDataset(self.data_cfg.dataset1_test_path, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.mode_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
        )

    def get_info(self) -> dict:
        """Trả về thông tin về số lượng ảnh trong mỗi split."""
        full_dataset_noxform = BrainMRIDataset(self.data_cfg.dataset1_train_path)
        ds1_unlabeled_count = len(self._unlabeled_indices)
        ds2_count = len(self._collect_dataset2_paths())
        val_dataset = BrainMRIDataset(self.data_cfg.dataset1_test_path)

        return {
            'labeled': len(self._labeled_indices),
            'unlabeled_ds1': ds1_unlabeled_count,
            'unlabeled_ds2': ds2_count,
            'unlabeled_total': ds1_unlabeled_count + ds2_count,
            'val': len(val_dataset),
        }
