from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import random
from configs.config import Config
from datasets.brain_mri_dataset import BrainMRIDataset, BrainMRIUnlabeledDataset
from datasets.transforms import get_train_transforms, get_val_transforms

class DataModule:
    def __init__(self, config: Config):
        self.config = config
        self.data_cfg = config.data
        self.mode_cfg = config.active_mode
        
    def get_train_dataloader(self):
        transform = get_train_transforms(self.mode_cfg.image_size)
        dataset = BrainMRIDataset(self.data_cfg.dataset1_train_path, transform=transform)
        
        subset_ratio = self.mode_cfg.subset_ratio
        if subset_ratio < 1.0:
            class_indices = defaultdict(list)
            for idx, label in enumerate(dataset.labels):
                class_indices[label].append(idx)
            
            subset_indices = []
            for label, indices in class_indices.items():
                num_samples = int(len(indices) * subset_ratio)
                subset_indices.extend(random.sample(indices, num_samples))
                
            dataset = Subset(dataset, subset_indices)
            
        return DataLoader(
            dataset,
            batch_size=self.mode_cfg.batch_size,
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True
        )
        
    def get_val_dataloader(self):
        transform = get_val_transforms(self.mode_cfg.image_size)
        dataset = BrainMRIDataset(self.data_cfg.dataset1_test_path, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.mode_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True
        )
        
    def get_unlabeled_dataloader(self):
        transform = get_train_transforms(self.mode_cfg.image_size)
        dataset = BrainMRIUnlabeledDataset(self.data_cfg.dataset2_path, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.mode_cfg.batch_size,
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True
        )
