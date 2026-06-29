from torch.utils.data import DataLoader
from configs.config import DataConfig
from datasets.brain_mri_dataset import BrainMRIDataset, BrainMRIUnlabeledDataset
from datasets.transforms import get_train_transforms, get_val_transforms

class DataModule:
    def __init__(self, config: DataConfig):
        self.config = config
        
    def get_train_dataloader(self):
        transform = get_train_transforms(self.config.image_size)
        dataset = BrainMRIDataset(self.config.dataset1_train_path, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
    def get_val_dataloader(self):
        transform = get_val_transforms(self.config.image_size)
        dataset = BrainMRIDataset(self.config.dataset1_test_path, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
    def get_unlabeled_dataloader(self):
        transform = get_train_transforms(self.config.image_size)
        dataset = BrainMRIUnlabeledDataset(self.config.dataset2_path, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
