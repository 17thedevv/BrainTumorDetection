import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional

class BrainMRIDataset(Dataset):
    """Dataset 1 (Supervised Baseline)
    Handles 4-class classification.
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.exists(cls_dir):
                print(f"Warning: directory {cls_dir} does not exist.")
                continue
                
            for img_path in glob.glob(os.path.join(cls_dir, '*.*')):
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class BrainMRIUnlabeledDataset(Dataset):
    """Dataset 2 (Br35H - Unlabeled for SSL)
    Returns only the image and the path (ignoring the Yes/No labels).
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        # We recursively search for images in dataset 2 since it has yes/no subdirs
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_path
