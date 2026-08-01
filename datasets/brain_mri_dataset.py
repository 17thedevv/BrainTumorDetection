import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, List, Tuple


class BrainMRIDataset(Dataset):
    """Dataset 1 (Supervised) — 4-class labeled data.

    Duyệt theo 4 thư mục class: glioma, meningioma, notumor, pituitary.
    Trả về (image, label).
    """

    CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(self.CLASSES)}

        self.image_paths: List[str] = []
        self.labels: List[int] = []

        for cls_name in self.CLASSES:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.exists(cls_dir):
                print(f"Warning: directory {cls_dir} does not exist.")
                continue
            for img_path in glob.glob(os.path.join(cls_dir, '*.*')):
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class UnlabeledDataset(Dataset):
    """Dataset không nhãn dùng cho SSL.

    Gộp:
    - Phần "dư" của Dataset1 Training không được chọn làm labeled
    - Toàn bộ Dataset2 (Br35H: yes/ + no/)
    Trả về (image, -1) — nhãn -1 nghĩa là chưa có nhãn thật.
    """

    def __init__(self, image_paths: List[str], transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Trả về ảnh trắng nếu đọc lỗi
            image = Image.new('RGB', (224, 224), color=0)
        if self.transform:
            image = self.transform(image)
        return image, -1  # -1 = no label


class PseudoLabelDataset(Dataset):
    """Dataset pseudo-labeled: gắn nhãn giả do model dự đoán.

    Dùng trong Phase B của SSL. Kết hợp với labeled_dataset qua ConcatDataset.
    image_tensors: list[Tensor] đã qua transform sẵn
    pseudo_labels: list[int] nhãn giả từ model
    """

    def __init__(self, image_tensors: List[torch.Tensor], pseudo_labels: List[int]):
        assert len(image_tensors) == len(pseudo_labels)
        self.image_tensors = image_tensors
        self.pseudo_labels = pseudo_labels

    def __len__(self) -> int:
        return len(self.image_tensors)

    def __getitem__(self, idx: int) -> Tuple:
        return self.image_tensors[idx], self.pseudo_labels[idx]


class BrainMRIUnlabeledDataset(Dataset):
    """Legacy — giữ để tương thích ngược với GUI / predictor cũ.

    Trả về (image, img_path).
    """

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths: List[str] = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path
