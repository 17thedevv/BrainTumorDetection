from torchvision import transforms


def get_train_transforms(image_size: int):
    """Augmentation pipeline mạnh cho training.

    So với phiên bản cũ (chỉ Flip + Rotation 15° + ColorJitter nhẹ),
    phiên bản mới thêm:
      - RandomResizedCrop   : crop ngẫu nhiên thay vì resize exact → đa dạng vùng nhìn
      - RandomVerticalFlip  : MRI không có hướng cố định
      - RandomAffine        : dịch chuyển + scale nhẹ
      - GaussianBlur        : mô phỏng khác biệt scanner/nhiễu
      - RandomErasing       : CHỦ CHỐT — buộc model học đặc trưng phân tán,
                              phá vỡ shortcut learning (bám viền sọ)
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])


def get_val_transforms(image_size: int):
    """Validation/Test transforms — chỉ resize + normalize, không augment."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
