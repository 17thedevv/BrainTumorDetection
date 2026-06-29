from PIL import Image

# ImageNet mean/std used in training
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def load_and_preprocess(image_path: str, image_size: int = 224):
    """Load a single MRI image and preprocess it for inference.

    Args:
        image_path: Absolute or relative path to the image file.
        image_size: The spatial resolution used during training (224 for Research, 128 for Dev).

    Returns:
        A 4-D tensor of shape (1, 3, image_size, image_size).
    """
    # Lazy import to avoid DLL init at GUI startup
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # add batch dimension
    return tensor

