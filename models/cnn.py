import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

class BaselineCNN(nn.Module):
    def __init__(self, name: str = 'resnet50', num_classes: int = 4, pretrained: bool = True):
        super(BaselineCNN, self).__init__()
        
        self.name = name.lower()
        if self.name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = resnet50(weights=weights)
        elif self.name == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported model name: {name}")

        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # Test model
    model = BaselineCNN(name='resnet50', num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
