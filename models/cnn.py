import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# BaselineCNN — Giữ nguyên để so sánh baseline
# ===========================================================================

class BaselineCNN(nn.Module):
    """Custom CNN 5-block xây dựng hoàn toàn từ đầu (from scratch).

    Dùng làm baseline để so sánh với ImprovedCNN.
    """

    def __init__(self, name: str = 'custom_cnn', num_classes: int = 4, pretrained: bool = False):
        super().__init__()
        self.name = name.lower()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# ===========================================================================
# ImprovedCNN — ResNet-18 style, from scratch, không dùng pretrained weights
# ===========================================================================

class _ResidualBlock(nn.Module):
    """BasicBlock kiểu ResNet với residual (skip) connection.

    Cấu trúc:
        Conv → BN → ReLU → Conv → BN
        + residual (identity hoặc projection nếu stride/channel khác)
        → ReLU
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut nếu dimension thay đổi
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # Residual connection
        out = F.relu(out, inplace=True)
        return out


class ImprovedCNN(nn.Module):
    """ResNet-18-style CNN xây dựng hoàn toàn từ đầu (from scratch).

    Kiến trúc:
        Stem: Conv(3→64, 7×7, stride=2) → BN → ReLU → MaxPool(3×3, stride=2)
        Layer1: 2 × ResidualBlock(64→64)
        Layer2: 2 × ResidualBlock(64→128, stride=2)
        Layer3: 2 × ResidualBlock(128→256, stride=2)
        Layer4: 2 × ResidualBlock(256→512, stride=2)
        AdaptiveAvgPool2d(1×1) → Flatten
        FC(512→256) → ReLU → Dropout(0.4) → FC(256→num_classes)

    Tổng tham số: ~11M (tương đương ResNet-18 gốc nhưng không có pretrained weights).
    Target layer cho Grad-CAM: layer4[-1] (conv cuối của stage 4).
    """

    def __init__(self, name: str = 'improved_cnn', num_classes: int = 4, pretrained: bool = False):
        super().__init__()
        self.name = name.lower()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 4 Stages với Residual Blocks
        self.layer1 = self._make_layer(64, 64, n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

        # Khởi tạo trọng số theo chuẩn He (Kaiming)
        self._initialize_weights()

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [_ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, n_blocks):
            layers.append(_ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    @property
    def grad_cam_target_layer(self) -> nn.Module:
        """Layer dùng để hook trong Grad-CAM (conv cuối stage 4)."""
        return self.layer4[-1].conv2


# ===========================================================================
# Factory function
# ===========================================================================

def build_model(name: str, num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    """Tạo model theo tên. pretrained=True bị bỏ qua (all from scratch)."""
    name = name.lower()
    if name in ('improved_cnn', 'resnet18_scratch', 'improved'):
        return ImprovedCNN(name=name, num_classes=num_classes, pretrained=False)
    elif name in ('custom_cnn', 'baseline_cnn', 'baseline'):
        return BaselineCNN(name=name, num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model name: '{name}'. Choices: improved_cnn, custom_cnn")


if __name__ == '__main__':
    # Quick test
    x = torch.randn(2, 3, 224, 224)

    baseline = BaselineCNN(num_classes=4)
    improved = ImprovedCNN(num_classes=4)

    print(f"BaselineCNN params  : {sum(p.numel() for p in baseline.parameters()):,}")
    print(f"ImprovedCNN params  : {sum(p.numel() for p in improved.parameters()):,}")
    print(f"BaselineCNN output  : {baseline(x).shape}")
    print(f"ImprovedCNN output  : {improved(x).shape}")
