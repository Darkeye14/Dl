import torch
import torch.nn as nn
from torchvision import models


class MNISTSmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(dataset: str) -> nn.Module:
    if dataset == "mnist":
        return MNISTSmallCNN(num_classes=10)
    elif dataset == "cifar10":
        try:
            # Torchvision >= 0.13 style
            return models.resnet18(weights=None, num_classes=10)
        except TypeError:
            # Older torchvision
            return models.resnet18(pretrained=False, num_classes=10)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
