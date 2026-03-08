"""
Small CNN for place-name audio classification from mel spectrograms.
Input: (batch, 1, n_mels, time). Output: (batch, n_classes) logits.
"""

import torch
import torch.nn as nn


class PlaceCNN(nn.Module):
    """
    Conv2D backbone + global average pooling + linear classifier.
    Robust to variable mel time length via AdaptiveAvgPool2d(1).
    """

    def __init__(self, n_classes: int, n_mels: int = 64):
        super().__init__()
        self.n_classes = n_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, time) -> add channel -> (B, 1, n_mels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
