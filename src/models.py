import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)
from transformers import SegformerForSemanticSegmentation


class DoubleConv(nn.Module):
    def __init__(self, a, b, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(True),
            nn.Dropout2d(dropout),
            nn.Conv2d(b, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.e1 = DoubleConv(3, 32)
        self.e2 = DoubleConv(32, 64)
        self.e3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.b = DoubleConv(128, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d3 = DoubleConv(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2 = DoubleConv(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, n, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


class DeepLabWrapper(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.model.classifier[4] = nn.Conv2d(256, n, 1)

    def forward(self, x):
        return self.model(x)["out"]


class SegFormerWrapper(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=n,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        y = self.model(pixel_values=x).logits
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(name, n):
    if name == "unet":
        return SmallUNet(n)
    if name == "deeplabv3":
        return DeepLabWrapper(n)
    if name == "segformer_b0":
        return SegFormerWrapper(n)
    raise ValueError(name)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
