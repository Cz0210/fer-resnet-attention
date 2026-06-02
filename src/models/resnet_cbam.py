"""ResNet18/34 classifiers with optional SE or CBAM attention."""

from __future__ import annotations

import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.shared_mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        max_pool = self.shared_mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        return x * self.sigmoid(avg + max_pool)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg, max_pool], dim=1)))
        return x * attention


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


def _make_torchvision_resnet(arch: str, pretrained: bool):
    from torchvision import models

    try:
        if arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            return models.resnet18(weights=weights)
        if arch == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            return models.resnet34(weights=weights)
    except AttributeError:
        if arch == "resnet18":
            return models.resnet18(pretrained=pretrained)
        if arch == "resnet34":
            return models.resnet34(pretrained=pretrained)
    raise ValueError(f"Unsupported ResNet architecture: {arch}")


def _adapt_first_conv(backbone: nn.Module, in_channels: int) -> None:
    if in_channels == backbone.conv1.in_channels:
        return
    old_conv = backbone.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        if in_channels == 1:
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        else:
            repeat = int(in_channels / old_conv.in_channels) + 1
            new_conv.weight.copy_(old_conv.weight.repeat(1, repeat, 1, 1)[:, :in_channels] / repeat)
    backbone.conv1 = new_conv


class ResNetAttentionClassifier(nn.Module):
    def __init__(
        self,
        arch: str = "resnet18",
        num_classes: int = 7,
        attention_type: str = "none",
        pretrained: bool = True,
        in_channels: int = 3,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.attention_type = attention_type.lower()
        self.backbone = _make_torchvision_resnet(arch, pretrained=pretrained)
        _adapt_first_conv(self.backbone, in_channels)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        channels = [64, 128, 256, 512]
        self.attentions = nn.ModuleList([self._make_attention(ch, reduction) for ch in channels])
        self.classifier = nn.Linear(in_features, num_classes)

    def _make_attention(self, channels: int, reduction: int) -> nn.Module:
        if self.attention_type == "none":
            return nn.Identity()
        if self.attention_type == "se":
            return SEModule(channels, reduction=reduction)
        if self.attention_type == "cbam":
            return CBAM(channels, reduction=reduction)
        raise ValueError(f"Unsupported attention_type: {self.attention_type}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = self.attentions[0](b.layer1(x))
        x = self.attentions[1](b.layer2(x))
        x = self.attentions[2](b.layer3(x))
        x = self.attentions[3](b.layer4(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_resnet_classifier(
    arch: str = "resnet18",
    num_classes: int = 7,
    attention_type: str = "none",
    pretrained: bool = True,
    in_channels: int = 3,
) -> ResNetAttentionClassifier:
    return ResNetAttentionClassifier(
        arch=arch,
        num_classes=num_classes,
        attention_type=attention_type,
        pretrained=pretrained,
        in_channels=in_channels,
    )

