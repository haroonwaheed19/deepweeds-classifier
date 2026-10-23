"""Pretrained torchvision backbones, each returning (model, classifier_in_features)."""
from torch import nn
from torchvision import models

_BUILDERS = {
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
}


def available() -> list[str]:
    return list(_BUILDERS)


def build_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """Return the full torchvision model (original 1000-way head still attached)."""
    if name not in _BUILDERS:
        raise ValueError(f"unknown backbone '{name}', choose from {available()}")
    ctor, weights = _BUILDERS[name]
    return ctor(weights=weights if pretrained else None)


def strip_head(model: nn.Module, name: str) -> nn.Module:
    """Replace the classifier with identity so the model outputs pooled embeddings."""
    if name == "resnet18":
        model.fc = nn.Identity()
    elif name == "mobilenet_v3_large":
        # keep the 1280-d hidden layer, drop only the final linear
        model.classifier[-1] = nn.Identity()
    elif name == "efficientnet_b0":
        model.classifier = nn.Identity()
    else:
        raise ValueError(name)
    return model
