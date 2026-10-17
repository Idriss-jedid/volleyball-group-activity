"""ImageNet backbones with a swappable classification head.

| name              | feature dim | params |
|-------------------|-------------|--------|
| alexnet           | 4096 (fc7)  | 61M    |
| resnet50          | 2048        | 25M    |
| convnext_tiny     | 768         | 28M    |
| efficientnet_v2_s | 1280        | 21M    |
"""

import torch.nn as nn
from torchvision import models

BACKBONES = {
    "alexnet": (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1),  # the paper's backbone
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1),
}


def _head_slot(model, name):
    """(parent module, attribute) of the final Linear layer."""
    if name == "resnet50":
        return model, "fc"
    # alexnet (its fc8, so the stripped output is fc7) / convnext / efficientnet: last of `classifier`
    return model.classifier, str(len(model.classifier) - 1)


def build_backbone(name, num_classes, dropout):
    build, weights = BACKBONES[name]
    model = build(weights=weights)
    parent, attr = _head_slot(model, name)
    linear = getattr(parent, attr)
    setattr(parent, attr, nn.Sequential(nn.Dropout(dropout), nn.Linear(linear.in_features, num_classes)))
    return model


def strip_head(model, name):
    """Replace the classification head with Identity, in place; returns the feature dimension."""
    parent, attr = _head_slot(model, name)
    head = getattr(parent, attr)
    dim = head[-1].in_features if isinstance(head, nn.Sequential) else head.in_features
    setattr(parent, attr, nn.Identity())
    return dim
