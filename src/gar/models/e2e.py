"""End-to-end model (beyond the paper, and the course's challenge): one network, two losses.

    player crops of T frames -> CNN backbone (shared) -> per-player features
                             -> actor transformer over players x frames (+ box positions) -> group logits
                             -> person head on every player                              -> action logits

The paper trains its two stages separately; here the group loss also fine-tunes the CNN, and the
person-action loss (weight `person_weight`) keeps the features grounded in what each player does.
The backbone starts from the B3-A person classifier and gets a lower learning rate than the head.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from gar.data.datasets import ClipCropDataset
from gar.data.splits import GROUP_ACTIVITIES, PERSON_ACTIONS
from gar.models.backbones import build_backbone, strip_head
from gar.models.transformer import ActorTransformer


class EndToEnd(nn.Module):
    def __init__(self, backbone, frames, d_model, layers, heads, dropout, person_weight, backbone_lr_scale, init=None):
        super().__init__()
        self.backbone = build_backbone(backbone, len(PERSON_ACTIONS), dropout)
        if init:
            self.backbone.load_state_dict(torch.load(init, map_location="cpu")["model"])
        head = self.backbone.fc if backbone == "resnet50" else self.backbone.classifier[-1]
        self.person_head = head[-1]  # the person classifier's Linear layer, kept as the auxiliary head
        dim = strip_head(self.backbone, backbone)
        self.actor = ActorTransformer(d_model, layers, heads, dropout, use_boxes=True, dim=dim, frames=frames)
        self.dim, self.person_weight, self.backbone_lr_scale = dim, person_weight, backbone_lr_scale

    def param_groups(self, lr):
        backbone = list(self.backbone.parameters())
        ids = {id(p) for p in backbone}
        return [
            {"params": backbone, "lr": lr * self.backbone_lr_scale},
            {"params": [p for p in self.parameters() if id(p) not in ids], "lr": lr},
        ]

    def forward(self, crops, mask, boxes, actions):  # crops: (B, T, K, 3, H, W)
        b, t, k = crops.shape[:3]
        valid = mask[:, None, :].expand(b, t, k)
        feats = self.backbone(crops[valid])  # only real players go through the CNN
        grid = feats.new_zeros(b, t, k, self.dim)
        grid[valid] = feats
        logits = self.actor(grid, mask, boxes)
        if not self.training:
            return logits
        acts = actions[:, None, :].expand(b, t, k)[valid]
        aux = F.cross_entropy(self.person_head(feats), acts, ignore_index=-1)
        return logits, self.person_weight * aux


def build_model(cfg):
    a = dict(cfg["model_args"])
    if a.get("init"):
        a["init"] = Path(cfg["paths"]["results"]) / a["init"]
    a["frames"] = len(cfg["data"]["e2e_frames"])
    return EndToEnd(**a)


def build_datasets(cfg, records):
    args = dict(
        crops_dir=cfg["paths"]["crops"],
        frames=cfg["data"]["e2e_frames"],
        size=cfg["data"]["crop_size"],
        hflip=cfg["data"].get("hflip", False),
    )
    return {
        split: ClipCropDataset([r for r in records if r["split"] == split], train=(split == "train"), **args)
        for split in ("train", "val", "test")
    }


CLASS_NAMES = GROUP_ACTIVITIES
