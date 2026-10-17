"""B3: fine-tuned person features of the target frame, max-pooled over players -> MLP -> 8 classes.

Step A (the person classifier) is gar.models.person, step B is gar.features.extractor; this module
is step C. `groups: 2` turns it into the team-pooled variant, for the ablation.
"""

import torch.nn as nn

from gar.data.datasets import FeatureDataset
from gar.data.splits import GROUP_ACTIVITIES
from gar.models.pooling import group_pool

TARGET_FRAME = 5  # index of the target frame among the 9 tracked frames (target-5 .. target+3)


class B3(nn.Module):
    def __init__(self, groups, pool, hidden, dropout, dim=2048):
        super().__init__()
        self.groups, self.pool = groups, pool
        if hidden:
            self.head = nn.Sequential(
                nn.Linear(groups * dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, len(GROUP_ACTIVITIES)),
            )
        else:  # hidden: 0 -> a plain softmax classifier, as in the paper's B2 / B3
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(groups * dim, len(GROUP_ACTIVITIES)))

    def forward(self, x, mask):  # x: (B, 12, 2048), mask: (B, 12)
        return self.head(group_pool(x, mask, self.groups, self.pool))


def build_model(cfg):
    return B3(**cfg["model_args"])


def build_datasets(cfg, records):
    path = f"{cfg['paths']['features']}/{cfg['features']}"
    return {
        split: FeatureDataset([r for r in records if r["split"] == split], path, frames=[TARGET_FRAME])
        for split in ("train", "val", "test")
    }


CLASS_NAMES = GROUP_ACTIVITIES
