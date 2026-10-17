"""B4: B1-tuned image features over the 9 frames -> LSTM -> 8 classes (paper: 63.1%, below B1)."""

import torch.nn as nn

from gar.data.datasets import ImageFeatureDataset
from gar.data.splits import GROUP_ACTIVITIES


class B4(nn.Module):
    def __init__(self, hidden, dropout, dim=2048):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, len(GROUP_ACTIVITIES)))

    def forward(self, x):  # x: (B, T, 2048)
        return self.head(self.lstm(x)[0][:, -1])


def build_model(cfg):
    return B4(**cfg["model_args"])


def build_datasets(cfg, records):
    path = f"{cfg['paths']['features']}/{cfg['features']}"
    return {
        split: ImageFeatureDataset([r for r in records if r["split"] == split], path)
        for split in ("train", "val", "test")
    }


CLASS_NAMES = GROUP_ACTIVITIES
