"""Stage 1 of the hierarchical model: LSTM over each player's 9 cached features -> 9 actions.

Trained alone (stage-wise, as in the paper), then frozen inside gar.models.hier.
"""

import torch.nn as nn

from gar.data.datasets import PlayerSequenceDataset
from gar.data.splits import PERSON_ACTIONS


class PersonLSTM(nn.Module):
    def __init__(self, hidden, dropout, dim=2048):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, len(PERSON_ACTIONS)))

    def sequence(self, x):  # x: (N, T, 2048) -> hidden states (N, T, hidden)
        return self.lstm(x)[0]

    def forward(self, x):
        return self.head(self.sequence(x)[:, -1])


def build_model(cfg):
    return PersonLSTM(**cfg["model_args"])


def build_datasets(cfg, records):
    path = f"{cfg['paths']['features']}/{cfg['features']}"
    return {
        split: PlayerSequenceDataset([r for r in records if r["split"] == split], path)
        for split in ("train", "val", "test")
    }


CLASS_NAMES = PERSON_ACTIONS
