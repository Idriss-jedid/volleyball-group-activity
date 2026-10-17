"""Hierarchical model and its ablations (course B5-B8), on cached player features.

    person stage   P_tk = x_tk ⊕ h_tk        h from the frozen person LSTM   (paper eq. 7)
                   or P_tk = x_tk            without the person LSTM
    pooling        Z_t  = pool over players, 1 group or 2 teams               (paper eq. 10-13)
    scene stage    FC -> scene LSTM -> last hidden -> classifier
                   or classifier on Z of the last frame, without the scene LSTM

| course | person_lstm | scene_lstm | groups | paper equivalent           |
|--------|-------------|------------|--------|----------------------------|
| B5     | yes         | no         | 1      | B7 (without LSTM 2)        |
| B6     | no          | yes        | 1      | B6 (without LSTM 1)        |
| B7     | yes         | yes        | 1      | full model, 1 group        |
| B8     | yes         | yes        | 2      | full model, 2 groups       |
"""

from pathlib import Path

import torch
import torch.nn as nn

from gar.data.datasets import FeatureDataset
from gar.data.splits import GROUP_ACTIVITIES
from gar.models.person_lstm import PersonLSTM
from gar.models.pooling import group_pool


class Hierarchical(nn.Module):
    def __init__(self, person_lstm, scene_lstm, groups, pool, fc_dim, scene_hidden, dropout, dim=2048):
        super().__init__()
        self.person, self.groups, self.pool = person_lstm, groups, pool
        person_dim = dim + (person_lstm.lstm.hidden_size if person_lstm else 0)
        self.proj = nn.Sequential(nn.Linear(groups * person_dim, fc_dim), nn.ReLU(), nn.Dropout(dropout))
        self.scene = nn.LSTM(fc_dim, scene_hidden, batch_first=True) if scene_lstm else None
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(scene_hidden if scene_lstm else fc_dim, len(GROUP_ACTIVITIES))
        )

    def train(self, mode=True):
        super().train(mode)
        if self.person is not None:
            self.person.eval()  # frozen stage 1: keep its dropout off
        return self

    def forward(self, x, mask):  # x: (B, T, K, D), mask: (B, K)
        b, t, k, d = x.shape
        if self.person is not None:
            with torch.no_grad():
                h = self.person.sequence(x.permute(0, 2, 1, 3).reshape(b * k, t, d))
            x = torch.cat([x, h.view(b, k, t, -1).permute(0, 2, 1, 3)], dim=-1)
        z = group_pool(x, mask.unsqueeze(1).expand(b, t, k), self.groups, self.pool)  # (B, T, G*D')
        z = self.proj(z)
        z = self.scene(z)[0][:, -1] if self.scene is not None else z[:, -1]
        return self.head(z)


def build_model(cfg):
    a = cfg["model_args"]
    person = None
    if a["person_lstm"]:
        pcfg = a["person_lstm"]
        person = PersonLSTM(hidden=pcfg["hidden"], dropout=0.0, dim=a.get("dim", 2048))
        state = torch.load(Path(cfg["paths"]["results"]) / pcfg["checkpoint"], map_location="cpu")["model"]
        person.load_state_dict(state)
        for p in person.parameters():
            p.requires_grad = False
    return Hierarchical(
        person,
        a["scene_lstm"],
        a["groups"],
        a["pool"],
        a["fc_dim"],
        a["scene_hidden"],
        a["dropout"],
        dim=a.get("dim", 2048),
    )


def build_datasets(cfg, records):
    path = f"{cfg['paths']['features']}/{cfg['features']}"
    return {
        split: FeatureDataset([r for r in records if r["split"] == split], path) for split in ("train", "val", "test")
    }


CLASS_NAMES = GROUP_ACTIVITIES
