"""Actor transformer (beyond the paper): self-attention over every player at every frame.

The 2016 model max-pools players, which throws away who is where and how players relate. Here each
(frame, player) is a token: its cached B3-A feature + its box position + its left-to-right rank +
its frame index. A CLS token attends over all T x K tokens and is classified into 8 activities.
`use_boxes: false` is the ablation without explicit geometry.
"""

import torch
import torch.nn as nn

from gar.data.datasets import TokenDataset
from gar.data.splits import GROUP_ACTIVITIES, MAX_PLAYERS


class ActorTransformer(nn.Module):
    def __init__(self, d_model, layers, heads, dropout, use_boxes=True, dim=2048, frames=9):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d_model))
        self.box = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model)) if use_boxes else None
        self.rank = nn.Embedding(MAX_PLAYERS, d_model)  # left-to-right slot: a proxy for the team side
        self.time = nn.Embedding(frames, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, len(GROUP_ACTIVITIES)))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x, mask, boxes):  # x: (B, T, K, D), mask: (B, K), boxes: (B, T, K, 4)
        b, t, k, _ = x.shape
        h = self.inp(x) + self.rank.weight[:k] + self.time.weight[:t, None]
        if self.box is not None:
            h = h + self.box(boxes)
        tokens = torch.cat([self.cls.expand(b, -1, -1), h.reshape(b, t * k, -1)], dim=1)
        pad = (~mask).unsqueeze(1).expand(b, t, k).reshape(b, t * k)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), pad], dim=1)
        return self.head(self.encoder(tokens, src_key_padding_mask=pad)[:, 0])


def build_model(cfg):
    return ActorTransformer(**cfg["model_args"])


def build_datasets(cfg, records):
    path = f"{cfg['paths']['features']}/{cfg['features']}"
    hflip = cfg["data"].get("hflip", False)
    return {
        split: TokenDataset([r for r in records if r["split"] == split], path, hflip=hflip and split == "train")
        for split in ("train", "val", "test")
    }


CLASS_NAMES = GROUP_ACTIVITIES
