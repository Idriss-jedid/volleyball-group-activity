"""PyTorch datasets: raw frames (zip or folder), player crops, cached features."""

import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from gar.data.frames import FrameSource
from gar.data.splits import ACTION_TO_IDX, ACTIVITY_TO_IDX, FLIP_ACTIVITY, GROUP_ACTIVITIES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def open_frame(source, video, clip, frame, draft_size=None):
    img = Image.open(io.BytesIO(source.frame(video, clip, frame)))
    if draft_size:
        # JPEG decodes at 1/2, 1/4 or 1/8 scale for free: ~3x faster than full-size decoding
        img.draft("RGB", draft_size)
    return img.convert("RGB")


class FrameDataset(Dataset):
    """Whole frames of each clip -> group activity label.

    `offsets` are relative to the target frame: (0,) for B1, range(-5, 4) for temporal baselines.
    Returns a tensor of shape (3, H, W) for one offset, (T, 3, H, W) otherwise.
    """

    def __init__(self, records, videos, offsets=(0,), size=(224, 384), train=False, hflip=True, augment=True):
        self.records = records
        self.source = FrameSource(videos)
        self.offsets = tuple(offsets)
        self.size = tuple(size)
        self.hflip = train and hflip and augment
        jitter = [transforms.ColorJitter(0.2, 0.2, 0.2)] if train and augment else []
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.size),
                *jitter,
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        draft = (self.size[1] * 2, self.size[0] * 2)
        frames = [open_frame(self.source, r["video"], r["clip"], r["clip"] + o, draft) for o in self.offsets]
        label = r["label"]
        # a mirrored court swaps the teams, so the label must swap too (l_spike <-> r_spike)
        if getattr(self, "force_flip", False) or (self.hflip and random.random() < 0.5):
            frames = [f.transpose(Image.FLIP_LEFT_RIGHT) for f in frames]
            label = ACTIVITY_TO_IDX[FLIP_ACTIVITY[GROUP_ACTIVITIES[label]]]
        x = torch.stack([self.transform(f) for f in frames])
        return (x[0] if len(self.offsets) == 1 else x), label

    @property
    def targets(self):
        return [r["label"] for r in self.records]


class CropReader:
    """Lazily opened per-video crop zips written by gar.data.crops."""

    def __init__(self, crops_dir):
        self.crops_dir = crops_dir
        self._zips = {}

    def __getstate__(self):
        return {"crops_dir": self.crops_dir, "_zips": {}}

    def crop(self, video, clip, frame, pid):
        if video not in self._zips:
            self._zips[video] = zipfile.ZipFile(f"{self.crops_dir}/{video}.zip")
        return Image.open(io.BytesIO(self._zips[video].read(f"{clip}/{frame}_{pid}.jpg"))).convert("RGB")


class PersonCropDataset(Dataset):
    """One sample per tracked player -> person action label (B3-A, the person classifier).

    Train: a random frame among the 9 tracked ones, so each epoch sees a different view of every
    player (the action label is the same on all 9 frames). Eval: always the target frame.
    """

    def __init__(self, records, crops_dir, size=(224, 112), train=False, augment=True):
        self.items = [(r, pid) for r in records for pid in r["tracks"]]
        self.reader = CropReader(crops_dir)
        self.train = train
        aug = (
            [
                transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.4, 0.6)),
                transforms.RandomHorizontalFlip(),  # an action does not change when mirrored
                transforms.ColorJitter(0.2, 0.2, 0.2),
            ]
            if train and augment
            else [transforms.Resize(size)]
        )
        self.transform = transforms.Compose(
            [*aug, transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        r, pid = self.items[idx]
        frame = random.choice(r["frames"]) if self.train else r["clip"]
        img = self.reader.crop(r["video"], r["clip"], frame, pid)
        return self.transform(img), ACTION_TO_IDX[r["track_actions"][pid]]

    @property
    def targets(self):
        return [ACTION_TO_IDX[r["track_actions"][pid]] for r, pid in self.items]


class FeatureDataset(Dataset):
    """Cached player features (gar.features.extractor) -> group activity label.

    `frames` indexes the 9 tracked frames (target = index 5). Returns
    x: (T, 12, 2048) float32 — or (12, 2048) for a single frame —, mask: (12,) bool, label.
    """

    def __init__(self, records, features_dir, frames=range(9)):
        self.features_dir = Path(features_dir)
        self.frames = list(frames)
        index = json.loads((self.features_dir / "index.json").read_text())
        order = {tuple(v): i for i, v in enumerate(index)}
        missing = [r for r in records if (r["video"], r["clip"]) not in order]
        if missing:
            raise ValueError(f"{features_dir} is missing {len(missing)} clips")
        self.rows = [order[(r["video"], r["clip"])] for r in records]
        self.labels = [r["label"] for r in records]
        self.mask = torch.from_numpy(np.load(self.features_dir / "mask.npy"))
        self._feats = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_feats"] = None
        return state

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        if self._feats is None:
            self._feats = np.load(self.features_dir / "feats.npy", mmap_mode="r")
        row = self.rows[idx]
        x = torch.from_numpy(np.asarray(self._feats[row, self.frames], dtype=np.float32))
        return ((x[0] if len(self.frames) == 1 else x), self.mask[row]), self.labels[idx]

    @property
    def targets(self):
        return self.labels


class PlayerSequenceDataset(Dataset):
    """One sample per tracked player: its cached feature sequence (T, 2048) -> person action.

    Trains the person-level LSTM (stage 1 of the hierarchical model) on top of the B3-A features.
    """

    def __init__(self, records, features_dir):
        self.features_dir = Path(features_dir)
        index = json.loads((self.features_dir / "index.json").read_text())
        order = {tuple(v): i for i, v in enumerate(index)}
        mask = np.load(self.features_dir / "mask.npy")
        actions = np.load(self.features_dir / "actions.npy")
        rows = [order[(r["video"], r["clip"])] for r in records]
        self.items = [(row, k) for row in rows for k in np.flatnonzero(mask[row])]
        self.labels = [int(actions[row, k]) for row, k in self.items]
        self._feats = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_feats"] = None
        return state

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self._feats is None:
            self._feats = np.load(self.features_dir / "feats.npy", mmap_mode="r")
        row, k = self.items[idx]
        return torch.from_numpy(np.asarray(self._feats[row, :, k], dtype=np.float32)), self.labels[idx]

    @property
    def targets(self):
        return self.labels


class ImageFeatureDataset(Dataset):
    """Cached whole-frame features (extractor --level image): (T, 2048) -> group activity (B4)."""

    def __init__(self, records, features_dir):
        self.features_dir = Path(features_dir)
        index = json.loads((self.features_dir / "index.json").read_text())
        order = {tuple(v): i for i, v in enumerate(index)}
        self.rows = [order[(r["video"], r["clip"])] for r in records]
        self.labels = [r["label"] for r in records]
        self._feats = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_feats"] = None
        return state

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        if self._feats is None:
            self._feats = np.load(self.features_dir / "feats.npy", mmap_mode="r")
        return torch.from_numpy(np.asarray(self._feats[self.rows[idx]], dtype=np.float32)), self.labels[idx]

    @property
    def targets(self):
        return self.labels


class TokenDataset(FeatureDataset):
    """FeatureDataset + normalized player boxes, for the actor transformer.

    Returns ((x (T, 12, 2048), mask (12,), boxes (T, 12, 4)), label); boxes are (cx, cy, w, h) / frame size.
    With `hflip` (train only) the scene is mirrored: player order reversed, cx -> 1 - cx, l_* <-> r_*.
    The crop features themselves are not mirrored, so this is an approximate augmentation.
    """

    def __init__(self, records, features_dir, frames=range(9), hflip=False):
        super().__init__(records, features_dir, frames)
        self.hflip = hflip
        self.boxes = torch.zeros(len(records), len(self.frames), 12, 4)
        for i, r in enumerate(records):
            for k, track in enumerate(r["tracks"].values()):
                for j, t in enumerate(self.frames):
                    x1, y1, x2, y2 = track[t]
                    self.boxes[i, j, k] = torch.tensor(
                        [
                            (x1 + x2) / 2 / r["width"],
                            (y1 + y2) / 2 / r["height"],
                            (x2 - x1) / r["width"],
                            (y2 - y1) / r["height"],
                        ]
                    )

    def __getitem__(self, idx):
        (x, mask), label = super().__getitem__(idx)
        boxes = self.boxes[idx].clone()
        if getattr(self, "force_flip", False) or (self.hflip and random.random() < 0.5):
            valid = torch.nonzero(mask).squeeze(1)
            order = torch.arange(mask.numel())
            order[valid] = valid.flip(0)  # reverse the players, keep padding at the end
            x, boxes = x[:, order], boxes[:, order]
            boxes[..., 0] = torch.where(mask, 1 - boxes[..., 0], boxes[..., 0])
            label = ACTIVITY_TO_IDX[FLIP_ACTIVITY[GROUP_ACTIVITIES[label]]]
        return (x, mask, boxes), label


def _feature_dim(self):
    """Last axis of the cached feature array (2048 for ResNet50, 768 for ConvNeXt-T, ...)."""
    return int(np.load(Path(self.features_dir) / "feats.npy", mmap_mode="r").shape[-1])


for _cls in (FeatureDataset, PlayerSequenceDataset, ImageFeatureDataset):
    _cls.dim = property(_feature_dim)


class ClipCropDataset(Dataset):
    """Raw player crops of a few frames per clip, for end-to-end training (gar.models.e2e).

    Returns ((crops (T, 12, 3, H, W), mask (12,), boxes (T, 12, 4), actions (12,)), label), padded slots
    are zeros (action -1). `frames` index the 9 tracked frames (target = 5). With `hflip` (train only)
    the whole scene is mirrored exactly: every crop flipped, players reversed, cx -> 1 - cx, l <-> r.
    """

    def __init__(self, records, crops_dir, frames, size=(224, 112), train=False, hflip=False):
        self.records, self.frames, self.size = records, list(frames), tuple(size)
        self.reader = CropReader(crops_dir)
        self.hflip = train and hflip
        jitter = [transforms.ColorJitter(0.2, 0.2, 0.2)] if train else []
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.size),
                *jitter,
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        t, k = len(self.frames), 12
        crops = torch.zeros(t, k, 3, *self.size)
        boxes = torch.zeros(t, k, 4)
        mask = torch.zeros(k, dtype=torch.bool)
        actions = torch.full((k,), -1, dtype=torch.long)
        flip = getattr(self, "force_flip", False) or (self.hflip and random.random() < 0.5)
        pids = list(r["tracks"])[::-1] if flip else list(r["tracks"])
        for slot, pid in enumerate(pids):
            mask[slot] = True
            actions[slot] = ACTION_TO_IDX.get(r["track_actions"][pid], -1)
            for j, fi in enumerate(self.frames):
                img = self.reader.crop(r["video"], r["clip"], r["frames"][fi], pid)
                if flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                crops[j, slot] = self.transform(img)
                x1, y1, x2, y2 = r["tracks"][pid][fi]
                cx = (x1 + x2) / 2 / r["width"]
                boxes[j, slot] = torch.tensor(
                    [
                        1 - cx if flip else cx,
                        (y1 + y2) / 2 / r["height"],
                        (x2 - x1) / r["width"],
                        (y2 - y1) / r["height"],
                    ]
                )
        label = r["label"]
        if flip:
            label = ACTIVITY_TO_IDX[FLIP_ACTIVITY[GROUP_ACTIVITIES[label]]]
        return (crops, mask, boxes, actions), label

    @property
    def targets(self):
        return [r["label"] for r in self.records]
