"""Extract 2048-d ResNet50 features for every tracked player crop, once, into fp16 memmaps.

Output folder (default data/features/<name>/):
    feats.npy    float16 [N_clips, 9, 12, 2048]  player slots ordered left to right on the target frame
    mask.npy     bool    [N_clips, 12]           False for padded slots (clips with < 12 tracks)
    actions.npy  int64   [N_clips, 12]           person action index, -1 for padding
    index.json   [[video, clip], ...]            row order == annotations.pkl order

Every person-level baseline (B3, B5-B8) trains on these arrays instead of images.

With `--level image` it instead stores whole-frame features of the B1 model for B4:
    feats.npy    float16 [N_clips, 9, 2048]       + index.json

Usage:
    python -m gar.features.extractor --checkpoint results/person/best.pt --name person_ft
    python -m gar.features.extractor --name imagenet              # frozen ImageNet backbone (paper B2)
    python -m gar.features.extractor --level image --config configs/b1.yaml \
        --checkpoint results/b1/best.pt --name b1_image
"""

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from gar.data.annotations import load, select
from gar.data.datasets import IMAGENET_MEAN, IMAGENET_STD, CropReader, FrameDataset
from gar.data.splits import ACTION_TO_IDX, MAX_PLAYERS
from gar.models.backbones import strip_head
from gar.utils.io import load_config


class ClipCrops(Dataset):
    """All crops of one clip: (9 * n_players, 3, H, W), frame-major."""

    def __init__(self, records, crops_dir, size):
        self.records = records
        self.reader = CropReader(crops_dir)
        self.transform = transforms.Compose(
            [transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        crops = [
            self.transform(self.reader.crop(r["video"], r["clip"], f, pid)) for f in r["frames"] for pid in r["tracks"]
        ]
        return idx, torch.stack(crops) if crops else torch.empty(0)  # detections can find nobody


def load_backbone(cfg, checkpoint, device):
    module = importlib.import_module(f"gar.models.{cfg['model']}")
    model = module.build_model(cfg)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    dim = strip_head(model, cfg["model_args"].get("backbone", "resnet50"))  # keep the pooled features
    return model.to(device).eval(), dim


def extract_image_level(cfg, records, model, dim, device, out, workers):
    """Whole frames of the 9 tracked frames -> [N, 9, 2048] (B4)."""
    offsets = [f - records[0]["clip"] for f in records[0]["frames"]]  # -5 .. +3
    dataset = FrameDataset(records, cfg["paths"]["videos"], offsets=offsets, size=cfg["data"]["frame_size"])
    feats = np.lib.format.open_memmap(out / "feats.npy", "w+", np.float16, (len(records), len(offsets), dim))
    loader = DataLoader(dataset, batch_size=4, num_workers=workers, pin_memory=True)
    t0, row = time.time(), 0
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            b = x.shape[0]
            f = model(x.flatten(0, 1).to(device, non_blocking=True)).view(b, len(offsets), dim)
            feats[row : row + b] = f.cpu().numpy().astype(np.float16)
            row += b
            if (i + 1) % 100 == 0:
                rate = row / (time.time() - t0)
                eta = (len(records) - row) / rate / 60
                print(f"{row}/{len(records)} clips | {rate:.1f} clips/s | eta {eta:.1f} min", flush=True)
    feats.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/person.yaml")
    parser.add_argument("--checkpoint", help="fine-tuned model; omit for frozen ImageNet weights")
    parser.add_argument("--name", required=True)
    parser.add_argument("--level", choices=["person", "image"], default="person")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--annotations", help="override paths.annotations (e.g. detected tracks)")
    parser.add_argument("--crops", help="override paths.crops")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key in ("annotations", "crops"):
        if getattr(args, key):
            cfg["paths"][key] = getattr(args, key)
    records = select(load(cfg["paths"]["annotations"]), cfg)
    out = Path(cfg["paths"]["features"]) / args.name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dim = load_backbone(cfg, args.checkpoint, device)
    (out / "index.json").write_text(json.dumps([[r["video"], r["clip"]] for r in records]))

    if args.level == "image":
        t0 = time.time()
        extract_image_level(cfg, records, model, dim, device, out, args.workers)
        print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")
        return

    n, t = len(records), len(records[0]["frames"])
    feats = np.lib.format.open_memmap(out / "feats.npy", "w+", np.float16, (n, t, MAX_PLAYERS, dim))
    mask = np.zeros((n, MAX_PLAYERS), bool)
    actions = np.full((n, MAX_PLAYERS), -1, np.int64)

    loader = DataLoader(
        ClipCrops(records, cfg["paths"]["crops"], cfg["data"]["crop_size"]),
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
    )
    t0 = time.time()
    with torch.no_grad():
        for i, (idx, crops) in enumerate(loader):
            r = records[idx]
            k = len(r["tracks"])
            if k == 0:
                continue
            f = model(crops.to(device, non_blocking=True)).view(t, k, dim)
            feats[idx, :, :k] = f.cpu().numpy().astype(np.float16)
            mask[idx, :k] = True
            actions[idx, :k] = [ACTION_TO_IDX[a] for a in r["track_actions"].values()]
            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"{i + 1}/{n} clips | {rate:.1f} clips/s | eta {(n - i - 1) / rate / 60:.1f} min", flush=True)

    feats.flush()
    np.save(out / "mask.npy", mask)
    np.save(out / "actions.npy", actions)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
