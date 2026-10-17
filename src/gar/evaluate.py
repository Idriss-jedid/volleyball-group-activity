"""Evaluate a checkpoint on val/test and dump metrics.

Usage:
    python -m gar.evaluate --run results/b1 --split test
    # same trained model, detected boxes instead of the ground-truth tracks:
    python -m gar.evaluate --run results/b8 --annotations data/processed/annotations_det.pkl
        --features person_ft_det --tag det
"""

import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gar.data.annotations import load, select
from gar.data.splits import ACTIVITY_TO_IDX, FLIP_ACTIVITY, GROUP_ACTIVITIES
from gar.utils.io import load_config, save_json
from gar.utils.metrics import compute_metrics, plot_confusion


def to_device(x, device):
    """Model inputs are a tensor, or a tuple of tensors such as (features, mask)."""
    if isinstance(x, (tuple, list)):
        return tuple(t.to(device, non_blocking=True) for t in x)
    return (x.to(device, non_blocking=True),)


@torch.no_grad()
def predict(model, loader, device, amp=False, max_batches=None, return_logits=False):
    model.eval()
    y_true, logits = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits.append(model(*to_device(x, device)).float().cpu())
        y_true += y.tolist()
    logits = torch.cat(logits) if logits else torch.zeros(0)
    y_pred = logits.argmax(1).tolist() if len(logits) else []
    return (y_true, y_pred, logits) if return_logits else (y_true, y_pred)


def predict_tta(model, loader, device, amp=False):
    """Average the softmax of each clip and of its mirror image (teams swapped, so l_* <-> r_*)."""
    y_true, _, logits = predict(model, loader, device, amp, return_logits=True)
    loader.dataset.force_flip = True
    try:
        _, _, flipped = predict(model, loader, device, amp, return_logits=True)
    finally:
        loader.dataset.force_flip = False
    perm = [ACTIVITY_TO_IDX[FLIP_ACTIVITY[a]] for a in GROUP_ACTIVITIES]
    probs = (logits.softmax(1) + flipped.softmax(1)[:, perm]) / 2
    return y_true, probs.log()  # log-probabilities: argmax and later softmax are unchanged


def evaluate_split(model, loader, device, class_names, out_dir, split, amp=False, max_batches=None, tta=False):
    if tta:
        y_true, logits = predict_tta(model, loader, device, amp)
        y_pred = logits.argmax(1).tolist()
    else:
        y_true, y_pred, logits = predict(model, loader, device, amp, max_batches, return_logits=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # logits + labels (in dataset order) let gar.ensemble combine models without re-running them
    np.savez(Path(out_dir) / f"logits_{split}.npz", logits=logits.numpy(), labels=np.array(y_true))
    metrics = compute_metrics(y_true, y_pred, class_names)
    save_json(metrics, Path(out_dir) / f"metrics_{split}.json")
    plot_confusion(
        metrics["confusion_matrix"],
        class_names,
        Path(out_dir) / f"confusion_{split}.png",
        title=f"{split}: acc {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f}",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="results/<run> folder written by gar.train")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--annotations", help="override paths.annotations")
    parser.add_argument("--features", help="override the cached feature set")
    parser.add_argument("--crops", help="override paths.crops (end-to-end models on detected boxes)")
    parser.add_argument("--tag", default="", help="suffix of the metric files, e.g. det")
    parser.add_argument("--tta", action="store_true", help="average with the mirrored clip (l <-> r swapped)")
    parser.add_argument("--out", help="write metrics/logits to this folder instead of the run folder")
    args = parser.parse_args()

    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    if args.annotations:
        cfg["paths"]["annotations"] = args.annotations
    if args.features:
        cfg["features"] = args.features
    if args.crops:
        cfg["paths"]["crops"] = args.crops
    module = importlib.import_module(f"gar.models.{cfg['model']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = module.build_model(cfg).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device)["model"])
    records = [r for r in select(load(cfg["paths"]["annotations"]), cfg) if r["split"] == args.split]
    dataset = module.build_datasets(cfg, records)[args.split]
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], num_workers=cfg["train"]["num_workers"])

    name = args.split + (f"_{args.tag}" if args.tag else "")
    m = evaluate_split(
        model, loader, device, module.CLASS_NAMES, args.out or run, name, cfg["train"]["amp"], tta=args.tta
    )
    print(f"{name}: accuracy {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
