"""Generic training loop driven by a yaml config.

Each baseline module in gar.models exposes build_model(cfg), build_datasets(cfg, records) and
CLASS_NAMES. Model selection uses the val `select_metric` (accuracy by default); test is evaluated
once, with the best checkpoint.

Usage:
    python -m gar.train --config configs/b1.yaml
    python -m gar.train --config configs/b1.yaml --resume          # continue results/b1/last.pt
    python -m gar.train --config configs/b1.yaml --smoke           # 3 batches, checks the pipeline
"""

import argparse
import importlib
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from gar.data.annotations import load, select
from gar.evaluate import evaluate_split, predict, to_device
from gar.utils.io import load_config
from gar.utils.metrics import compute_metrics
from gar.utils.seed import seed_everything


def fit(cfg, out, resume=False, smoke=False, test=True, records=None, on_epoch=None):
    """Train `cfg` into folder `out`; returns the best val score (the config's select_metric).

    test=False skips the test split entirely (hyper-parameter search must never look at it);
    on_epoch(epoch, val_score) is called after every epoch and may raise to stop (Optuna pruning).
    """
    tc = cfg["train"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(out / "log.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    seed_everything(tc["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = importlib.import_module(f"gar.models.{cfg['model']}")
    records = records if records is not None else load(cfg["paths"]["annotations"])
    datasets = module.build_datasets(cfg, select(records, cfg))
    if not test:
        datasets.pop("test")
    if getattr(datasets["train"], "dim", None):  # feature models follow the backbone that produced the cache
        cfg["model_args"]["dim"] = datasets["train"].dim
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    # Windows spawns each worker as a full process that loads torch + CUDA DLLs (~1 GB committed
    # each): keep the worker count low, and only the train/val loaders' workers alive across epochs.
    workers = {"train": tc["num_workers"], "val": min(2, tc["num_workers"]), "test": min(2, tc["num_workers"])}
    loaders = {
        s: DataLoader(
            d,
            batch_size=tc["batch_size"],
            shuffle=(s == "train"),
            drop_last=(s == "train"),
            num_workers=workers[s],
            pin_memory=True,
            persistent_workers=workers[s] > 0 and s != "test",
        )
        for s, d in datasets.items()
    }
    log(f"[{out.name}] " + ", ".join(f"{s}={len(d)}" for s, d in datasets.items()) + f" | device={device}")

    model = module.build_model(cfg).to(device)
    # models may give their backbone a smaller learning rate (gar.models.e2e)
    params = model.param_groups(tc["lr"]) if hasattr(model, "param_groups") else model.parameters()
    if tc.get("optimizer", "adamw") == "sgd":  # the paper: SGD, momentum 0.9, fixed learning rate
        optimizer = torch.optim.SGD(params, lr=tc["lr"], momentum=0.9, weight_decay=tc["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    if tc.get("scheduler", "cosine") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    weight = None
    if tc.get("class_weights") == "sqrt_inv":
        counts = torch.bincount(torch.tensor(datasets["train"].targets), minlength=len(module.CLASS_NAMES))
        weight = 1.0 / counts.clamp(min=1).float().sqrt()
        weight = (weight / weight.mean()).to(device)
        log("class weights: " + ", ".join(f"{n}={w:.2f}" for n, w in zip(module.CLASS_NAMES, weight.tolist())))
    criterion = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=tc.get("label_smoothing", 0.0))
    metric = tc.get("select_metric", "accuracy")
    scaler = torch.amp.GradScaler(enabled=tc["amp"])

    start_epoch, best_score, bad_epochs = 0, -1.0, 0
    if resume and (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best = ckpt["best_score"] if "best_score" in ckpt else ckpt["best_acc"]  # older checkpoints
        start_epoch, best_score, bad_epochs = ckpt["epoch"] + 1, best, ckpt["bad_epochs"]
        log(f"resumed at epoch {start_epoch}, best val {metric} {best_score:.4f}")

    max_batches = 3 if smoke else None
    epochs = 1 if smoke else tc["epochs"]
    for epoch in range(start_epoch, epochs):
        model.train()
        t0, total_loss, seen = time.time(), 0.0, 0
        for i, (x, y) in enumerate(loaders["train"]):
            if max_batches and i >= max_batches:
                break
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=tc["amp"]):
                output = model(*to_device(x, device))
                # a model can return (logits, auxiliary loss), e.g. the person-action loss of gar.models.e2e
                loss = criterion(output[0], y) + output[1] if isinstance(output, tuple) else criterion(output, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y)
            seen += len(y)
        scheduler.step()

        y_true, y_pred = predict(model, loaders["val"], device, tc["amp"], max_batches)
        val = compute_metrics(y_true, y_pred, module.CLASS_NAMES)
        improved = val[metric] > best_score
        if improved:
            best_score, bad_epochs = val[metric], 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, out / "best.pt")
        else:
            bad_epochs += 1
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "bad_epochs": bad_epochs,
            },
            out / "last.pt",
        )
        log(
            f"epoch {epoch + 1:2d}/{epochs} | loss {total_loss / max(seen, 1):.4f} | "
            f"val acc {val['accuracy']:.4f} f1 {val['macro_f1']:.4f}{' *' if improved else ''} | "
            f"{time.time() - t0:.0f}s"
        )
        if on_epoch:
            on_epoch(epoch, val[metric])
        if bad_epochs >= tc["early_stopping_patience"]:
            log(f"early stop: no val improvement for {bad_epochs} epochs")
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    for split in ("val", "test") if test else ("val",):
        m = evaluate_split(model, loaders[split], device, module.CLASS_NAMES, out, split, tc["amp"], max_batches)
        log(f"best checkpoint | {split}: accuracy {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}")
    return best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", help="results/<name>, defaults to the model name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run a few batches only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = args.name or cfg["model"] + ("_smoke" if args.smoke else "")
    fit(cfg, Path(cfg["paths"]["results"]) / name, resume=args.resume, smoke=args.smoke)


if __name__ == "__main__":
    main()
