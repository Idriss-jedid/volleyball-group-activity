"""Fine-tune a YOLO detector on volleyball players (one class: "player").

COCO detectors find every person: players, referees, coaches, the crowd. Trained on the tracked
player boxes of the train videos, the detector learns to return only the 12 players, and gets better
at the small, far-court ones. Validation uses val videos; test videos are never seen.

Dataset (YOLO format, written under --out): 3 frames per train clip (target-3, target, target+3) and
the target frame of --val-clips val clips. Images are symlinked from the dataset folder when possible.

Usage:
    python -m gar.detect.finetune --videos /kaggle/input/.../ --weights yolo11m.pt --out data/processed/yolo_ft
Writes <out>/runs/train/weights/best.pt, copied to results/<name>.pt.
"""

import argparse
import io
import random
import shutil
from pathlib import Path

from PIL import Image

from gar.data.annotations import load
from gar.data.frames import FrameSource


def write_image(source, videos_root, video, clip, frame, dst):
    """Symlink the original jpg if the dataset is a folder, else extract it from the zip."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    src = Path(videos_root) / "videos" / str(video) / str(clip) / f"{frame}.jpg"
    if not str(videos_root).endswith(".zip") and src.exists():
        dst.symlink_to(src)
    else:
        Image.open(io.BytesIO(source.frame(video, clip, frame))).save(dst, quality=95)


def write_labels(r, t, dst):
    """One line per tracked player: class cx cy w h, normalized."""
    lines = []
    for boxes in r["tracks"].values():
        x1, y1, x2, y2 = boxes[t]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(r["width"], x2), min(r["height"], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(
            f"0 {(x1 + x2) / 2 / r['width']:.6f} {(y1 + y2) / 2 / r['height']:.6f} "
            f"{(x2 - x1) / r['width']:.6f} {(y2 - y1) / r['height']:.6f}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")


def build_dataset(records, videos, out, frames_idx, val_clips):
    source = FrameSource(videos)
    rng = random.Random(0)
    val = rng.sample([r for r in records if r["split"] == "val"], val_clips)
    jobs = [(r, t, "train") for r in records if r["split"] == "train" for t in frames_idx]
    jobs += [(r, r["frames"].index(r["clip"]), "val") for r in val]
    for i, (r, t, split) in enumerate(jobs):
        stem = f"{r['video']}_{r['clip']}_{r['frames'][t]}"
        write_image(source, videos, r["video"], r["clip"], r["frames"][t], out / "images" / split / f"{stem}.jpg")
        write_labels(r, t, out / "labels" / split / f"{stem}.txt")
        if (i + 1) % 1000 == 0:
            print(f"dataset: {i + 1}/{len(jobs)} images", flush=True)
    data = out / "data.yaml"
    data.write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: player\n")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--videos", default="D:/volleyball.zip")
    parser.add_argument("--annotations", default="data/processed/annotations.pkl")
    parser.add_argument("--out", default="data/processed/yolo_ft")
    parser.add_argument("--weights", default="yolo11m.pt")
    parser.add_argument("--name", default="yolo11m_volleyball", help="results/<name>.pt")
    parser.add_argument("--results", default="results")
    parser.add_argument("--frames", type=int, nargs="+", default=[2, 5, 8], help="indices into the 9 tracked frames")
    parser.add_argument("--val-clips", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--only-videos", type=int, nargs="*", help="debugging subset")
    args = parser.parse_args()

    from ultralytics import YOLO  # optional dependency

    records = load(args.annotations)
    if args.only_videos:
        records = [r for r in records if r["video"] in set(args.only_videos)]
    out = Path(args.out)
    val_clips = min(args.val_clips, sum(r["split"] == "val" for r in records))
    data = build_dataset(records, args.videos, out, args.frames, val_clips)

    model = YOLO(args.weights)
    model.train(
        data=str(data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=10,
        project=str((out / "runs").resolve()),
        name="train",
        exist_ok=True,
        workers=4,
        close_mosaic=5,
        fliplr=0.5,
        single_cls=True,
        plots=False,
        verbose=False,
    )
    best = out / "runs" / "train" / "weights" / "best.pt"
    Path(args.results).mkdir(parents=True, exist_ok=True)
    shutil.copy(best, Path(args.results) / f"{args.name}.pt")
    metrics = model.val(data=str(data), imgsz=args.imgsz, batch=args.batch, plots=False, verbose=False)
    print(f"val mAP50 {metrics.box.map50:.4f} | mAP50-95 {metrics.box.map:.4f} -> {args.results}/{args.name}.pt")


if __name__ == "__main__":
    main()
