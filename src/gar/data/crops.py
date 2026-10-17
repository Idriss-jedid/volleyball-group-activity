"""Cut every tracked player out of the 9 frames once, and store the crops as small JPEGs.

Decoding a full 1280x720 frame costs ~30 ms; a 256x128 crop ~1 ms. Every person-level baseline
(B3, B5-B8) reads crops many times, so we pay the full-frame decoding exactly once here.

Output: one zip per video, data/processed/crops/<video>.zip, entries `<clip>/<frame>_<player_id>.jpg`.
55 zips, ~3 GB in total: small enough to upload to Kaggle/Colab as one dataset.

Each box is first widened or heightened to the crop's aspect ratio around its centre, so a falling
or digging player keeps their shape (with a bit more context) instead of being squashed.

Usage:
    python -m gar.data.crops --workers 8
"""

import argparse
import io
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from gar.data.annotations import load
from gar.data.frames import FrameSource


def expand_to_aspect(box, aspect, width, height):
    """Grow (x1, y1, x2, y2) around its centre until h / w == aspect, then clip to the frame."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if h / max(w, 1) > aspect:
        w = h / aspect
    else:
        h = w * aspect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (
        max(0, round(cx - w / 2)),
        max(0, round(cy - h / 2)),
        min(width, round(cx + w / 2)),
        min(height, round(cy + h / 2)),
    )


def crop_video(job):
    video, records, source, out_dir, size, quality = job
    out = Path(out_dir) / f"{video}.zip"
    tmp = out.with_suffix(".tmp")
    h, w = size
    n = 0
    src = FrameSource(source)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as dst:
        for r in records:
            for t, frame in enumerate(r["frames"]):
                img = Image.open(io.BytesIO(src.frame(video, r["clip"], frame))).convert("RGB")
                for pid, boxes in r["tracks"].items():
                    box = expand_to_aspect(boxes[t], h / w, r["width"], r["height"])
                    buf = io.BytesIO()
                    img.crop(box).resize((w, h), Image.BILINEAR).save(buf, "JPEG", quality=quality)
                    dst.writestr(f"{r['clip']}/{frame}_{pid}.jpg", buf.getvalue())
                    n += 1
    tmp.replace(out)  # only complete zips get the final name, so an interrupted run can resume
    return video, n


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--videos", dest="source", default="D:/volleyball.zip", help="dataset zip, or a folder containing videos/"
    )
    parser.add_argument("--annotations", default="data/processed/annotations.pkl")
    parser.add_argument("--out", default="data/processed/crops")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 128], metavar=("H", "W"))
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", type=int, nargs="*", help="subset of video ids, for testing")
    args = parser.parse_args()

    records = load(args.annotations)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    videos = args.only if args.only else sorted({r["video"] for r in records})
    todo = [v for v in videos if not (Path(args.out) / f"{v}.zip").exists()]
    print(f"{len(videos) - len(todo)} videos already done, {len(todo)} to go")

    jobs = [
        (v, [r for r in records if r["video"] == v], args.source, args.out, tuple(args.size), args.quality)
        for v in todo
    ]
    t0, total = time.time(), 0
    with Pool(args.workers) as pool:
        for i, (video, n) in enumerate(pool.imap_unordered(crop_video, jobs), 1):
            total += n
            print(f"[{i}/{len(jobs)}] video {video}: {n} crops | {time.time() - t0:.0f}s", flush=True)
    print(f"done: {total} crops in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
