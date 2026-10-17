"""Replace the ground-truth player tracks with detected + tracked ones, for any detector.

1. Detect persons on the 9 frames of each clip (see gar.detect.detectors), cache the raw boxes.
2. Track them with the IoU tracker (gar.detect.tracker), the same for every detector.
3. Detectors also find referees, coaches and the crowd: a small classifier on track geometry,
   trained on train clips (a track overlapping a ground-truth player at IoU >= 0.5 is a positive),
   scores how player-like each track is.
4. The (up to) 12 most player-like tracks above a threshold, tuned on val, become the clip's tracks.

Output in <out-dir>/<detector>/: raw.pkl (cached detections), annotations.pkl (gar.data.annotations
format, val + test), quality.json (recall / precision vs the ground truth on the target frame, fps).

Usage:
    python -m gar.detect.pipeline --detector yolo11m.pt
    python -m gar.detect.pipeline --detector frcnn_r50 --videos /kaggle/input/.../ --out-dir data/processed/det
"""

import argparse
import io
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

from gar.data.annotations import load
from gar.data.frames import FrameSource
from gar.data.splits import MAX_PLAYERS
from gar.detect.detectors import build_detector, detector_name
from gar.detect.tracker import iou_matrix, track


def nearest(tr, t):
    """Box of track `tr` at frame index t, or at the closest frame where it was seen."""
    return tr[min(tr, key=lambda f: abs(f - t))][0]


def track_features(tracks, r):
    target = r["frames"].index(r["clip"])
    tids = list(tracks)
    boxes = [nearest(tracks[tid], target) for tid in tids]
    median_h = float(np.median([b[3] - b[1] for b in boxes])) if boxes else 1.0
    feats = []
    for tid, (x1, y1, x2, y2) in zip(tids, boxes):
        w, h = (x2 - x1) / r["width"], (y2 - y1) / r["height"]
        scores = [s for _, s in tracks[tid].values()]
        feats.append(
            [
                (x1 + x2) / 2 / r["width"],
                (y1 + y2) / 2 / r["height"],
                w,
                h,
                h / w if w else 0,
                (y2 - y1) / max(median_h, 1),
                float(np.mean(scores)),
                float(np.max(scores)),
                len(tracks[tid]) / len(r["frames"]),
                float(target in tracks[tid]),
            ]
        )
    return tids, np.array(feats, dtype=np.float32).reshape(len(tids), 10)


def gt_target_boxes(r):
    t = r["frames"].index(r["clip"])
    return [boxes[t] for boxes in r["tracks"].values()]


def match(pred, gt, thr=0.5):
    """Greedy one-to-one IoU matching -> (number of matches, their IoUs)."""
    m = iou_matrix(pred, gt)
    pairs = sorted(((m[i, j], i, j) for i in range(len(pred)) for j in range(len(gt)) if m[i, j] >= thr), reverse=True)
    used_p, used_g, ious = set(), set(), []
    for v, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            ious.append(v)
    return len(ious), ious


def select_tracks(r, tracks, selector, threshold):
    """Keep the (up to) 12 most player-like tracks, ordered left to right on the target frame."""
    target = r["frames"].index(r["clip"])
    new = dict(r, detected=True, tracks={}, track_actions={})
    if not tracks:
        return new
    tids, feats = track_features(tracks, r)
    prob = selector.predict_proba(feats)[:, 1]
    keep = [tids[i] for i in np.argsort(-prob)[:MAX_PLAYERS] if prob[i] >= threshold]
    keep.sort(key=lambda tid: sum(nearest(tracks[tid], target)[::2]) / 2)
    for tid in keep:
        new["tracks"][tid] = [tuple(round(v) for v in nearest(tracks[tid], t)) for t in range(len(r["frames"]))]
        new["track_actions"][tid] = "standing"  # unknown for detections; unused by the group models
    return new


def select_all(items, selector, threshold):
    recs, gt_n, raw_m, sel_n, ious = [], 0, 0, 0, []
    for r, tracks in items:
        target = r["frames"].index(r["clip"])
        gt = gt_target_boxes(r)
        new = select_tracks(r, tracks, selector, threshold)
        m, mi = match([b[target] for b in new["tracks"].values()], gt)
        raw_m += match([nearest(tr, target) for tr in tracks.values()], gt)[0]
        gt_n, sel_n = gt_n + len(gt), sel_n + len(new["tracks"])
        ious += mi
        recs.append(new)
    recall, precision = len(ious) / max(gt_n, 1), len(ious) / max(sel_n, 1)
    return recs, {
        "clips": len(recs),
        "raw_recall": raw_m / max(gt_n, 1),  # any track at all, before selection
        "recall": recall,
        "precision": precision,
        "f1": 2 * recall * precision / max(recall + precision, 1e-9),
        "players_per_clip": sel_n / max(len(recs), 1),
        "mean_iou_of_matches": float(np.mean(ious)) if ious else 0.0,
        "clips_without_players": sum(not r["tracks"] for r in recs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--detector", required=True, help="yolo11m.pt, rtdetr-l.pt, frcnn_r50, or a .pt path")
    parser.add_argument("--name", help="output folder name, default derived from --detector")
    parser.add_argument("--videos", default="D:/volleyball.zip")
    parser.add_argument("--annotations", default="data/processed/annotations.pkl")
    parser.add_argument("--out-dir", default="data/processed/det")
    parser.add_argument("--conf", type=float, default=0.1, help="low on purpose: the tracker uses low-score boxes")
    parser.add_argument("--imgsz", type=int, default=1280, help="ultralytics input size")
    parser.add_argument("--train-clips", type=int, default=500, help="train clips used to fit the player selector")
    parser.add_argument("--val-clips", type=int, default=400, help="val clips used to tune the threshold")
    parser.add_argument("--test-clips", type=int, help="limit, for a quick test")
    args = parser.parse_args()

    name = args.name or detector_name(Path(args.detector).name)
    out = Path(args.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    records = load(args.annotations)
    rng = random.Random(0)
    by_split = {s: [r for r in records if r["split"] == s] for s in ("train", "val", "test")}
    chosen = {
        "train": rng.sample(by_split["train"], min(args.train_clips, len(by_split["train"]))),
        "val": rng.sample(by_split["val"], min(args.val_clips, len(by_split["val"]))),
        "test": by_split["test"][: args.test_clips] if args.test_clips else by_split["test"],
    }

    # 1. detection, cached: re-running the selection never needs the GPU again
    raw_path = out / "raw.pkl"
    raw = pickle.loads(raw_path.read_bytes()) if raw_path.exists() else {}
    todo = [r for rs in chosen.values() for r in rs if (r["video"], r["clip"]) not in raw]
    detector = build_detector(args.detector, args.conf, args.imgsz) if todo else None
    source, t_det, n_frames = FrameSource(args.videos), 0.0, 0
    for i, r in enumerate(todo):
        images = [Image.open(io.BytesIO(source.frame(r["video"], r["clip"], f))).convert("RGB") for f in r["frames"]]
        t0 = time.time()
        raw[(r["video"], r["clip"])] = detector(images)
        t_det, n_frames = t_det + time.time() - t0, n_frames + len(images)
        if (i + 1) % 200 == 0:
            print(f"[{name}] {i + 1}/{len(todo)} clips | {n_frames / t_det:.1f} fps", flush=True)
            raw_path.write_bytes(pickle.dumps(raw))
    raw_path.write_bytes(pickle.dumps(raw))

    # 2. tracking + player selector fitted on train clips
    tracked = {s: [(r, track(raw[(r["video"], r["clip"])])) for r in rs] for s, rs in chosen.items()}
    X, y = [], []
    for r, tracks in tracked["train"]:
        tids, feats = track_features(tracks, r)
        target, gt = r["frames"].index(r["clip"]), gt_target_boxes(r)
        ious = iou_matrix([nearest(tracks[t], target) for t in tids], gt)
        X += list(feats)
        y += list(ious.max(axis=1) >= 0.5) if ious.size else [False] * len(tids)
    selector = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(np.array(X), np.array(y))

    # 3. threshold tuned on val, then val + test written and measured
    threshold = max(np.arange(0.05, 0.95, 0.05), key=lambda t: select_all(tracked["val"], selector, t)[1]["f1"])
    quality = {
        "detector": args.detector,
        "threshold": round(float(threshold), 2),
        "fps": round(n_frames / t_det, 1) if t_det else None,
        "selector_train_tracks": len(y),
        "selector_positive_rate": float(np.mean(y)),
    }
    det_records = []
    for split in ("val", "test"):
        recs, quality[split] = select_all(tracked[split], selector, threshold)
        det_records += recs
    (out / "annotations.pkl").write_bytes(pickle.dumps(det_records))
    (out / "quality.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
