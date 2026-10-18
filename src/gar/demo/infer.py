"""Run the full pipeline on raw clips and save what the demo videos draw.

    41 raw frames -> YOLO (players) -> IoU tracker -> 12 player tracks, ordered left to right
                  -> crops of 4 frames -> end-to-end model -> team activity + each player's action

No ground-truth box is used. The ground truth (activity, and the action of the nearest annotated
player) is stored next to the predictions, only for display.

Usage:
    python -m gar.demo.infer --detector results/yolo11m_volleyball.pt --model results_kaggle/.../e2e_cnx \
        --clips 50 --out demo/clips.json
"""

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from gar.data.annotations import load
from gar.data.crops import expand_to_aspect
from gar.data.datasets import IMAGENET_MEAN, IMAGENET_STD
from gar.data.frames import FrameSource
from gar.data.splits import GROUP_ACTIVITIES, MAX_PLAYERS, PERSON_ACTIONS
from gar.detect.detectors import build_detector
from gar.detect.tracker import iou_matrix, track
from gar.models.e2e import build_model, predict_with_actions
from gar.utils.io import load_config

WINDOW = range(-20, 21)  # every frame of the clip, for a smooth video
MODEL_OFFSETS = [-4, -2, 0, 2]  # the end-to-end model's 4 frames (configs/e2e.yaml: indices 1, 3, 5, 7)


def player_tracks(tracks, n_frames, target):
    """The (up to) 12 longest-lived tracks seen near the target frame, ordered left to right there."""
    near = {tid: tr for tid, tr in tracks.items() if any(abs(f - target) <= 4 for f in tr)}
    best = sorted(near, key=lambda tid: (-len(near[tid]), -np.mean([s for _, s in near[tid].values()])))
    keep = best[:MAX_PLAYERS]

    def box_at(tid, f):
        tr = tracks[tid]
        return tr[min(tr, key=lambda g: abs(g - f))][0]

    keep.sort(key=lambda tid: sum(box_at(tid, target)[::2]) / 2)
    return keep, box_at


def crop_tensor(img, box, width, height, size, transform):
    x1, y1, x2, y2 = expand_to_aspect(tuple(round(v) for v in box), size[0] / size[1], width, height)
    return transform(img.crop((x1, y1, x2, y2)).resize((128, 256), Image.BILINEAR))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--videos", default="D:/volleyball.zip")
    parser.add_argument("--annotations", default="data/processed/annotations.pkl")
    parser.add_argument("--detector", default="yolo11m.pt")
    parser.add_argument("--model", required=True, help="run folder with config.yaml + best.pt (e2e)")
    parser.add_argument("--clips", type=int, default=40, help="random test clips")
    parser.add_argument("--only", nargs="*", help="specific clips as video:clip")
    parser.add_argument("--out", default="demo/clips.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(Path(args.model) / "config.yaml")
    cfg["model_args"]["init"] = None  # the checkpoint holds every weight
    model = build_model(cfg)
    model.load_state_dict(torch.load(Path(args.model) / "best.pt", map_location="cpu")["model"])
    model = model.to(device).eval()
    size = tuple(cfg["data"]["crop_size"])
    transform = transforms.Compose(
        [transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    detector = build_detector(args.detector, conf=0.25, imgsz=1280)
    source = FrameSource(args.videos)

    test = [r for r in load(args.annotations) if r["split"] == "test"]
    if args.only:
        wanted = {tuple(map(int, c.split(":"))) for c in args.only}
        clips = [r for r in test if (r["video"], r["clip"]) in wanted]
    else:
        clips = random.Random(0).sample(test, args.clips)

    out = []
    for n, r in enumerate(clips):
        frames = [r["clip"] + o for o in WINDOW]
        images = [Image.open(io.BytesIO(source.frame(r["video"], r["clip"], f))).convert("RGB") for f in frames]
        dets = []
        for i in range(0, len(images), 8):
            dets += detector(images[i : i + 8])
        target = frames.index(r["clip"])
        tracks = track(dets)
        keep, box_at = player_tracks(tracks, len(frames), target)

        t = len(MODEL_OFFSETS)
        crops = torch.zeros(1, t, MAX_PLAYERS, 3, *size)
        boxes = torch.zeros(1, t, MAX_PLAYERS, 4)
        mask = torch.zeros(1, MAX_PLAYERS, dtype=torch.bool)
        for slot, tid in enumerate(keep):
            mask[0, slot] = True
            for j, off in enumerate(MODEL_OFFSETS):
                fi = target + off
                b = box_at(tid, fi)
                crops[0, j, slot] = crop_tensor(images[fi], b, r["width"], r["height"], size, transform)
                boxes[0, j, slot] = torch.tensor(
                    [
                        (b[0] + b[2]) / 2 / r["width"],
                        (b[1] + b[3]) / 2 / r["height"],
                        (b[2] - b[0]) / r["width"],
                        (b[3] - b[1]) / r["height"],
                    ]
                )
        logits, person = predict_with_actions(model, crops.to(device), mask.to(device), boxes.to(device))
        probs = logits.softmax(-1)[0].cpu().tolist()
        person = person[0].cpu()

        # ground truth, for display only: the annotated player overlapping each detected player
        gt_t = r["frames"].index(r["clip"])
        gt_boxes = [b[gt_t] for b in r["tracks"].values()]
        gt_actions = list(r["track_actions"].values())
        det_boxes = [box_at(tid, target) for tid in keep]
        ious = iou_matrix(det_boxes, gt_boxes) if keep and gt_boxes else np.zeros((len(keep), 0))

        players = []
        for slot, tid in enumerate(keep):
            p = person[slot]
            j = int(ious[slot].argmax()) if ious.shape[1] else -1
            players.append(
                {
                    "slot": slot,
                    "team": "left" if slot < MAX_PLAYERS // 2 else "right",  # the model's 2-group split
                    "action": PERSON_ACTIONS[int(p.argmax())],
                    "action_prob": float(p.max()),
                    "true_action": gt_actions[j] if j >= 0 and ious[slot, j] >= 0.3 else None,
                    "boxes": [
                        [round(v, 1) for v in box_at(tid, i)] if i in tracks[tid] else None for i in range(len(frames))
                    ],
                }
            )
        pred = GROUP_ACTIVITIES[int(np.argmax(probs))]
        out.append(
            {
                "video": r["video"],
                "clip": r["clip"],
                "frames": frames,
                "target_index": target,
                "width": r["width"],
                "height": r["height"],
                "activity": pred,
                "probs": dict(zip(GROUP_ACTIVITIES, probs)),
                "true_activity": r["activity"],
                "players": players,
            }
        )
        print(
            f"[{n + 1}/{len(clips)}] v{r['video']}/{r['clip']}: {pred} ({max(probs):.2f}) | true {r['activity']}"
            f" | {len(keep)} players",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    correct = sum(c["activity"] == c["true_activity"] for c in out)
    print(f"{correct}/{len(out)} correct -> {args.out}")


if __name__ == "__main__":
    main()
