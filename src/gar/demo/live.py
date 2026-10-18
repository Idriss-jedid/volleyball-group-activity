"""Live predictions over a whole rally: the model re-run at every frame, for the demo's moving bars.

Annotated clips of one match are close in time (~35 frames apart, 41 frames each), so a chain of them
covers a continuous rally. On every frame of that rally:

    YOLO (players) -> IoU tracker over the whole rally -> the 12 players around frame t
    -> crops at t-4, t-2, t, t+2 -> end-to-end model -> team-activity probabilities + each action at t

Usage:
    python -m gar.demo.live --detector results/yolo11s_volleyball.pt --model results/e2e_cnx \
        --rally 20:25465:25615 --out results/live_20_25465.json
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from gar.data.annotations import load
from gar.data.datasets import IMAGENET_MEAN, IMAGENET_STD
from gar.data.frames import FrameSource
from gar.data.splits import GROUP_ACTIVITIES, MAX_PLAYERS, PERSON_ACTIONS
from gar.demo.infer import MODEL_OFFSETS, crop_tensor
from gar.detect.detectors import build_detector
from gar.detect.tracker import track
from gar.models.e2e import build_model, predict_with_actions
from gar.utils.io import load_config


def nearest_box(tracks, tid, f):
    """Box of track tid at frame f, or at the closest frame where it was seen."""
    tr = tracks[tid]
    return tr[min(tr, key=lambda g: abs(g - f))][0]


def rally_frames(records, video, first, last):
    """Every frame from first-20 to last+20, each read from an annotated clip folder that holds it."""
    clips = sorted(r["clip"] for r in records if r["video"] == video and first <= r["clip"] <= last)
    frames = []
    for f in range(first - 20, last + 21):
        holder = min(clips, key=lambda c: abs(c - f))
        if abs(holder - f) > 20:
            raise ValueError(f"frame {f} is not covered by any clip of video {video}")
        frames.append((f, holder))
    return frames, clips


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--videos", default="D:/volleyball.zip")
    parser.add_argument("--annotations", default="data/processed/annotations.pkl")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--model", required=True, help="run folder with config.yaml + best.pt (e2e)")
    parser.add_argument("--rally", required=True, nargs="+", help="video:first_clip:last_clip")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(Path(args.model) / "config.yaml")
    cfg["model_args"]["init"] = None
    model = build_model(cfg)
    model.load_state_dict(torch.load(Path(args.model) / "best.pt", map_location="cpu")["model"])
    model = model.to(device).eval()
    size = tuple(cfg["data"]["crop_size"])
    transform = transforms.Compose(
        [transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    detector = build_detector(args.detector, conf=0.25, imgsz=1280)
    source = FrameSource(args.videos)
    records = load(args.annotations)

    for spec in args.rally:
        video, first, last = map(int, spec.split(":"))
        frames, clips = rally_frames(records, video, first, last)
        rec = next(r for r in records if r["video"] == video)
        width, height = rec["width"], rec["height"]
        images = [Image.open(io.BytesIO(source.frame(video, holder, f))).convert("RGB") for f, holder in frames]
        dets = []
        for i in range(0, len(images), 8):
            dets += detector(images[i : i + 8])
        tracks = track(dets)

        per_frame = []
        for t in range(len(frames)):
            window = [t + o for o in MODEL_OFFSETS]
            players_now = [(tid, tr[t][0]) for tid, tr in tracks.items() if t in tr]
            entry = {"players": [{"box": [round(v, 1) for v in b]} for _, b in players_now], "probs": None}
            if min(window) >= 0 and max(window) < len(frames):
                # the 12 tracks seen around t, longest first, ordered left to right at t
                near = {tid: tr for tid, tr in tracks.items() if any(abs(f - t) <= 4 for f in tr)}
                keep = sorted(near, key=lambda tid: -sum(1 for f in near[tid] if abs(f - t) <= 6))[:MAX_PLAYERS]

                keep.sort(key=lambda tid: sum(nearest_box(tracks, tid, t)[::2]) / 2)
                crops = torch.zeros(1, len(window), MAX_PLAYERS, 3, *size)
                boxes = torch.zeros(1, len(window), MAX_PLAYERS, 4)
                mask = torch.zeros(1, MAX_PLAYERS, dtype=torch.bool)
                for slot, tid in enumerate(keep):
                    mask[0, slot] = True
                    for j, fi in enumerate(window):
                        b = nearest_box(tracks, tid, fi)
                        crops[0, j, slot] = crop_tensor(images[fi], b, width, height, size, transform)
                        boxes[0, j, slot] = torch.tensor(
                            [
                                (b[0] + b[2]) / 2 / width,
                                (b[1] + b[3]) / 2 / height,
                                (b[2] - b[0]) / width,
                                (b[3] - b[1]) / height,
                            ]
                        )
                logits, person = predict_with_actions(model, crops.to(device), mask.to(device), boxes.to(device))
                entry["probs"] = [round(float(p), 4) for p in logits.softmax(-1)[0].cpu()]
                person = person[0].cpu()
                entry["players"] = [
                    {
                        "box": [round(v, 1) for v in nearest_box(tracks, tid, t)],
                        "team": "left" if slot < MAX_PLAYERS // 2 else "right",
                        "action": PERSON_ACTIONS[int(person[slot].argmax())],
                        "action_prob": round(float(person[slot].max()), 3),
                    }
                    for slot, tid in enumerate(keep)
                ]
            per_frame.append(entry)
            if (t + 1) % 50 == 0:
                print(f"v{video} rally: {t + 1}/{len(frames)} frames", flush=True)

        gt = [
            {
                "clip": c,
                "index": c - frames[0][0],
                "activity": next(r["activity"] for r in records if r["video"] == video and r["clip"] == c),
            }
            for c in clips
        ]
        out = {
            "video": video,
            "frames": [f for f, _ in frames],
            "holders": [h for _, h in frames],
            "width": width,
            "height": height,
            "activities": GROUP_ACTIVITIES,
            "ground_truth": gt,
            "per_frame": per_frame,
        }
        path = Path(args.out_dir) / f"live_{video}_{first}.json"
        path.write_text(json.dumps(out), encoding="utf-8")
        hits = [
            np.argmax(per_frame[g["index"]]["probs"]) == GROUP_ACTIVITIES.index(g["activity"])
            for g in gt
            if per_frame[g["index"]]["probs"]
        ]
        print(f"-> {path}: {len(frames)} frames, {sum(hits)}/{len(hits)} annotated moments correct", flush=True)


if __name__ == "__main__":
    main()
