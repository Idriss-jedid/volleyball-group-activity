"""Render a whole rally (gar.demo.live) as the 4-panel LinkedIn video, with bars that move live.

Real speed (25 fps, the dataset's own frames: no frame doubling, no lag). Panel 4 keeps the eight
activities in a fixed order, so the bars visibly rise and fall as the rally goes pass -> set -> spike.

Usage:
    python -m gar.demo.live_render --live results_kaggle/gar-live/results/live_20_25465.json \
        --out demo/out/linkedin_live.mp4
"""

import argparse
import io
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from gar.data.frames import FrameSource
from gar.demo.render import BG, LEFT, MUT, OK, RIGHT, TXT
from gar.demo.story import (
    ACTION_RGB,
    F_BAR,
    F_GTAG,
    F_OK,
    F_PANEL,
    F_VERDICT,
    PH,
    PLAIN,
    PW,
    WORDS,
    H,
    W,
    chip,
    verdict,
)
from gar.detect.tracker import iou_matrix

FPS = 25
ORDER = ["l_set", "l_spike", "l_pass", "l_winpoint", "r_set", "r_spike", "r_pass", "r_winpoint"]


def smoothed_probs(per_frame, activities, alpha=0.3):
    """Exponential moving average of the probabilities, so the bars glide instead of flickering."""
    out, cur = [], None
    for e in per_frame:
        if e["probs"] is not None:
            p = dict(zip(activities, e["probs"]))
            cur = p if cur is None else {a: cur[a] + alpha * (p[a] - cur[a]) for a in activities}
        out.append(cur)
    return out


def steady_leaders(probs, hold=10):
    """The activity named in the verdict: it changes only once a new leader stays on top for `hold` frames."""
    out, cur, cand, run = [], None, None, 0
    for p in probs:
        top = max(p, key=p.get) if p else None
        if top == cur:
            cand, run = None, 0
        elif top == cand:
            run += 1
        else:
            cand, run = top, 1
        if cur is None or run >= hold:
            cur, cand, run = top, None, 0
        out.append(cur)
    return out


def steady_actions(per_frame, t, window=4):
    """Majority action of each player over nearby frames (matched by IoU): no flickering tags."""
    players = per_frame[t]["players"]
    boxes = [p["box"] for p in players]
    votes = [Counter() for _ in players]
    for u in range(max(0, t - window), min(len(per_frame), t + window + 1)):
        other = [p for p in per_frame[u]["players"] if "action" in p]
        if not other or not boxes:
            continue
        m = iou_matrix(boxes, [p["box"] for p in other])
        for i in range(len(players)):
            j = int(m[i].argmax())
            if m[i, j] >= 0.5:
                votes[i][other[j]["action"]] += 1
    return [v.most_common(1)[0][0] if v else None for v in votes]


def panel(img, players, actions, mode, scale):
    p = img.copy().convert("RGBA")
    over = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    xs = []
    for pl, act in zip(players, actions):
        x1, y1, x2, y2 = (v * scale for v in pl["box"])
        team = pl.get("team")
        tc = LEFT if team == "left" else RIGHT if team == "right" else (255, 255, 255)
        if team:
            xs.append(((x1 + x2) / 2, team))
        if mode == "find":
            d.rounded_rectangle((x1, y1, x2, y2), radius=4, outline=(255, 255, 255, 240), width=3)
        elif mode == "actions":
            word = WORDS.get(act)
            c = ACTION_RGB.get(word, (148, 163, 184))
            d.rounded_rectangle((x1, y1, x2, y2), radius=4, outline=c + (255 if word else 140,), width=3 if word else 2)
            if word:
                tw = d.textlength(word, font=F_GTAG)
                d.rounded_rectangle((x1, y1 - 28, x1 + tw + 14, y1 - 4), radius=6, fill=c + (250,))
                d.text((x1 + 7, y1 - 27), word, font=F_GTAG, fill=(10, 14, 26))
        else:
            d.rounded_rectangle((x1, y1, x2, y2), radius=4, fill=tc + (80,), outline=tc + (255,), width=3)
    if mode == "teams":
        lx = [x for x, t in xs if t == "left"]
        rx = [x for x, t in xs if t == "right"]
        if lx and rx and max(lx) < min(rx):
            split = (max(lx) + min(rx)) / 2
            for y in range(70, PH - 70, 16):
                d.line((split, y, split, y + 9), fill=(255, 255, 255, 220), width=3)
            chip(d, split - 196, PH - 62, "left team", F_PANEL, dot=LEFT)
            chip(d, split + 16, PH - 62, "right team", F_PANEL, dot=RIGHT)
    return Image.alpha_composite(p, over).convert("RGB")


def bars(probs, leader, truth):
    p = Image.new("RGB", (PW, PH), (16, 23, 40))
    d = ImageDraw.Draw(p)
    x0, y0, bw = 250, 82, 560
    for k, a in enumerate(ORDER):
        y = y0 + k * 42
        v = probs[a] if probs else 0.0
        win = a == leader
        name = PLAIN[a]
        d.text((x0 - 18 - d.textlength(name, font=F_BAR), y), name, font=F_BAR, fill=TXT if win else MUT)
        d.rounded_rectangle((x0, y + 6, x0 + bw, y + 30), radius=12, fill=(34, 46, 72))
        c = OK if win else (LEFT if a[0] == "l" else RIGHT)
        if v > 0.01:
            d.rounded_rectangle((x0, y + 6, x0 + max(24, bw * v), y + 30), radius=12, fill=c)
        d.text((x0 + bw + 14, y + 2), f"{v * 100:.0f}%", font=F_BAR, fill=TXT if win else MUT)
    if leader:
        team, act, c = verdict(leader)
        msg = f"now: {team} · {act}"
        d.text((PW / 2 - d.textlength(msg, font=F_VERDICT) / 2, PH - 100), msg, font=F_VERDICT, fill=c)
    if truth:  # close to an annotated moment: show the annotators' label
        ok = truth == leader
        tag = f"annotated: {PLAIN[truth]}"
        tw = d.textlength(tag, font=F_OK)
        x = PW / 2 - tw / 2
        d.rounded_rectangle((x - 16, PH - 46, x + tw + 16, PH - 12), radius=17, fill=OK if ok else (100, 116, 139))
        d.text((x, PH - 44), tag, font=F_OK, fill=(10, 14, 26))
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--live", required=True)
    parser.add_argument("--videos", default="D:/volleyball.zip")
    parser.add_argument("--trim", type=int, nargs=2, metavar=("START", "END"), help="frame index range to keep")
    parser.add_argument("--hold", type=float, default=1.5, help="seconds the last frame is held")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    live = json.loads(Path(args.live).read_text(encoding="utf-8"))
    source = FrameSource(args.videos)
    per_frame, frames = live["per_frame"], live["frames"]
    probs = smoothed_probs(per_frame, live["activities"])
    leaders = steady_leaders(probs)
    truth_at = {}
    for g in live["ground_truth"]:  # annotated label shown around its moment
        for t in range(g["index"] - 4, g["index"] + 9):
            truth_at[t] = g["activity"]
    lo, hi = args.trim or (0, len(frames))
    tmp = Path(args.out).with_suffix("")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    scale = PW / live["width"]
    for n, t in enumerate(range(lo, hi)):
        img = Image.open(io.BytesIO(source.frame(live["video"], live["holders"][t], frames[t]))).convert("RGB")
        small = img.resize((PW, PH), Image.LANCZOS)
        players = per_frame[t]["players"]
        actions = steady_actions(per_frame, t)
        canvas = Image.new("RGB", (W, H), BG)
        canvas.paste(panel(small, players, actions, "find", scale), (0, 0))
        canvas.paste(panel(small, players, actions, "actions", scale), (PW, 0))
        canvas.paste(panel(small, players, actions, "teams", scale), (0, PH))
        canvas.paste(bars(probs[t], leaders[t], truth_at.get(t)), (PW, PH))
        d = ImageDraw.Draw(canvas, "RGBA")
        chip(d, 18, 16, "1  ·  find the players", F_PANEL, dot=(255, 255, 255))
        chip(d, PW + 18, 16, "2  ·  what each player does", F_PANEL, dot=ACTION_RGB["spike"])
        chip(d, 18, PH + 16, "3  ·  split the two teams", F_PANEL, dot=LEFT)
        chip(d, PW + 18, PH + 16, "4  ·  what the team is doing", F_PANEL, dot=OK)
        d.line((PW, 0, PW, H), fill=BG, width=6)
        d.line((0, PH, W, PH), fill=BG, width=6)
        canvas.save(tmp / f"f{n:05d}.jpg", quality=93)
    for k in range(1, int(FPS * args.hold) + 1):  # hold the last verdict so it can be read
        shutil.copy(tmp / f"f{n:05d}.jpg", tmp / f"f{n + k:05d}.jpg")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(tmp / "f%05d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            args.out,
        ],
        check=True,
    )
    shutil.rmtree(tmp)
    print(f"{hi - lo} frames + {args.hold} s hold -> {args.out}")


if __name__ == "__main__":
    main()
