"""Render the demo predictions (gar.demo.infer) as a 2x2 video, LinkedIn style.

    ┌──────────────────────────┬──────────────────────────┐
    │ ● Player detection       │ ● Each player's action   │
    │   YOLO + tracker         │   per-player head        │
    ├──────────────────────────┼──────────────────────────┤
    │ ● Team pooling           │ ● Team activity · live   │
    │   left | right groups    │   8-class probabilities  │
    └──────────────────────────┴──────────────────────────┘

Usage:
    python -m gar.demo.render --clips demo/out/clips.json --pick 25:77985 29:52510 --out demo/out/linkedin.mp4
"""

import argparse
import io
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from gar.data.frames import FrameSource
from gar.data.splits import GROUP_ACTIVITIES

W, H = 1920, 1080
PW, PH = W // 2, H // 2
BG = (10, 14, 26)
CARD = (20, 27, 46)
LINE = (36, 48, 73)
TXT = (232, 237, 247)
MUT = (139, 151, 179)
LEFT, RIGHT = (96, 165, 250), (251, 146, 60)  # team colours
OK, BAD = (52, 211, 153), (251, 113, 133)
ACTION_COLORS = {
    "spiking": (251, 113, 133),
    "blocking": (167, 139, 250),
    "setting": (52, 211, 153),
    "digging": (250, 204, 21),
    "jumping": (244, 114, 182),
    "falling": (248, 113, 113),
    "moving": (96, 165, 250),
    "waiting": (148, 163, 184),
    "standing": (100, 116, 139),
}
NICE = {a: a.replace("_", " ").replace("r ", "right ").replace("l ", "left ") for a in GROUP_ACTIVITIES}


def font(names, size):
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


BOLD = ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]
MONO = ["consolab.ttf", "consola.ttf", "DejaVuSansMono-Bold.ttf"]
F_PILL, F_LABEL, F_BAR, F_BIG, F_SMALL = font(BOLD, 20), font(BOLD, 15), font(BOLD, 22), font(BOLD, 34), font(MONO, 17)


def pill(draw, xy, text, dot, f=None):
    f = f or F_PILL
    x, y = xy
    h = f.size + 16
    tw = draw.textlength(text, font=f)
    draw.rounded_rectangle((x, y, x + tw + h + 12, y + h), radius=h // 2, fill=(10, 14, 26))
    r = h // 6
    draw.ellipse((x + h // 2 - r, y + h // 2 - r, x + h // 2 + r, y + h // 2 + r), fill=dot)
    draw.text((x + h, y + 6), text, font=f, fill=TXT)


def label(draw, x, y, text, color):
    tw = draw.textlength(text, font=F_LABEL)
    draw.rounded_rectangle((x, y - 22, x + tw + 10, y), radius=5, fill=color)
    draw.text((x + 5, y - 21), text, font=F_LABEL, fill=(10, 14, 26))


def held_box(player, i):
    """Box at frame i, or the closest frame where the tracker saw the player."""
    boxes = player["boxes"]
    if boxes[i] is not None:
        return boxes[i]
    seen = [j for j, b in enumerate(boxes) if b is not None]
    return boxes[min(seen, key=lambda j: abs(j - i))] if seen else None


def scaled(box, sx, sy):
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def panel_frame(img, clip, i, mode, size=(PW, PH)):
    """One video panel: 'raw', 'detect', 'actions' or 'teams'."""
    pw, ph = size
    sx, sy = pw / clip["width"], ph / clip["height"]
    p = img.resize((pw, ph), Image.BILINEAR).convert("RGBA")
    if mode == "raw":
        return p.convert("RGB")
    over = Image.new("RGBA", p.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    xs = []
    for pl in clip["players"]:
        b = held_box(pl, i)
        if b is None:
            continue
        x1, y1, x2, y2 = scaled(b, sx, sy)
        xs.append(((x1 + x2) / 2, pl["team"]))
        if mode == "detect":
            d.rectangle((x1, y1, x2, y2), outline=(255, 255, 255, 230), width=2)
            label(d, x1, y1, f"#{pl['slot'] + 1}", (255, 255, 255, 230))
        elif mode == "actions":
            c = ACTION_COLORS[pl["action"]]
            d.rectangle((x1, y1, x2, y2), outline=c + (255,), width=3)
            if pl["action"] != "standing":
                label(d, x1, y1, f"{pl['action']} {pl['action_prob']:.2f}", c + (255,))
        else:
            c = LEFT if pl["team"] == "left" else RIGHT
            d.rectangle((x1, y1, x2, y2), fill=c + (70,), outline=c + (255,), width=3)
    if mode == "teams" and xs:
        lx = [x for x, t in xs if t == "left"]
        rx = [x for x, t in xs if t == "right"]
        if lx and rx:
            split = (max(lx) + min(rx)) / 2
            for y in range(60, ph - 20, 18):
                d.line((split, y, split, y + 10), fill=(255, 255, 255, 200), width=2)
            pill(d, (split - 222, 64), "left team · pooled", LEFT)
            pill(d, (split + 14, 64), "right team · pooled", RIGHT)
    return Image.alpha_composite(p, over).convert("RGB")


def bars_panel(clip, progress):
    """8 activity probabilities growing until the target frame; the winner and the truth at the end."""
    p = Image.new("RGB", (PW, PH), CARD)
    d = ImageDraw.Draw(p)
    probs = clip["probs"]
    order = sorted(GROUP_ACTIVITIES, key=lambda a: -probs[a])
    x0, y0, bw = 250, 78, 560
    for k, a in enumerate(order):
        y = y0 + k * 46
        v = probs[a] * progress
        win = a == clip["activity"] and progress >= 1
        d.text((x0 - 20 - d.textlength(NICE[a], font=F_BAR), y + 2), NICE[a], font=F_BAR, fill=TXT if win else MUT)
        d.rounded_rectangle((x0, y + 6, x0 + bw, y + 28), radius=11, fill=LINE)
        color = (LEFT if a.startswith("l") else RIGHT) if not win else OK
        if v > 0.005:
            d.rounded_rectangle((x0, y + 6, x0 + max(22, bw * v), y + 28), radius=11, fill=color)
        d.text((x0 + bw + 14, y + 4), f"{v * 100:4.1f}%", font=F_SMALL, fill=TXT if win else MUT)
    if progress >= 1:
        ok = clip["activity"] == clip["true_activity"]
        msg = f"prediction: {NICE[clip['activity']]}   ·   ground truth: {NICE[clip['true_activity']]}"
        d.text((PW / 2 - d.textlength(msg, font=F_LABEL) / 2, PH - 58), msg, font=F_LABEL, fill=OK if ok else BAD)
    return p


SINGLE = {
    "raw": ("Broadcast clip · 41 frames", (255, 255, 255)),
    "detect": ("Player detection · YOLO11 + tracker", (255, 255, 255)),
    "actions": ("Each player's action", ACTION_COLORS["spiking"]),
    "teams": ("Team pooling · left | right", LEFT),
}


def compose(clip, img, i, fps_progress, layout="grid"):
    if layout != "grid":  # one full-screen panel, for the narrated demo
        frame = panel_frame(img, clip, i, layout, size=(W, H))
        pill(ImageDraw.Draw(frame), (24, 22), *SINGLE[layout], f=F_BIG)
        return frame
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(panel_frame(img, clip, i, "detect"), (0, 0))
    canvas.paste(panel_frame(img, clip, i, "actions"), (PW, 0))
    canvas.paste(panel_frame(img, clip, i, "teams"), (0, PH))
    canvas.paste(bars_panel(clip, fps_progress), (PW, PH))
    d = ImageDraw.Draw(canvas)
    pill(d, (18, 16), "Player detection · YOLO11 + tracker", (255, 255, 255))
    pill(d, (PW + 18, 16), "Each player's action", ACTION_COLORS["spiking"])
    pill(d, (18, PH + 16), "Team pooling · left | right", LEFT)
    pill(d, (PW + 18, PH + 16), "Team activity · live", OK)
    d.line((PW, 0, PW, H), fill=BG, width=4)
    d.line((0, PH, W, PH), fill=BG, width=4)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clips", default="demo/out/clips.json")
    parser.add_argument("--pick", nargs="+", required=True, help="clips as video:clip, in order")
    parser.add_argument("--videos", default="D:/volleyball.zip")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--out", default="demo/out/linkedin.mp4")
    parser.add_argument("--still", help="write a single PNG (first clip, target frame) instead of a video")
    parser.add_argument("--layout", default="grid", choices=["grid", *SINGLE])
    args = parser.parse_args()

    clips = {(c["video"], c["clip"]): c for c in json.loads(Path(args.clips).read_text(encoding="utf-8"))}
    source = FrameSource(args.videos)
    tmp = Path(args.out).with_suffix("")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n = 0
    for key in args.pick:
        clip = clips[tuple(map(int, key.split(":")))]
        target = clip["target_index"]
        for i, f in enumerate(clip["frames"]):
            img = Image.open(io.BytesIO(source.frame(clip["video"], clip["clip"], f))).convert("RGB")
            progress = min(1.0, (i + 1) / (target + 1))
            frame = compose(clip, img, i, progress, args.layout)
            if args.still and i == target:
                frame.save(args.still)
                print(f"still -> {args.still}")
                return
            frame.save(tmp / f"f{n:05d}.png")
            n += 1
        for _ in range(args.fps // 2):  # hold the decision for half a second
            frame.save(tmp / f"f{n:05d}.png")
            n += 1
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(tmp / "f%05d.png"),
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
    print(f"{n} frames, {n / args.fps:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
