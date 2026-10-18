"""Plain-language demo video: one full-screen clip, team colours, simple action words, a big verdict.

Made for people who do not know the project (LinkedIn, the narrated demo): no model names, no
probabilities table, just what the system sees and what it decides.

    [title card] -> clip 1 -> clip 2 -> ... -> [end card]

Usage:
    python -m gar.demo.story --clips results_kaggle/gar-demo/results/demo_clips.json \
        --pick 4:24690 29:17385 47:12895 --cards --out demo/out/linkedin.mp4
    python -m gar.demo.story ... --no-banner --out demo/assets/plain.mp4     # raw clip, players only
"""

import argparse
import io
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from gar.data.frames import FrameSource
from gar.demo.render import BG, LEFT, MUT, OK, RIGHT, TXT, font, held_box

W, H = 1920, 1080
FPS, SLOW, HOLD = 30, 1.6, 45  # motion-interpolated slow motion (1/1.6 speed), verdict held 1.5 s
SRC_FPS = 25
BLACK = ["seguibl.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]
BOLD = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]
SEMI = ["seguisb.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
F_HUGE, F_BIG, F_MID, F_TAG, F_SMALL = font(BLACK, 96), font(BLACK, 66), font(BOLD, 40), font(BOLD, 26), font(SEMI, 38)
WORDS = {
    "spiking": "spike",
    "blocking": "block",
    "setting": "set",
    "digging": "dig",
    "falling": "fall",
    "jumping": "jump",
}  # standing / waiting / moving players get no tag: less clutter
ACTIVITY = {"set": "SET", "spike": "SPIKE", "pass": "PASS", "winpoint": "WINS THE POINT"}


def verdict(activity):
    side, kind = activity.split("_", 1)
    return ("LEFT TEAM" if side == "l" else "RIGHT TEAM"), ACTIVITY[kind], LEFT if side == "l" else RIGHT


def chip(d, x, y, text, f, fill=(10, 14, 26, 215), color=TXT, dot=None):
    h = f.size + 22
    tw = d.textlength(text, font=f)
    pad = h if dot else 18
    d.rounded_rectangle((x, y, x + tw + pad + 18, y + h), radius=h // 2, fill=fill)
    if dot:
        r = f.size // 3
        d.ellipse((x + h // 2 - r, y + h // 2 - r, x + h // 2 + r, y + h // 2 + r), fill=dot)
    d.text((x + pad, y + 9), text, font=f, fill=color)
    return x + tw + pad + 18


def smooth_boxes(player, window=2):
    """Per source frame, the player's box averaged over +-window frames: removes tracker jitter."""
    held = [held_box(player, i) for i in range(len(player["boxes"]))]
    out = []
    for i in range(len(held)):
        near = [b for b in held[max(0, i - window) : i + window + 1] if b is not None]
        out.append([sum(v) / len(near) for v in zip(*near)] if near else None)
    return out


def box_at(boxes, s):
    """Box at fractional source frame s, linearly interpolated."""
    i = min(int(s), len(boxes) - 1)
    j = min(i + 1, len(boxes) - 1)
    a, b, t = boxes[i], boxes[j], s - int(s)
    if a is None or b is None:
        return a or b
    return [u + (v - u) * t for u, v in zip(a, b)]


def check(d, x, y, size, color):
    d.line((x, y + size * 0.55, x + size * 0.38, y + size * 0.9, x + size, y + size * 0.15), fill=color, width=6)


def clip_frame(frame, clip, s, boxes, progress, banner=True):
    """frame: the (interpolated) 1920x1080 image at fractional source frame s."""
    frame = frame.convert("RGBA")
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    sx, sy = W / clip["width"], H / clip["height"]
    if banner:  # darken the bottom so the verdict reads on any court
        grad = Image.new("L", (1, 260))
        grad.putdata([int(200 * (k / 259) ** 1.6) for k in range(260)])
        over.paste((5, 8, 16, 255), (0, H - 260, W, H), grad.resize((W, 260)))
    for pl, track in zip(clip["players"], boxes):
        b = box_at(track, s)
        if b is None:
            continue
        x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
        c = LEFT if pl["team"] == "left" else RIGHT
        d.rounded_rectangle((x1, y1, x2, y2), radius=6, outline=c + (255,), width=3)
        word = WORDS.get(pl["action"])
        if word:
            tw = d.textlength(word, font=F_TAG)
            d.rounded_rectangle((x1, y1 - 40, x1 + tw + 20, y1 - 6), radius=8, fill=c + (245,))
            d.text((x1 + 10, y1 - 38), word, font=F_TAG, fill=(10, 14, 26))
    x = chip(d, 36, 32, "What is the team doing?", F_MID)
    chip(d, x + 14, 32, "left team", F_MID, dot=LEFT)
    chip(d, x + 14 + d.textlength("left team", font=F_MID) + 90, 32, "right team", F_MID, dot=RIGHT)
    if banner:
        if progress < 1:
            d.text((60, H - 150), "watching the players…", font=F_BIG, fill=TXT + (235,))
            d.rounded_rectangle((60, H - 60, 760, H - 46), radius=7, fill=(255, 255, 255, 60))
            d.rounded_rectangle((60, H - 60, 60 + 700 * progress, H - 46), radius=7, fill=OK + (255,))
        else:
            team, act, c = verdict(clip["activity"])
            d.text((60, H - 172), team, font=F_BIG, fill=c + (255,))
            tx = 60 + d.textlength(team, font=F_BIG) + 30
            d.text((tx, H - 172), "·  " + act, font=F_BIG, fill=TXT + (255,))
            ok = clip["activity"] == clip["true_activity"]
            conf = f"{max(clip['probs'].values()) * 100:.0f}% sure"
            x = chip(
                d,
                64,
                H - 92,
                "      correct" if ok else "wrong",
                F_MID,
                fill=(OK if ok else (251, 113, 133)) + (255,),
                color=(10, 14, 26),
            )
            if ok:
                check(d, 88, H - 74, 26, (10, 14, 26))
            chip(d, x + 14, H - 92, conf, F_MID)
    return Image.alpha_composite(frame, over).convert("RGB")


def interpolated(source, clip, work):
    """The clip's frames slowed down with motion-compensated interpolation (ffmpeg minterpolate).

    Yields (fractional source frame index, 1920x1080 image); the boxes are interpolated to match.
    """
    n = len(clip["frames"])
    work = work / f"{clip['video']}_{clip['clip']}"  # cached: interpolation is the slow part
    if not any(work.glob("i*.jpg")):
        make_interpolated(source, clip, work)
    out = sorted(work.glob("i*.jpg"))
    for j, path in enumerate(out):
        yield j / max(1, len(out) - 1) * (n - 1), Image.open(path).convert("RGB")


def make_interpolated(source, clip, work):
    shutil.rmtree(work, ignore_errors=True)
    (work / "raw").mkdir(parents=True)
    for i, f in enumerate(clip["frames"]):
        img = Image.open(io.BytesIO(source.frame(clip["video"], clip["clip"], f))).convert("RGB")
        img.resize((W, H), Image.LANCZOS).save(work / "raw" / f"r{i:03d}.jpg", quality=95)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(SRC_FPS),
            "-i",
            str(work / "raw" / "r%03d.jpg"),
            "-vf",
            f"setpts={SLOW}*PTS,minterpolate=fps={FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
            "-q:v",
            "2",
            str(work / "i%04d.jpg"),
        ],
        check=True,
    )


PW, PH = W // 2, H // 2
F_PANEL, F_GTAG, F_BAR = font(BOLD, 30), font(BOLD, 19), font(BOLD, 26)
F_VERDICT, F_OK = font(BLACK, 44), font(BOLD, 24)
ACTION_RGB = {
    "spike": (251, 113, 133),
    "block": (167, 139, 250),
    "set": (52, 211, 153),
    "dig": (250, 204, 21),
    "fall": (248, 113, 113),
    "jump": (244, 114, 182),
}
KIND = {"set": "set", "spike": "spike", "pass": "pass", "winpoint": "point won"}
PLAIN = {
    a: ("left" if a[0] == "l" else "right") + " · " + KIND[a.split("_", 1)[1]]
    for a in ["l_set", "l_spike", "l_pass", "l_winpoint", "r_set", "r_spike", "r_pass", "r_winpoint"]
}


def panel_title(d, n, text, color):
    chip(d, 18, 16, f"{n}  ·  {text}", F_PANEL, dot=color)


def video_panel(small, clip, s, boxes, mode, size=(PW, PH)):
    """A panel over the (interpolated) frame: 'find', 'actions' or 'teams' (960x540, or full size)."""
    pw, ph = size
    p = small.convert("RGBA")
    over = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    sx, sy = pw / clip["width"], ph / clip["height"]
    centres = []
    for pl, track in zip(clip["players"], boxes):
        b = box_at(track, s)
        if b is None:
            continue
        x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
        team = LEFT if pl["team"] == "left" else RIGHT
        centres.append(((x1 + x2) / 2, pl["team"]))
        if mode == "find":
            d.rounded_rectangle((x1, y1, x2, y2), radius=4, outline=(255, 255, 255, 240), width=3)
        elif mode == "actions":
            word = WORDS.get(pl["action"])
            c = ACTION_RGB.get(word, (148, 163, 184))
            d.rounded_rectangle((x1, y1, x2, y2), radius=4, outline=c + (255 if word else 150,), width=3 if word else 2)
            if word:
                tw = d.textlength(word, font=F_GTAG)
                d.rounded_rectangle((x1, y1 - 28, x1 + tw + 14, y1 - 4), radius=6, fill=c + (250,))
                d.text((x1 + 7, y1 - 27), word, font=F_GTAG, fill=(10, 14, 26))
        else:
            d.rounded_rectangle((x1, y1, x2, y2), radius=4, fill=team + (80,), outline=team + (255,), width=3)
    if mode == "teams":
        lx = [x for x, t in centres if t == "left"]
        rx = [x for x, t in centres if t == "right"]
        if lx and rx:
            split = (max(lx) + min(rx)) / 2
            for y in range(70, ph - 70, 16):
                d.line((split, y, split, y + 9), fill=(255, 255, 255, 220), width=3)
            chip(d, split - 196, ph - 62, "left team", F_PANEL, dot=LEFT)
            chip(d, split + 16, ph - 62, "right team", F_PANEL, dot=RIGHT)
    return Image.alpha_composite(p, over).convert("RGB")


def stats_panel(clip, progress):
    """Live probabilities of the 8 team activities; the winner and the verdict once decided."""
    p = Image.new("RGB", (PW, PH), (16, 23, 40))
    d = ImageDraw.Draw(p)
    probs = clip["probs"]
    order = sorted(probs, key=lambda a: -probs[a])
    x0, y0, bw = 250, 82, 560
    for k, a in enumerate(order):
        y = y0 + k * 42
        v = probs[a] * min(1.0, progress)
        win = progress >= 1 and a == clip["activity"]
        name = PLAIN[a]
        d.text((x0 - 18 - d.textlength(name, font=F_BAR), y), name, font=F_BAR, fill=TXT if win else MUT)
        d.rounded_rectangle((x0, y + 6, x0 + bw, y + 30), radius=12, fill=(34, 46, 72))
        c = OK if win else (LEFT if a[0] == "l" else RIGHT)
        if v > 0.01:
            d.rounded_rectangle((x0, y + 6, x0 + max(24, bw * v), y + 30), radius=12, fill=c)
        d.text((x0 + bw + 14, y + 2), f"{v * 100:.0f}%", font=F_BAR, fill=TXT if win else MUT)
    if progress >= 1:
        team, act, c = verdict(clip["activity"])
        ok = clip["activity"] == clip["true_activity"]
        msg = f"{team} · {act}"
        tw = d.textlength(msg, font=F_VERDICT)
        tag = "correct" if ok else "wrong"
        cw = d.textlength(tag, font=F_OK) + 52
        x = PW / 2 - (tw + 24 + cw) / 2
        d.text((x, PH - 92), msg, font=F_VERDICT, fill=c)
        bx = x + tw + 24
        d.rounded_rectangle((bx, PH - 82, bx + cw, PH - 44), radius=19, fill=OK if ok else (251, 113, 133))
        if ok:
            check(d, bx + 12, PH - 73, 20, (10, 14, 26))
        d.text((bx + 40, PH - 80), tag, font=F_OK, fill=(10, 14, 26))
    else:
        msg = "deciding…"
        d.text((PW / 2 - d.textlength(msg, font=F_VERDICT) / 2, PH - 92), msg, font=F_VERDICT, fill=MUT)
    return p


def grid_frame(frame, clip, s, boxes, progress):
    small = frame.resize((PW, PH), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(video_panel(small, clip, s, boxes, "find"), (0, 0))
    canvas.paste(video_panel(small, clip, s, boxes, "actions"), (PW, 0))
    canvas.paste(video_panel(small, clip, s, boxes, "teams"), (0, PH))
    canvas.paste(stats_panel(clip, progress), (PW, PH))
    d = ImageDraw.Draw(canvas, "RGBA")
    panel_title(d, 1, "find the players", (255, 255, 255))
    d2 = ImageDraw.Draw(canvas, "RGBA")
    chip(d2, PW + 18, 16, "2  ·  what each player does", F_PANEL, dot=ACTION_RGB["spike"])
    chip(d2, 18, PH + 16, "3  ·  split the two teams", F_PANEL, dot=LEFT)
    chip(d2, PW + 18, PH + 16, "4  ·  what the team is doing", F_PANEL, dot=OK)
    d.line((PW, 0, PW, H), fill=BG, width=6)
    d.line((0, PH, W, PH), fill=BG, width=6)
    return canvas


def card(lines):
    """Dark title / end card: [(text, font, colour), ...] centred."""
    img = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), BG)
    ImageDraw.Draw(glow).ellipse((W / 2 - 700, H / 2 - 420, W / 2 + 700, H / 2 + 420), fill=(32, 44, 92))
    img = Image.blend(img, glow.filter(ImageFilter.GaussianBlur(160)), 0.85)
    d = ImageDraw.Draw(img)
    total = sum(f.size * 1.35 for _, f, _ in lines)
    y = H / 2 - total / 2
    for text, f, c in lines:
        d.text((W / 2 - d.textlength(text, font=f) / 2, y), text, font=f, fill=c)
        y += f.size * 1.35
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clips", required=True)
    parser.add_argument("--pick", nargs="+", required=True, help="clips as video:clip, in order")
    parser.add_argument("--videos", default="D:/volleyball.zip")
    parser.add_argument("--cards", action="store_true", help="add the title and end cards (LinkedIn)")
    parser.add_argument("--no-banner", action="store_true", help="players only, no verdict")
    parser.add_argument(
        "--layout",
        choices=["story", "grid", "teams", "raw"],
        default="story",
        help="grid: 4 panels (LinkedIn) · teams / raw: one full-screen view",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    clips = {(c["video"], c["clip"]): c for c in json.loads(Path(args.clips).read_text(encoding="utf-8"))}
    source = FrameSource(args.videos)
    tmp = Path(args.out).with_suffix("")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    frames = []

    def emit(img, n=1):
        for _ in range(n):
            path = tmp / f"f{len(frames):05d}.jpg"
            img.save(path, quality=93)
            frames.append(path)

    def fade(a, b, n=12):
        for k in range(1, n + 1):
            emit(Image.blend(a, b, k / n))

    title = card(
        [
            ("Can AI tell what a", F_HUGE, TXT),
            ("volleyball team is doing?", F_HUGE, TXT),
            ("from ordinary broadcast video  ·  12 players  ·  8 team activities", F_SMALL, MUT),
        ]
    )
    end = card(
        [
            ("93.9% accuracy", F_HUGE, OK),
            ("on matches it has never seen", F_BIG, TXT),
            ("", F_SMALL, TXT),
            ("CVPR 2016 paper reimplemented, then my own end-to-end model", F_SMALL, MUT),
            ("Idriss Abderraouf Jedid  ·  github.com/Idriss-jedid/volleyball-group-activity", F_SMALL, MUT),
        ]
    )
    prev = None
    if args.cards:
        emit(title, int(FPS * 2.2))
        prev = title
    for key in args.pick:
        clip = clips[tuple(map(int, key.split(":")))]
        target = clip["target_index"]
        boxes = [smooth_boxes(pl) for pl in clip["players"]]
        for s, img in interpolated(source, clip, Path(args.out).parent / "interp_cache"):
            progress = min(1.0, s / target)
            if args.layout == "grid":
                frame = grid_frame(img, clip, s, boxes, progress)
            elif args.layout == "teams":
                frame = video_panel(img, clip, s, boxes, "teams", size=(W, H))
            elif args.layout == "raw":
                frame = img
            else:
                frame = clip_frame(img, clip, s, boxes, progress, banner=not args.no_banner)
            if prev is not None:
                fade(prev, frame)
                prev = None
            emit(frame)
        last = frame
        emit(last, HOLD)
        prev = last
    if args.cards:
        fade(prev, end, 15)
        emit(end, int(FPS * 2.8))
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
    print(f"{len(frames)} frames, {len(frames) / FPS:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
