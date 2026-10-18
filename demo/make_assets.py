"""Still images for the narrated deck: an annotated frame (the data slide) and video posters.

    python demo/make_assets.py

Needs results_kaggle/gar-demo/results/demo_clips.json (kaggle step demo_infer) and the dataset zip.
The posters are the first frame of each video placeholder, shown while the deck is recorded
(the real videos are composited over them by build.py, at full frame rate).
"""

import io
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from gar.data.frames import FrameSource
from gar.demo.render import LEFT, RIGHT
from gar.demo.story import BOLD, F_PANEL, chip, font

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
CLIPS = HERE.parent / "results_kaggle" / "gar-demo" / "results" / "demo_clips.json"
CLIP = (4, 24690)  # the left-team spike of the opening video
F_LABEL = font(BOLD, 22)


def annotated(videos="D:/volleyball.zip"):
    clips = {(c["video"], c["clip"]): c for c in json.loads(CLIPS.read_text(encoding="utf-8"))}
    clip = clips[CLIP]
    t = clip["target_index"]
    source = FrameSource(videos)
    img = Image.open(io.BytesIO(source.frame(clip["video"], clip["clip"], clip["frames"][t]))).convert("RGB")
    scale = 1920 / img.width
    img = img.resize((1920, round(img.height * scale)), Image.LANCZOS).convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for pl in clip["players"]:
        box = pl["boxes"][min(t, len(pl["boxes"]) - 1)]
        if box is None:
            continue
        x1, y1, x2, y2 = (v * scale for v in box)
        c = LEFT if pl["team"] == "left" else RIGHT
        d.rounded_rectangle((x1, y1, x2, y2), radius=5, outline=c + (255,), width=4)
        word = pl["true_action"]
        tw = d.textlength(word, font=F_LABEL)
        d.rounded_rectangle((x1, y1 - 34, x1 + tw + 16, y1 - 4), radius=8, fill=c + (245,))
        d.text((x1 + 8, y1 - 33), word, font=F_LABEL, fill=(10, 14, 26))
    y = img.height - 680  # legend at the top of the cropped court
    chip(d, 24, y, "left team", F_PANEL, dot=LEFT)
    chip(d, 230, y, "right team", F_PANEL, dot=RIGHT)
    out = Image.alpha_composite(img, over).convert("RGB")
    out.crop((0, img.height - 700, 1920, img.height)).save(ASSETS / "annotated.jpg", quality=90)  # the court


def poster(video, name):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(ASSETS / video), "-frames:v", "1", "-q:v", "3", str(ASSETS / name)],
        check=True,
    )


def main():
    annotated()
    for video in ("raw", "teams", "live"):
        if (ASSETS / f"{video}.mp4").exists():
            poster(f"{video}.mp4", f"poster_{video}.jpg")
    print(f"-> {ASSETS}")


if __name__ == "__main__":
    main()
