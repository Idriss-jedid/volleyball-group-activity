"""Build the narrated demo video: narration -> sentence timing -> screen recording -> final mix.

    python demo/build.py            # out/demo.mp4
    python demo/build.py --stills   # out/stills/, one screenshot per slide (layout check)
    python demo/build.py --mix      # redo only the final mix, e.g. after a video of the deck changed

Every sentence of narration.json is spoken separately, and the deck reveals what that sentence talks
about at the moment it starts (deck.html, __step). The videos of the deck are not recorded from the
browser (which drops frames): build.py composites the original MP4s over their placeholders, burns in
subtitles timed on the voice, and mixes the audio.

Needs: edge-tts (pip), Node with playwright-core (npm install in demo/), Microsoft Edge, ffmpeg (libass).
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
VOICE, RATE = "en-US-AndrewMultilingualNeural", "-4%"
BEFORE, AFTER, SLIDE_END = 0.3, 0.35, 0.7  # seconds around each sentence, and after the last one of a slide
FADE = 0.5  # slide cross-fade of deck.html


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout)


def speak(key, text):
    mp3, txt = OUT / "tts" / f"{key}.mp3", OUT / "tts" / f"{key}.txt"
    if not mp3.exists() or not txt.exists() or txt.read_text(encoding="utf-8") != text + VOICE + RATE:
        cmd = [sys.executable, "-m", "edge_tts", "--voice", VOICE, f"--rate={RATE}", "--text", text,
               "--write-media", str(mp3)]
        for attempt in range(4):  # the TTS service drops some requests
            if subprocess.run(cmd, capture_output=True).returncode == 0 and mp3.stat().st_size > 1000:
                break
            time.sleep(2 + 3 * attempt)
        else:
            raise RuntimeError(f"edge-tts failed: {key}")
        txt.write_text(text + VOICE + RATE, encoding="utf-8")
    return duration(mp3)


def narrate(slides):
    """Speak every sentence; write timing.js (milliseconds per sentence, per slide)."""
    (OUT / "tts").mkdir(parents=True, exist_ok=True)
    jobs = [(f"{s['id']}_{k}", text) for s in slides for k, text in enumerate(s["lines"])]
    with ThreadPoolExecutor(3) as pool:
        durations = dict(zip((key for key, _ in jobs), pool.map(lambda job: speak(*job), jobs)))
    timing = {}
    for s in slides:
        n = len(s["lines"])
        timing[s["id"]] = [
            round((BEFORE + durations[f"{s['id']}_{k}"] + AFTER + (SLIDE_END if k == n - 1 else 0)) * 1000)
            for k in range(n)
        ]
    (HERE / "timing.js").write_text(f"window.TIMING = {json.dumps(timing)};\n", encoding="utf-8")
    total = sum(sum(v) for v in timing.values()) / 1000
    print(f"narration: {len(jobs)} sentences, {total:.1f} s")
    return durations


def ass_time(t):
    h, rem = divmod(max(0.0, t), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def subtitles(slides, marks, durations, lead):
    rows = []
    for s, mark in zip(slides, marks["slides"]):
        for k, text in enumerate(s["lines"]):
            start = mark["lines"][k] - lead + BEFORE
            end = start + durations[f"{s['id']}_{k}"] + 0.25
            rows.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Sub,,0,0,0,,{text}")
    head = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        "Style: Sub,Segoe UI Semibold,44,&H00FFFFFF,&H00FFFFFF,&H30281710,&H30281710,0,0,0,0,100,100,0,0,3,12,0,"
        "2,240,240,40,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    (OUT / "subs.ass").write_text("\n".join(head + rows) + "\n", encoding="utf-8")


def mix(slides, durations):
    marks = json.loads((OUT / "marks.json").read_text(encoding="utf-8"))
    lead = marks["lead"]
    subtitles(slides, marks, durations, lead)
    starts = [m["start"] - lead for m in marks["slides"]] + [marks["end"] - lead]

    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(lead), "-i", "deck.webm"]
    filters, last, n = ["[0:v]fps=25,setpts=PTS-STARTPTS[v0]"], "v0", 1
    for i, m in enumerate(marks["slides"]):  # the deck's videos, at their own frame rate
        for v in m["videos"]:
            s, e = starts[i], starts[i + 1] + FADE
            cmd += ["-stream_loop", "-1", "-t", f"{e - s:.2f}", "-i", str(HERE / "assets" / v["src"])]
            filters.append(
                f"[{n}:v]fps=25,scale={v['w']}:{v['h']},format=yuva420p,"
                f"fade=in:st=0:d={FADE}:alpha=1,fade=out:st={e - s - FADE:.2f}:d={FADE}:alpha=1,"
                f"setpts=PTS-STARTPTS+{s:.3f}/TB[o{n}]"
            )
            filters.append(f"[{last}][o{n}]overlay={v['x']}:{v['y']}:eof_action=pass[v{n}]")
            last, n = f"v{n}", n + 1
    filters.append(f"[{last}]subtitles=subs.ass[vout]")

    audio = []
    for s, m in zip(slides, marks["slides"]):  # every sentence starts when the deck reveals it
        for k in range(len(s["lines"])):
            cmd += ["-i", str(OUT / "tts" / f"{s['id']}_{k}.mp3")]
            delay = round((m["lines"][k] - lead + BEFORE) * 1000)
            filters.append(f"[{n}:a]adelay={delay}|{delay}[a{n}]")
            audio.append(f"[a{n}]")
            n += 1
    filters.append(f"{''.join(audio)}amix=inputs={len(audio)}:normalize=0[aout]")
    cmd += [
        "-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "25",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-t", f"{starts[-1]:.2f}", "demo.mp4",
    ]
    subprocess.run(cmd, check=True, cwd=OUT)
    print(f"-> {OUT / 'demo.mp4'} ({duration(OUT / 'demo.mp4'):.1f} s)")
    chapters = [[round(t, 1), s["id"]] for t, s in zip(starts, slides)]
    (OUT / "chapters.json").write_text(json.dumps(chapters), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stills", action="store_true")
    parser.add_argument("--mix", action="store_true", help="only redo the final mix (videos changed)")
    args = parser.parse_args()
    slides = json.loads((HERE / "narration.json").read_text(encoding="utf-8"))
    durations = narrate(slides)
    if args.stills:
        subprocess.run(["node", str(HERE / "record.mjs"), "--stills", str(OUT / "stills")], check=True)
        return
    if not args.mix:
        subprocess.run(["node", str(HERE / "record.mjs")], check=True)
    mix(slides, durations)


if __name__ == "__main__":
    main()
