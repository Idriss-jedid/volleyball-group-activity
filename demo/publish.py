"""Publish the project page to the gh-pages branch: site/index.html + the demo videos + figures.

    python demo/publish.py            # build into build/site and push gh-pages
    python demo/publish.py --dry-run  # build only

Inputs: demo/out/demo.mp4 + chapters.json (demo/build.py), demo/out/linkedin.mp4 (gar.demo.live_render),
demo/out/stills/ (demo/build.py --stills). Commits carry the repository's own author only.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "out"
DIST = ROOT / "build" / "site"
CHAPTERS = {
    "hook": "Intro", "goal": "The goal", "paper": "The paper", "plan": "Three phases", "data": "The data",
    "insight": "What the labels tell", "model": "The paper's model", "v1": "Phase 1 · reproduction",
    "v2": "Phase 2 · modernization", "teams": "Team pooling", "v3": "Phase 3 · my architecture",
    "ladder": "Results", "detect": "Detectors", "finetune": "Fine-tuned YOLO", "live": "Full system", "end": "Summary",
}
FIGURES = {"insight": "05_insight", "v2": "08_v2", "ladder": "11_ladder", "detect": "12_detect"}


def git(*args, cwd=ROOT, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def build():
    shutil.rmtree(DIST, ignore_errors=True)
    (DIST / "img").mkdir(parents=True)
    chapters = [[t, CHAPTERS.get(i, i)] for t, i in json.loads((OUT / "chapters.json").read_text())]
    html = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    html = html.replace("/*CHAPTERS*/[]", json.dumps(chapters, ensure_ascii=False))
    (DIST / "index.html").write_text(html, encoding="utf-8")
    (DIST / ".nojekyll").write_text("")
    shutil.copy(OUT / "demo.mp4", DIST / "demo.mp4")
    shutil.copy(OUT / "linkedin.mp4", DIST / "linkedin.mp4")
    for name, still in FIGURES.items():
        shutil.copy(OUT / "stills" / f"{still}.jpg", DIST / "img" / f"{name}.jpg")
    for name, (video, t) in {"poster_demo": ("demo.mp4", 1.5), "poster_wild": ("linkedin.mp4", 2.0)}.items():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", str(OUT / video), "-frames:v", "1",
                        "-q:v", "3", str(DIST / "img" / f"{name}.jpg")], check=True)
    print(f"built {DIST}")


def publish():
    work = ROOT / "build" / "gh-pages"
    shutil.rmtree(work, ignore_errors=True)
    git("worktree", "prune")
    if git("ls-remote", "--exit-code", "--heads", "origin", "gh-pages", check=False).returncode == 0:
        git("fetch", "origin", "gh-pages")
        git("worktree", "add", "-B", "gh-pages", str(work), "origin/gh-pages")
    else:
        git("worktree", "add", "--detach", str(work))
        git("checkout", "--orphan", "gh-pages", cwd=work)
        git("rm", "-rf", "--quiet", ".", cwd=work)
    for p in work.iterdir():
        if p.name != ".git":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    shutil.copytree(DIST, work, dirs_exist_ok=True)
    git("add", "-A", cwd=work)
    if git("diff", "--cached", "--quiet", cwd=work, check=False).returncode:
        git("commit", "-m", "Update project page", cwd=work)
        git("push", "origin", "gh-pages", cwd=work)
        print("pushed gh-pages")
    else:
        print("gh-pages already up to date")
    git("worktree", "remove", "--force", str(work))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build()
    if not args.dry_run:
        publish()


if __name__ == "__main__":
    main()
