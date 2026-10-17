"""Kaggle entry point: build the crop cache, then run the pipeline steps on the GPU.

Pushed as a script kernel by scripts/kaggle_push.py (which can override STEPS). Inputs:
  - sherif31/group-activity-recognition-volleyball   raw frames, videos/<vid>/<clip>/<frame>.jpg
  - <user>/gar-code                                  src/, configs/, data/annotations.pkl
  - optionally the output of an earlier pipeline kernel: its features and checkpoints are reused
Everything is written under /kaggle/working (= the kernel output). A failed step is logged and the
run goes on, so one bug does not waste a 10-hour session; results/steps.json records each step.

Step names: any config name trains that config, plus
  crops, feats_person, feats_image, feats_person_cnx      caches
  v1/feats_ft, v1/feats_imagenet, v1/feats_image          V1 (AlexNet fc7) caches
  logits                                                  re-evaluate old runs to save their logits
  detect                                                  every detector in DETECTORS, end to end
  finetune_yolo, detect_ft                                YOLO11m fine-tuned on players, then evaluated
  ensemble                                                late fusion of the combinations in ENSEMBLES
  ensemble_greedy                                         greedy (Caruana) selection over every run
  optuna/<space>                                          gar.tune with configs/optuna/<space>.yaml
  tta/<run>                                               mirror test-time augmentation -> results/<run>_tta
  drop_crops                                              delete the crop cache (disk is limited to 20 GB)
"""

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

STEPS = ["crops", "b1", "person", "feats_person", "b3", "feats_image", "b4", "person_lstm", "b6", "b5", "b7", "b8"]
DETECTORS = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11x.pt",
    "yolov8m.pt",
    "rtdetr-l.pt",
    "frcnn_r50",
    "frcnn_mobile",
    "retinanet_r50",
]
# trained models re-evaluated with every detector's boxes instead of the ground-truth tracks
EVAL_DET_RUNS = ["b8", "transformer", "e2e", "e2e_cnx"]
ENSEMBLES = [
    ["b8", "b4"],
    ["b8", "transformer"],
    ["b8", "transformer", "b4"],
    ["b8", "e2e"],
    ["e2e", "transformer", "b4"],
    ["b8", "e2e", "transformer", "b4"],
    ["b8_cnx", "transformer_cnx", "b1_cnx"],
    ["b8", "b8_cnx", "e2e", "e2e_cnx", "transformer_cnx", "b4"],
]
KEEP_FEATURES = True  # keep data/features in the output so later kernels can reuse them

WORK = Path("/kaggle/working")


def find_all(predicate, root=Path("/kaggle/input"), depth=5):
    """Breadth-first over /kaggle/input, without descending into the 200k-file videos/ tree."""
    found, level = [], [root]
    for _ in range(depth):
        nxt = []
        for d in level:
            if predicate(d):
                found.append(d)
                continue
            with contextlib.suppress(PermissionError, OSError):
                nxt += [p for p in d.iterdir() if p.is_dir() and p.name not in ("videos", "features")]
        level = nxt
    return found


def find_input(predicate):
    found = find_all(predicate)
    if not found:
        raise FileNotFoundError("input not found under /kaggle/input")
    return found[0]


def reuse_previous_output():
    """Reuse the features and checkpoints of every earlier pipeline kernel attached as input."""
    for prev in find_all(lambda d: (d / "results").is_dir() and (d / "data" / "features").is_dir()):
        print(f"reusing outputs of {prev}", flush=True)
        for feats in (prev / "data" / "features").iterdir():
            dst = WORK / "data" / "features" / feats.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(feats, target_is_directory=True)  # read-only memmaps: no need to copy GBs
        # merge: a run seen in several kernels keeps every file (e.g. logits added by a later kernel)
        shutil.copytree(prev / "results", WORK / "results", dirs_exist_ok=True)


def main():
    os.chdir(WORK)
    code = find_input(lambda d: (d / "src" / "gar" / "__init__.py").exists())
    videos = find_input(lambda d: (d / "videos" / "0").is_dir())
    annot = str(code / "data" / "annotations.pkl")
    print(f"code: {code}\nvideos: {videos}", flush=True)
    reuse_previous_output()

    import torch  # never run the pipeline on CPU: it would burn hours for nothing

    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        sys.exit("no GPU attached: check the accelerator setting / phone verification of the account")
    print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    overlay = WORK / "kaggle_overlay.yaml"
    overlay.write_text(
        json.dumps(
            {  # JSON is valid YAML
                "paths": {"videos": str(videos), "annotations": annot},
                "train": {"amp": True, "num_workers": 4},
            }
        )
    )
    env = dict(os.environ, PYTHONPATH=str(code / "src"), GAR_OVERLAY=str(overlay), PYTHONUNBUFFERED="1")
    py = [sys.executable, "-m"]
    cfg = lambda name: str(code / "configs" / f"{name}.yaml")
    status = {}

    def run(name, cmd):
        print(f"\n=== {time.strftime('%H:%M:%S')} {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        ok = subprocess.run(cmd, env=env).returncode == 0
        status[name] = {"ok": ok, "minutes": round((time.time() - t0) / 60, 1)}
        print(f"=== {name} {'done' if ok else 'FAILED'} in {status[name]['minutes']} min", flush=True)
        Path("results").mkdir(exist_ok=True)
        Path("results/steps.json").write_text(json.dumps(status, indent=2))
        return ok

    def extract(name, config, checkpoint, *extra):  # checkpoint None -> frozen ImageNet weights
        ckpt = ["--checkpoint", checkpoint] if checkpoint else []
        return run(name, py + ["gar.features.extractor", "--config", cfg(config), *ckpt, "--workers", "4", *extra])

    def detect_one(spec, name):
        """Detect + track + select players, rebuild their crops and features, re-evaluate the models."""
        det = f"data/processed/det/{name}"
        if not run(
            f"detect_{name}",
            py
            + [
                "gar.detect.pipeline",
                "--detector",
                spec,
                "--name",
                name,
                "--videos",
                str(videos),
                "--annotations",
                annot,
                "--out-dir",
                "data/processed/det",
            ],
        ):
            return
        run(
            f"crops_{name}",
            py
            + [
                "gar.data.crops",
                "--videos",
                str(videos),
                "--annotations",
                f"{det}/annotations.pkl",
                "--out",
                f"{det}/crops",
                "--workers",
                "4",
            ],
        )
        extract(
            f"feats_{name}",
            "person",
            "results/person/best.pt",
            "--name",
            f"person_ft_det_{name}",
            "--annotations",
            f"{det}/annotations.pkl",
            "--crops",
            f"{det}/crops",
        )
        for model in EVAL_DET_RUNS:
            if (WORK / "results" / model / "best.pt").exists():
                run(
                    f"eval_{model}_{name}",
                    py
                    + [
                        "gar.evaluate",
                        "--run",
                        f"results/{model}",
                        "--annotations",
                        f"{det}/annotations.pkl",
                        "--features",
                        f"person_ft_det_{name}",
                        "--crops",
                        f"{det}/crops",
                        "--tag",
                        f"det_{name}",
                    ],
                )
        shutil.rmtree(f"{det}/crops", ignore_errors=True)
        shutil.rmtree(f"data/features/person_ft_det_{name}", ignore_errors=True)

    if any(step.startswith(("detect", "finetune")) for step in STEPS):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=False)

    for step in STEPS:
        if step == "crops":
            run(
                step,
                py
                + [
                    "gar.data.crops",
                    "--videos",
                    str(videos),
                    "--annotations",
                    annot,
                    "--out",
                    "data/processed/crops",
                    "--workers",
                    "4",
                ],
            )
        elif step == "feats_person":
            extract(step, "person", "results/person/best.pt", "--name", "person_ft")
        elif step == "feats_person_cnx":
            extract(step, "person_cnx", "results/person_cnx/best.pt", "--name", "person_cnx")
        elif step == "feats_image":
            extract(step, "b1", "results/b1/best.pt", "--name", "b1_image", "--level", "image")
        # V1, the paper-faithful AlexNet reproduction (configs/v1)
        elif step == "v1/feats_ft":
            extract(step, "v1/person", "results/v1/person/best.pt", "--name", "v1_ft")
        elif step == "v1/feats_imagenet":
            extract(step, "v1/person", None, "--name", "v1_imagenet")
        elif step == "v1/feats_image":
            extract(step, "v1/b1", "results/v1/b1/best.pt", "--name", "v1_image", "--level", "image")
        elif step == "logits":  # runs trained before logits were saved
            for r in sorted(Path("results").iterdir()):
                if (r / "best.pt").exists() and not (r / "logits_test.npz").exists():
                    for split in ("val", "test"):
                        run(f"logits_{r.name}_{split}", py + ["gar.evaluate", "--run", str(r), "--split", split])
        elif step == "detect":
            for spec in DETECTORS:
                detect_one(spec, spec.removesuffix(".pt").replace("-", "_"))
        elif step == "finetune_yolo":  # YOLO11m fine-tuned on the train videos' player boxes
            run(
                step,
                py
                + [
                    "gar.detect.finetune",
                    "--videos",
                    str(videos),
                    "--annotations",
                    annot,
                    "--out",
                    "data/processed/yolo_ft",
                    "--weights",
                    "yolo11m.pt",
                    "--name",
                    "yolo11m_volleyball",
                    "--results",
                    "results",
                ],
            )
        elif step == "detect_ft":
            detect_one("results/yolo11m_volleyball.pt", "yolo11m_volleyball")
        elif step.startswith("optuna/"):
            space = step.split("/", 1)[1]
            run(
                step,
                py
                + [
                    "gar.tune",
                    "--space",
                    str(code / "configs" / "optuna" / f"{space}.yaml"),
                    "--name",
                    f"optuna_{space}",
                ],
            )
        elif step.startswith("tta/"):
            name = step.split("/", 1)[1]
            for split in ("val", "test"):
                run(
                    f"{step}_{split}",
                    py
                    + [
                        "gar.evaluate",
                        "--run",
                        f"results/{name}",
                        "--split",
                        split,
                        "--tta",
                        "--out",
                        f"results/{name}_tta",
                    ],
                )
        elif step == "ensemble_greedy":
            run(step, py + ["gar.ensemble", "--greedy"])
        elif step == "ensemble":
            for runs in ENSEMBLES:
                if all((WORK / "results" / r / "logits_test.npz").exists() for r in runs):
                    run("ensemble_" + "+".join(runs), py + ["gar.ensemble", "--runs", *runs])
        elif step == "drop_crops":  # free 3.4 GB of the 20 GB working disk once no step needs crops
            shutil.rmtree("data/processed/crops", ignore_errors=True)
        else:
            run(step, py + ["gar.train", "--config", cfg(step), "--name", step])
            (WORK / "results" / step / "last.pt").unlink(missing_ok=True)  # only needed to resume

    # keep the output small: crops can be rebuilt in ~6 min, last.pt is only for resuming
    shutil.rmtree("data/processed/crops", ignore_errors=True)
    shutil.rmtree("data/processed/yolo_ft/images", ignore_errors=True)  # symlinks / extracted frames
    for link in Path("data/features").glob("*"):
        if link.is_symlink():
            link.unlink()  # reused inputs are not part of this kernel's output
    if not KEEP_FEATURES:
        shutil.rmtree("data/features", ignore_errors=True)
    for p in Path("results").rglob("last.pt"):
        p.unlink()
    write_summary()


def write_summary():
    rows = []
    for m in sorted(Path("results").rglob("metrics_test.json")):
        d = json.loads(m.read_text())
        rows.append(
            f"| {m.parent.relative_to('results').as_posix()} | {d['accuracy'] * 100:.1f} | {d['macro_f1'] * 100:.1f} |"
        )
    lines = ["| run | test acc | test macro-F1 |", "|---|---:|---:|", *rows, ""]

    ens = sorted(Path("results").glob("ensemble_*.json"))
    if ens:
        lines += ["| ensemble | weights | val acc | test acc | test macro-F1 |", "|---|---|---:|---:|---:|"]
        for e in ens:
            d = json.loads(e.read_text())
            lines.append(
                f"| {e.stem.removeprefix('ensemble_')} | {d['weights']} | {d['val_accuracy'] * 100:.1f} | "
                f"{d['test_accuracy'] * 100:.1f} | {d['test_macro_f1'] * 100:.1f} |"
            )
        lines.append("")
    for sj in sorted(Path("results").glob("optuna_*/seeds.json")):
        d = json.loads(sj.read_text())
        lines.append(
            f"{sj.parent.name}: best val {d['best_val'] * 100:.1f}, seeds test "
            f"{d['seed_test_mean'] * 100:.1f} ± {d['seed_test_std'] * 100:.1f}, "
            f"seed ensemble test {d['seed_ensemble_test_accuracy'] * 100:.1f}"
        )

    dets = sorted(Path("data/processed/det").glob("*/quality.json"))
    if dets:
        header = "| detector | fps | recall | precision | " + " | ".join(f"{m} acc" for m in EVAL_DET_RUNS) + " |"
        lines += [header, "|---" * (4 + len(EVAL_DET_RUNS)) + "|"]
        for q in dets:
            d, name = json.loads(q.read_text()), q.parent.name
            accs = []
            for m in EVAL_DET_RUNS:
                f = Path("results") / m / f"metrics_test_det_{name}.json"
                accs.append(f"{json.loads(f.read_text())['accuracy'] * 100:.1f}" if f.exists() else "-")
            lines.append(
                f"| {name} | {d['fps']} | {d['test']['recall'] * 100:.1f} | "
                f"{d['test']['precision'] * 100:.1f} | " + " | ".join(accs) + " |"
            )
            shutil.copy(q, Path("results") / f"detection_{name}.json")
    Path("results/summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
