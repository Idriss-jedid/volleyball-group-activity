"""Late fusion of several trained models: weighted sum of softmax probabilities.

The weights are chosen on val (grid over the simplex, step 0.1); test is scored once with them.
Every group model evaluates the clips in the same order (annotations order), which is checked
through the saved labels.

With --greedy, models are picked automatically among every run that has 8-class logits: greedy
forward selection with replacement (Caruana et al., 2004), each step adding the model that most
improves val accuracy; the weights are the selection counts.

Usage:
    python -m gar.ensemble --runs b1 b8 transformer          # results/<run>/logits_{val,test}.npz
    python -m gar.ensemble --greedy                          # search over every run in results/
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from gar.data.splits import GROUP_ACTIVITIES
from gar.utils.metrics import compute_metrics


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load_split(results, runs, split):
    probs, labels = [], None
    for run in runs:
        d = np.load(Path(results) / run / f"logits_{split}.npz")
        if labels is not None and not np.array_equal(labels, d["labels"]):
            raise ValueError(f"{run}: {split} clips are not in the same order as {runs[0]}")
        labels = d["labels"]
        probs.append(softmax(d["logits"]))
    return np.stack(probs), labels


def weight_grid(n, step=0.1):
    ticks = np.round(np.arange(0, 1 + 1e-9, step), 2)
    for w in itertools.product(ticks, repeat=n - 1):
        if sum(w) <= 1 + 1e-9:
            yield np.array([*w, 1 - sum(w)])


def discover(results):
    """Runs with 8-class val and test logits, all on the same clips as the first one found."""
    runs, ref = [], None
    for f in sorted(Path(results).rglob("logits_test.npz")):
        if not (f.parent / "logits_val.npz").exists():
            continue
        v, t = np.load(f.parent / "logits_val.npz"), np.load(f)
        if v["logits"].ndim != 2 or v["logits"].shape[1] != len(GROUP_ACTIVITIES):
            continue
        key = (v["labels"].tobytes(), t["labels"].tobytes())
        ref = ref or key
        if key == ref:
            runs.append(f.parent.relative_to(results).as_posix())
    return runs


def greedy(results, max_steps=20):
    runs = discover(results)
    pv, yv = load_split(results, runs, "val")
    pt, yt = load_split(results, runs, "test")
    counts, total, history = np.zeros(len(runs)), np.zeros_like(pv[0]), []
    for _ in range(max_steps):
        scores = [((total + pv[i]).argmax(1) == yv).mean() for i in range(len(runs))]
        i = int(np.argmax(scores))
        counts[i] += 1
        total += pv[i]
        history.append((float(scores[i]), counts.copy()))
    best_val, best_counts = max(history, key=lambda h: (h[0], -h[1].sum()))  # earliest best, fewest models
    w = best_counts / best_counts.sum()
    test = compute_metrics(yt.tolist(), np.tensordot(w, pt, 1).argmax(1).tolist(), GROUP_ACTIVITIES)
    return {
        "mode": "greedy",
        "candidates": runs,
        "weights": {r: round(float(x), 3) for r, x in zip(runs, w) if x > 0},
        "val_accuracy": best_val,
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "single_test_accuracy": {r: float((pt[i].argmax(1) == yt).mean()) for i, r in enumerate(runs)},
        "test_confusion_matrix": test["confusion_matrix"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--greedy", action="store_true", help="pick the models automatically on val")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", help="json file for the result, default results/ensemble_<runs>.json")
    args = parser.parse_args()

    if args.greedy:
        result = greedy(args.results)
        out = Path(args.out or Path(args.results) / "ensemble_greedy.json")
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            json.dumps({k: v for k, v in result.items() if k not in ("test_confusion_matrix", "candidates")}, indent=2)
        )
        return
    if not args.runs:
        parser.error("give --runs or --greedy")

    pv, yv = load_split(args.results, args.runs, "val")
    pt, yt = load_split(args.results, args.runs, "test")
    acc = lambda p, y, w: float((np.tensordot(w, p, 1).argmax(1) == y).mean())
    best = max(weight_grid(len(args.runs)), key=lambda w: (acc(pv, yv, w), -np.abs(w - 1 / len(w)).sum()))

    test = compute_metrics(yt.tolist(), np.tensordot(best, pt, 1).argmax(1).tolist(), GROUP_ACTIVITIES)
    result = {
        "runs": args.runs,
        "weights": dict(zip(args.runs, best.round(2).tolist())),
        "val_accuracy": acc(pv, yv, best),
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "single_test_accuracy": {r: float((pt[i].argmax(1) == yt).mean()) for i, r in enumerate(args.runs)},
        "test_confusion_matrix": test["confusion_matrix"],
    }
    out = Path(args.out or Path(args.results) / f"ensemble_{'+'.join(r.replace('/', '-') for r in args.runs)}.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "test_confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
