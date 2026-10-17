"""Hyper-parameter search with Optuna (TPE sampler + median pruning), selected on val only.

The search space is a yaml file (configs/optuna/*.yaml): a base config plus dotted config paths and
their ranges. Each trial trains on the train split and reports the val metric after every epoch, so
weak trials are pruned early; the test split is never loaded during the search. The best config is
then retrained with several seeds (test evaluated once per seed) and the seeds are averaged.

Usage:
    python -m gar.tune --space configs/optuna/transformer.yaml --name optuna_transformer
    python -m gar.tune --space configs/optuna/b8.yaml --name optuna_b8 --trials 30 --seeds 5

Writes results/<name>/: study.db (resumable), trials.csv, best.yaml, importance.json, seeds.json,
and best_s<k>/ for every seed.
"""

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import optuna
import yaml

from gar.data.annotations import load
from gar.ensemble import load_split
from gar.train import fit
from gar.utils.io import load_config


def set_path(cfg, dotted, value):
    *parents, leaf = dotted.split(".")
    node = cfg
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def suggest(trial, name, spec):
    kind = spec["type"]
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"{name}: unknown type {kind}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--space", required=True)
    parser.add_argument("--name", required=True, help="results/<name>")
    parser.add_argument("--trials", type=int, help="override the space file")
    parser.add_argument("--seeds", type=int, default=5, help="retrain the best config with this many seeds")
    args = parser.parse_args()

    space_path = Path(args.space)
    space = yaml.safe_load(space_path.read_text(encoding="utf-8"))
    base = load_config(space_path.parent / space["base"])
    out = Path(base["paths"]["results"]) / args.name
    out.mkdir(parents=True, exist_ok=True)
    records = load(base["paths"]["annotations"])

    def objective(trial):
        cfg = copy.deepcopy(base)
        for key, spec in space["space"].items():
            set_path(cfg, key, suggest(trial, key, spec))
        cfg["train"]["epochs"] = space.get("epochs", cfg["train"]["epochs"])

        def on_epoch(epoch, score):
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial_dir = out / f"trial_{trial.number:03d}"
        try:
            return fit(cfg, trial_dir, test=False, records=records, on_epoch=on_epoch)
        finally:  # keep the log and val metrics, drop the checkpoints
            for ckpt in trial_dir.glob("*.pt"):
                ckpt.unlink()

    study = optuna.create_study(
        study_name=args.name,
        direction="maximize",
        storage=f"sqlite:///{(out / 'study.db').as_posix()}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    done = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for t in study.trials)
    remaining = (args.trials or space.get("trials", 30)) - done
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    print(f"best val {study.best_value:.4f} with {study.best_params}", flush=True)
    try:
        importance = optuna.importance.get_param_importances(study)
        (out / "importance.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")
    except Exception as e:  # needs enough completed trials
        print(f"no importance: {e}")

    best = copy.deepcopy(base)
    for key, value in study.best_params.items():
        set_path(best, key, value)
    (out / "best.yaml").write_text(yaml.safe_dump(best, sort_keys=False), encoding="utf-8")

    # retrain the best config with several seeds, then average their probabilities
    runs = []
    for seed in range(args.seeds):
        cfg = copy.deepcopy(best)
        cfg["train"]["seed"] = seed
        run_dir = out / f"best_s{seed}"
        if not (run_dir / "logits_test.npz").exists():
            shutil.rmtree(run_dir, ignore_errors=True)
            fit(cfg, run_dir, records=records)
        runs.append(f"{args.name}/best_s{seed}")
    probs, labels = load_split(base["paths"]["results"], runs, "test")
    single = [float((p.argmax(1) == labels).mean()) for p in probs]
    summary = {
        "best_val": study.best_value,
        "best_params": study.best_params,
        "trials": len(study.trials),
        "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "seed_test_accuracy": single,
        "seed_test_mean": float(np.mean(single)),
        "seed_test_std": float(np.std(single)),
        "seed_ensemble_test_accuracy": float((probs.mean(0).argmax(1) == labels).mean()),
    }
    (out / "seeds.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
