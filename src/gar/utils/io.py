"""Load yaml configs (with `defaults:` inheritance) and save results."""

import json
import os
from pathlib import Path

import yaml


def _merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("defaults", None)
    if parent:
        cfg = _merge(_load(path.parent / parent), cfg)
    return cfg


def load_config(path):
    """Read a yaml config; `defaults: base.yaml` is resolved relative to the file and merged under it.

    If the GAR_OVERLAY environment variable names a yaml file, it is merged on top of everything:
    that is how the same configs run on Kaggle (other paths, AMP on) without being edited.
    """
    cfg = _load(path)
    overlay = os.environ.get("GAR_OVERLAY")
    if overlay:
        cfg = _merge(cfg, yaml.safe_load(Path(overlay).read_text(encoding="utf-8")) or {})
    return cfg


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
