"""Invariants of data/processed/annotations.pkl (see docs/DATASET.md). Run: pytest -q"""

from pathlib import Path

import pytest

from gar.data.annotations import load
from gar.data.crops import expand_to_aspect
from gar.data.splits import GROUP_ACTIVITIES, PERSON_ACTIONS

ANNOT = Path(__file__).parents[1] / "data/processed/annotations.pkl"


@pytest.fixture(scope="module")
def records():
    if not ANNOT.exists():
        pytest.skip("run `python -m gar.data.annotations` first")
    return load(ANNOT)


def test_counts(records):
    assert len(records) == 4830
    assert len({r["video"] for r in records}) == 55
    assert {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")} == {
        "train": 2152,
        "val": 1341,
        "test": 1337,
    }


def test_labels(records):
    assert all(r["activity"] in GROUP_ACTIVITIES for r in records)
    assert all(GROUP_ACTIVITIES[r["label"]] == r["activity"] for r in records)
    assert all(a in PERSON_ACTIONS for r in records for a in r["track_actions"].values())


def test_tracks_cover_target_minus5_to_plus3(records):
    for r in records:
        assert r["frames"] == list(range(r["clip"] - 5, r["clip"] + 4))
        assert 8 <= len(r["tracks"]) <= 12
        assert all(len(boxes) == 9 for boxes in r["tracks"].values())


def test_tracks_sorted_left_to_right_on_target(records):
    for r in records[:500]:
        t = r["frames"].index(r["clip"])
        xs = [(b[t][0] + b[t][2]) / 2 for b in r["tracks"].values()]
        assert xs == sorted(xs)


def test_expand_to_aspect():
    # a wide (falling) player gets taller, a thin one gets wider; result clipped to the frame
    x1, y1, x2, y2 = expand_to_aspect((100, 100, 200, 150), 2.0, 1280, 720)
    assert (x2 - x1, y2 - y1) == (100, 200)
    x1, y1, x2, y2 = expand_to_aspect((100, 100, 120, 200), 2.0, 1280, 720)
    assert (x2 - x1, y2 - y1) == (50, 100)
    assert expand_to_aspect((0, 0, 10, 100), 2.0, 1280, 720)[0] == 0
