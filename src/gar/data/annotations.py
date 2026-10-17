"""Parse the Volleyball annotations into one plain-Python pickle.

Two sources are merged per clip:
  - videos/<vid>/annotations.txt inside the dataset zip: group activity + the players of the target
    frame, as `X Y W H action` (the README wrongly says `action X Y W H`).
  - annot_all.pkl (course format, from Kaggle `sherif31/group-activity-recognition-volleyball`):
    tracked boxes with player ids over 9 frames, target-5 .. target+3, as `x1 y1 x2 y2`.

Output records contain only builtins (no custom classes), so loading them needs no special code:

    {"video": int, "clip": int, "split": str, "activity": str, "label": int,
     "width": int, "height": int,
     "players": [(x1, y1, x2, y2, action), ...],            # target frame, sorted by box centre x
     "frames": [frame_id, ...],                             # the 9 tracked frames, ascending
     "tracks": {player_id: [(x1, y1, x2, y2), ...]},        # one box per frame in `frames`
     "track_actions": {player_id: action}}

Usage:
    python -m gar.data.annotations --zip D:/volleyball.zip \
        --tracks data/raw/kaggle_annot/annot_all.pkl --out data/processed/annotations.pkl
"""

import argparse
import collections
import pickle
import zipfile
from pathlib import Path

from gar.data.splits import ACTIVITY_TO_IDX, NUM_VIDEOS, frame_size, normalize_activity, split_of


def read_txt_annotations(zip_path):
    """Return {(video, clip): (activity, players)} from every annotations.txt in the zip."""
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for vid in range(NUM_VIDEOS):
            for line in z.read(f"videos/{vid}/annotations.txt").decode("utf-8").splitlines():
                tokens = line.split()
                if not tokens:
                    continue
                frame, activity, rest = tokens[0], tokens[1], tokens[2:]
                if len(rest) % 5:
                    raise ValueError(f"video {vid}: malformed line for {frame}")
                players = []
                for i in range(0, len(rest), 5):
                    x, y, w, h = map(int, rest[i : i + 4])
                    players.append((x, y, x + w, y + h, rest[i + 4]))
                out[(vid, int(frame.removesuffix(".jpg")))] = (normalize_activity(activity), players)
    return out


class _BoxInfo:
    """Stand-in for the course's boxinfo.BoxInfo, so the pickle loads without importing it."""


class _SafeUnpickler(pickle.Unpickler):
    # annot_all.pkl is a downloaded file: only allow the one class it is supposed to contain.
    def find_class(self, module, name):
        if (module, name) in {("boxinfo", "BoxInfo"), ("__main__", "BoxInfo")}:
            return _BoxInfo
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def read_tracks(pkl_path):
    """Return {(video, clip): (frames, tracks, track_actions)} from the course's annot_all.pkl."""
    with open(pkl_path, "rb") as f:
        raw = _SafeUnpickler(f).load()

    out = {}
    for vid, clips in raw.items():
        for clip, entry in clips.items():
            frames = sorted(entry["frame_boxes_dct"])
            tracks = collections.defaultdict(dict)
            actions = {}
            for frame in frames:
                for b in entry["frame_boxes_dct"][frame]:
                    tracks[b.player_ID][frame] = tuple(b.box)
                    actions[b.player_ID] = b.category
            # keep only players present in every frame, so each track is a full sequence
            full = {pid: [boxes[f] for f in frames] for pid, boxes in tracks.items() if len(boxes) == len(frames)}
            out[(int(vid), int(clip))] = (frames, full, {pid: actions[pid] for pid in full})
    return out


def _centre_x(box):
    return (box[0] + box[2]) / 2


def build(zip_path, tracks_path):
    txt = read_txt_annotations(zip_path)
    tracks = read_tracks(tracks_path)
    if txt.keys() != tracks.keys():
        raise ValueError(
            f"clip sets differ: {len(txt.keys() - tracks.keys())} only in txt, "
            f"{len(tracks.keys() - txt.keys())} only in tracks"
        )

    records = []
    for vid, clip in sorted(txt):
        activity, players = txt[(vid, clip)]
        frames, clip_tracks, track_actions = tracks[(vid, clip)]
        width, height = frame_size(vid)
        # tracks ordered left to right on the target frame: the order B8 splits into teams
        target_idx = frames.index(clip)
        order = sorted(clip_tracks, key=lambda pid: _centre_x(clip_tracks[pid][target_idx]))
        records.append(
            {
                "video": vid,
                "clip": clip,
                "split": split_of(vid),
                "activity": activity,
                "label": ACTIVITY_TO_IDX[activity],
                "width": width,
                "height": height,
                "players": sorted(players, key=_centre_x),
                "frames": frames,
                "tracks": {pid: clip_tracks[pid] for pid in order},
                "track_actions": {pid: track_actions[pid] for pid in order},
            }
        )
    return records


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--zip", default="D:/volleyball.zip")
    parser.add_argument("--tracks", default="data/raw/kaggle_annot/annot_all.pkl")
    parser.add_argument("--out", default="data/processed/annotations.pkl")
    args = parser.parse_args()

    records = build(args.zip, args.tracks)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(records, f)

    per_split = collections.Counter(r["split"] for r in records)
    print(f"{len(records)} clips -> {args.out}  {dict(per_split)}")


if __name__ == "__main__":
    main()


def select(records, cfg):
    """Apply `data.only_videos` (a debugging subset) from the config, if set."""
    only = cfg.get("data", {}).get("only_videos")
    return [r for r in records if r["video"] in set(only)] if only else records
