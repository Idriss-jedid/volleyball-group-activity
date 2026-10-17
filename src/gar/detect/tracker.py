"""Detector-agnostic IoU tracker over the 9 frames of a clip.

Players move ~4 px per frame against ~54 px wide boxes (docs/ANALYSIS.md §7), so frame-to-frame IoU
is high and a greedy chain works. As in ByteTrack, confident boxes are associated first and the
low-score ones are only used to extend existing tracks (occluded players get low scores).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None]
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = w * h
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def track(frames, high=0.4, min_iou=0.3):
    """frames: per frame [(box, score)] -> {track_id: {frame_index: (box, score)}}."""
    tracks, next_id = {}, 0
    for t, dets in enumerate(frames):
        unmatched = list(range(len(dets)))
        alive = [tid for tid, tr in tracks.items() if t - max(tr) <= 2]  # lost for at most 2 frames
        for stage in ("high", "low"):
            cand = [i for i in unmatched if (dets[i][1] >= high) == (stage == "high")]
            if not cand or not alive:
                continue
            ious = iou_matrix([tracks[tid][max(tracks[tid])][0] for tid in alive], [dets[i][0] for i in cand])
            rows, cols = linear_sum_assignment(-ious)
            used_tracks = set()
            for r, c in zip(rows, cols):
                if ious[r, c] >= min_iou:
                    tracks[alive[r]][t] = dets[cand[c]]
                    unmatched.remove(cand[c])
                    used_tracks.add(alive[r])
            alive = [tid for tid in alive if tid not in used_tracks]
        for i in unmatched:
            if dets[i][1] >= high:  # only confident boxes start a new track
                tracks[next_id] = {t: dets[i]}
                next_id += 1
    return tracks
