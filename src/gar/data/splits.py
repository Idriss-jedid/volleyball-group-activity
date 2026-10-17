"""Volleyball dataset constants, verified against D:\\volleyball.zip (see docs/DATASET.md)."""

# Official splits (videos/Info.txt) — by video, so there is no frame leakage.
TRAIN_VIDEOS = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
VAL_VIDEOS = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_VIDEOS = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

# These videos are 1920x1080; all the others are 1280x720. Normalize boxes per frame.
HD_VIDEOS = {2, 37, 38, 39, 40, 41, 44, 45}

NUM_VIDEOS = 55
NUM_CLIPS = 4830
FRAMES_PER_CLIP = 41  # target-20 .. target+20
MAX_PLAYERS = 12  # at most: 12 in 73.8% of frames, down to 7

GROUP_ACTIVITIES = ["r_set", "r_spike", "r_pass", "r_winpoint", "l_set", "l_spike", "l_pass", "l_winpoint"]

PERSON_ACTIONS = ["waiting", "setting", "digging", "falling", "spiking", "blocking", "jumping", "moving", "standing"]

ACTIVITY_TO_IDX = {a: i for i, a in enumerate(GROUP_ACTIVITIES)}
ACTION_TO_IDX = {a: i for i, a in enumerate(PERSON_ACTIONS)}

# Mirroring an image swaps the two teams, so the label has to follow.
FLIP_ACTIVITY = {a: ("l" + a[1:] if a[0] == "r" else "r" + a[1:]) for a in GROUP_ACTIVITIES}


def normalize_activity(label: str) -> str:
    """The files mix separators: 'l-pass', 'r-pass', 'l-spike' vs 'r_spike', 'l_set'."""
    return label.strip().lower().replace("-", "_")


def split_of(video_id: int) -> str:
    if video_id in TRAIN_VIDEOS:
        return "train"
    if video_id in VAL_VIDEOS:
        return "val"
    if video_id in TEST_VIDEOS:
        return "test"
    raise ValueError(f"unknown video id: {video_id}")


def frame_size(video_id: int) -> tuple[int, int]:
    return (1920, 1080) if video_id in HD_VIDEOS else (1280, 720)
