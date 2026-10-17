"""Read raw frames from the dataset zip (local) or from a folder that contains videos/ (Kaggle)."""

import zipfile
from pathlib import Path


class FrameSource:
    """`root` is D:/volleyball.zip, or e.g. /kaggle/input/group-activity-recognition-volleyball.

    The zip handle is opened lazily: zipfile objects cannot be shared across DataLoader workers.
    """

    def __init__(self, root):
        self.root = str(root)
        self._zip = None

    def __getstate__(self):
        return {"root": self.root, "_zip": None}

    def read(self, name):
        """Bytes of `videos/...` (or any path relative to the dataset root)."""
        if self.root.endswith(".zip"):
            if self._zip is None:
                self._zip = zipfile.ZipFile(self.root)
            return self._zip.read(name)
        return (Path(self.root) / name).read_bytes()

    def frame(self, video, clip, frame):
        return self.read(f"videos/{video}/{clip}/{frame}.jpg")
