"""B1-tuned: fine-tuned ResNet50 on the middle frame -> 8 group activities."""

from gar.data.datasets import FrameDataset
from gar.data.splits import GROUP_ACTIVITIES
from gar.models.backbones import build_backbone


def build_model(cfg):
    a = cfg["model_args"]
    return build_backbone(a.get("backbone", "resnet50"), len(GROUP_ACTIVITIES), a["dropout"])


def build_datasets(cfg, records):
    args = dict(
        videos=cfg["paths"]["videos"],
        offsets=(0,),
        size=cfg["data"]["frame_size"],
        augment=cfg["data"].get("augment", True),
    )
    return {
        split: FrameDataset([r for r in records if r["split"] == split], train=(split == "train"), **args)
        for split in ("train", "val", "test")
    }


CLASS_NAMES = GROUP_ACTIVITIES
