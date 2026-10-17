"""B3-A: person action classifier — an ImageNet backbone fine-tuned on player crops -> 9 actions.

Reused by every person-level baseline: B3 pools its features, B5-B8 feed them to LSTMs.
"""

from gar.data.datasets import PersonCropDataset
from gar.data.splits import PERSON_ACTIONS
from gar.models.backbones import build_backbone


def build_model(cfg):
    a = cfg["model_args"]
    return build_backbone(a.get("backbone", "resnet50"), len(PERSON_ACTIONS), a["dropout"])


def build_datasets(cfg, records):
    args = {
        "crops_dir": cfg["paths"]["crops"],
        "size": cfg["data"]["crop_size"],
        "augment": cfg["data"].get("augment", True),
    }
    return {
        split: PersonCropDataset([r for r in records if r["split"] == split], train=(split == "train"), **args)
        for split in ("train", "val", "test")
    }


CLASS_NAMES = PERSON_ACTIONS
