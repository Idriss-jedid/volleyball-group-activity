"""Person detectors behind one interface: detector(list of PIL images) -> per image [(box_xyxy, score)].

Specs:
    yolo11n.pt / yolo11s.pt / yolo11m.pt / yolo11x.pt / yolov8m.pt ...   ultralytics YOLO
    results/yolo11m_volleyball.pt                                       YOLO fine-tuned on players (class 0)
    rtdetr-l.pt / rtdetr-x.pt                                           ultralytics RT-DETR (transformer)
    frcnn_r50                                                           torchvision Faster R-CNN R50-FPN v2
    frcnn_mobile                                                        torchvision Faster R-CNN MobileNetV3-FPN
    retinanet_r50                                                       torchvision RetinaNet R50-FPN v2
All COCO-pretrained (person class) except the fine-tuned YOLO (gar.detect.finetune).
"""

import torch


class UltralyticsDetector:
    def __init__(self, weights, conf, imgsz, device):
        from ultralytics import RTDETR, YOLO  # optional dependency

        self.model = (RTDETR if weights.startswith("rtdetr") else YOLO)(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def __call__(self, images):
        results = self.model.predict(
            images, classes=[0], conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )
        return [[(tuple(b), s) for b, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())] for r in results]


class TorchvisionDetector:
    PERSON = 1  # COCO category id in torchvision detection models

    def __init__(self, name, conf, device):
        from torchvision.models import detection as d

        builders = {
            "frcnn_r50": (d.fasterrcnn_resnet50_fpn_v2, d.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
            "frcnn_mobile": (d.fasterrcnn_mobilenet_v3_large_fpn, d.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
            "retinanet_r50": (d.retinanet_resnet50_fpn_v2, d.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT),
        }
        build, weights = builders[name]
        self.model = build(weights=weights, box_score_thresh=conf).to(device).eval()
        self.transform = weights.transforms()
        self.device = device

    @torch.no_grad()
    def __call__(self, images):
        batch = [self.transform(img).to(self.device) for img in images]
        out = []
        for o in self.model(batch):
            keep = o["labels"] == self.PERSON
            out.append([(tuple(b), s) for b, s in zip(o["boxes"][keep].tolist(), o["scores"][keep].tolist())])
        return out


def build_detector(spec, conf=0.1, imgsz=1280, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if spec.endswith(".pt"):
        return UltralyticsDetector(spec, conf, imgsz, device)
    return TorchvisionDetector(spec, conf, device)


def detector_name(spec):
    return spec.removesuffix(".pt").replace("-", "_")
