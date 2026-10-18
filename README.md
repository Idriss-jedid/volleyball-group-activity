<h1 align="center">Group Activity Recognition in Volleyball</h1>
<h3 align="center">A CVPR 2016 hierarchical model reproduced, modernized and extended · 93.9% test accuracy</h3>

<p align="center">
  <b><a href="https://idriss-jedid.github.io/volleyball-group-activity/">Project page</a></b>
  &nbsp;·&nbsp; <a href="https://idriss-jedid.github.io/volleyball-group-activity/demo.mp4">Narrated demo</a>
  &nbsp;·&nbsp; <a href="docs/RESULTS.md">All results</a>
  &nbsp;·&nbsp; <a href="https://arxiv.org/abs/1607.02643">Original paper</a>
  &nbsp;·&nbsp; <a href="https://github.com/mostafa-saad/deep-activity-rec">Volleyball dataset</a>
</p>

<p align="center">
  <video src="https://idriss-jedid.github.io/volleyball-group-activity/linkedin.mp4" controls autoplay muted loop width="860"></video>
</p>

## Abstract

Group activity recognition asks what a team is doing rather than what each person does. In a
volleyball rally, the answer (a set, a spike, a pass or a won point, by the left or the right team)
depends on what each of the twelve players does and on which side of the net they are.

This work reimplements the hierarchical deep temporal model of Ibrahim et al. (CVPR 2016), in which a
first LSTM describes each player over time and a second LSTM recognizes the team activity from the
pooled players, in three versions. **V1** reproduces the paper faithfully, with AlexNet features and
the original protocol: the full model reaches 85.9% test accuracy against the 81.9% reported, and
every conclusion of the paper's ablation holds. **V2** is the modernized version of the course built
by the paper's author, with a ResNet50 backbone: the two-stage model with one representation per team
reaches 91.8%. A preliminary analysis of the labels explains why team pooling matters: the player
actions predict the kind of activity with 94% accuracy, but the side only becomes recoverable, at
89%, once the two teams are pooled separately.

**V3** extends the model with an actor transformer that attends over every player at every frame and
an end-to-end network whose ConvNeXt-T backbone is trained with both the team loss and the
player-action loss (92.7%). Optuna hyper-parameter search, mirror test-time augmentation and seed and
model ensembles bring the final accuracy to **93.9%** (macro-F1 94.3) on the 1,337 test clips. To
remove the need for ground-truth boxes, nine COCO detectors (YOLO11, YOLOv8, RT-DETR, Faster R-CNN
and RetinaNet) and a YOLO11 detector fine-tuned on the players replace the annotated tracks: with the
fine-tuned detector, the end-to-end model reaches 92.3% from raw frames, 0.4 points below the
annotated tracks.

**Keywords:** group activity recognition · volleyball · hierarchical LSTM · actor transformer ·
end-to-end learning · ConvNeXt · ResNet50 · YOLO11 · Faster R-CNN · tracking · Optuna ·
test-time augmentation · ensembles · PyTorch

## Results (test split, 1,337 clips)

| Model | Test accuracy | Paper (AlexNet) |
|---|---:|---:|
| V1 · paper reproduction, full model (AlexNet) | 85.9 | 81.9 |
| B1 · image classifier (ResNet50) | 80.0 | 66.7 |
| B7 · two-stage LSTM, one pool over all players | 78.3 | 70.3 |
| **B8 · two-stage LSTM, pooled per team** (the paper's full model) | **91.8** | **81.9** |
| End-to-end ConvNeXt-T + transformer (group + person losses) | 92.7 | – |
| Actor transformer, ConvNeXt-T features, Optuna-tuned, 5 seeds | 92.4 ± 0.4 | – |
| End-to-end ConvNeXt-T, 3 seeds + mirror TTA (equal weights) | 93.8 | – |
| **Greedy ensemble** (models and weights chosen on val) | **93.9** | – |
| YOLO11s fine-tuned on players + end-to-end ConvNeXt-T (no ground-truth boxes) | 92.3 | – |

Full tables, ablations and analysis: [`docs/RESULTS.md`](docs/RESULTS.md).

## Three versions

| | What | Configs |
|---|---|---|
| **V1** | Faithful reproduction of the paper, to validate the implementation against its numbers: AlexNet fc7, 224×224, no augmentation, SGD momentum 0.9 with a fixed learning rate, person LSTM 3000 / scene LSTM 2000, stage-wise training. Paper numbering (B1–B7 + full model, Tables 5–6). | `configs/v1/` |
| **V2** | The modernized version of the course by the paper's author: ResNet50, full 16:9 frames, baselines B1 and B3–B8, team-wise pooling, plus the paper's ablations (2 vs 4 groups, max vs avg pooling). | `configs/b*.yaml`, `person*.yaml` |
| **V3** | My extensions: an actor transformer over players × frames, an end-to-end model (CNN + transformer trained with group and person losses), ConvNeXt-T backbones, Optuna hyper-parameter search, test-time augmentation, ensembles, and 9 detectors (YOLO11, YOLOv8, RT-DETR, Faster R-CNN, RetinaNet, plus YOLO11 fine-tuned on players) replacing the ground-truth tracks. | `configs/transformer*.yaml`, `e2e*.yaml`, `*_cnx.yaml`, `configs/optuna/`, `gar.detect` |

## What the data says

Before modelling, [`docs/DATASET.md`](docs/DATASET.md) and [`docs/ANALYSIS.md`](docs/ANALYSIS.md)
document the dataset and what it implies. The key finding drives the whole project: with ground-truth
person actions, the activity *kind* (set/spike/pass/winpoint) is 94% predictable, but the *side*
(left/right) is chance unless the representation keeps the two teams apart. Pooling all players
together caps the label information near 50%; pooling per team lifts it to ~86%. The experiments
confirm it: B7 → B8 is +13.5 points.

## How it is built

```
clip (9 tracked frames) ──► player crops ──► CNN (ResNet50 / ConvNeXt-T / AlexNet)
                                                │
                     ┌──────────────────────────┴───────────────────────────┐
           V1/V2: person LSTM → pool per team → scene LSTM      V3: actor transformer over 12 × T tokens
                     └──────────────────────────┬───────────────────────────┘
                                                ▼
                                       8 team activities
```

- **Data** (`src/gar/data`): parses `annotations.txt` (which documents its own format wrongly) and
  the tracked boxes, builds a crop cache once (518k crops, 3.4 GB) so training never decodes full
  frames again, and reads frames from the dataset zip or folder.
- **Training** (`src/gar/train.py`): one loop for every model, driven by YAML configs with
  inheritance; selection on val, test evaluated once with the best checkpoint; logits saved for
  ensembling.
- **Features** (`src/gar/features`): CNN features cached as fp16 memmaps, so the LSTM / transformer
  models train in minutes.
- **Search** (`src/gar/tune.py`): Optuna (TPE + median pruning) on val only, then the best config
  retrained with 5 seeds.
- **Fusion** (`src/gar/ensemble.py`): weighted or greedy (Caruana) selection of models on val.
- **Detection** (`src/gar/detect`): any detector → IoU tracker → player selector (a classifier on
  track geometry that drops referees and the crowd) → the same models, unchanged.

## Running it

```bash
pip install -e .                    # + requirements.txt
python -m gar.data.annotations      # dataset zip + course annot_all.pkl -> data/processed/annotations.pkl
python -m gar.data.crops            # player crop cache
python -m gar.train --config configs/b1.yaml
python -m gar.train --config configs/person.yaml
python -m gar.features.extractor --checkpoint results/person/best.pt --name person_ft
python -m gar.train --config configs/person_lstm.yaml
python -m gar.train --config configs/b8.yaml
python -m gar.tune --space configs/optuna/transformer.yaml --name optuna_transformer
python -m gar.ensemble --greedy
python -m gar.detect.pipeline --detector yolo11m.pt
pytest -q && ruff check .
```

Every run writes `results/<name>/`: config, log, checkpoint, `metrics_{val,test}.json`,
`logits_{val,test}.npz`, confusion matrices.

All reported numbers come from Kaggle T4 runs: `kaggle/run.py` is the entry point, and
`scripts/kaggle_push.py` uploads the code and launches it. The public Volleyball dataset on Kaggle
provides the frames, so nothing large is uploaded. `GAR_OVERLAY` (a YAML merged over every config)
switches paths and mixed precision without editing the configs. `scripts/smoke_*.sh` run every step
on 3 videos locally before a Kaggle run.

## Repository layout

```
configs/          one YAML per experiment (v1/, optuna/, overlays/)
docs/             DATASET.md, ANALYSIS.md, RESULTS.md, PLAN.md, figures/
kaggle/run.py     Kaggle entry point
scripts/          Kaggle push/queue, smoke tests
src/gar/          data/, features/, models/, detect/, train.py, evaluate.py, ensemble.py, tune.py
tests/            data and model invariants
```

## References

- M. S. Ibrahim, S. Muralidharan, Z. Deng, A. Vahdat, G. Mori. *A Hierarchical Deep Temporal Model
  for Group Activity Recognition.* CVPR 2016. Extended: arXiv:1607.02643.
- Dataset and original Caffe code: [github.com/mostafa-saad/deep-activity-rec](https://github.com/mostafa-saad/deep-activity-rec).
- Project brief from Dr. Mostafa S. Ibrahim's machine-learning course (baselines B1–B8).
