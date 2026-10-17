# Group Activity Recognition — Volleyball

PyTorch reimplementation of Ibrahim et al., **"A Hierarchical Deep Temporal Model for Group Activity
Recognition"** (CVPR 2016; extended version arXiv:1607.02643), modernized with current backbones,
then extended with my own architectures and a detection pipeline.

The task: given a short volleyball clip, recognize the team activity (8 classes: left/right × set,
spike, pass, winpoint), using the 12 players and their actions (9 classes).

## Results (test split, 1,337 clips)

| Model | Test accuracy | Paper (AlexNet) |
|---|---:|---:|
| V1 — paper reproduction, full model (AlexNet) | 85.9 | 81.9 |
| B1 — image classifier (ResNet50) | 80.0 | 66.7 |
| B7 — two-stage LSTM, one pool over all players | 78.3 | 70.3 |
| **B8 — two-stage LSTM, pooled per team** (the paper's full model) | **91.8** | **81.9** |
| End-to-end ConvNeXt-T + transformer (group + person losses) | 92.7 | – |
| Actor transformer, ConvNeXt-T features, Optuna-tuned, 5 seeds | 92.4 ± 0.4 | – |
| **Greedy ensemble** (models and weights chosen on val) | **93.7** | – |
| YOLO11x detections + end-to-end model (no ground-truth boxes) | 85.3 | – |

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
