# Results

All numbers are on the official **test** split (16 videos, 1,337 clips), with the model selected on
val. Trained on a Kaggle T4 (kernel `gar-pipeline`), ResNet50 backbone, PyTorch. Paper numbers are
from the extension paper (arXiv 1607.02643, Table 5–6), AlexNet.

## Baselines (course numbering)

| Run | What it is | Test acc | Test macro-F1 | Paper |
|---|---|---:|---:|---:|
| B1 | fine-tuned image classifier, middle frame | 80.0 | 81.3 | 66.7 |
| B3 | person classifier → max-pool all players → MLP | 75.1 | 71.9 | 68.1 |
| B4 | B1 image features over 9 frames → LSTM | 81.9 | 83.4 | 63.1 |
| B5 | person LSTM → pool all players → MLP | 73.9 | 69.8 | – |
| B6 | pool all players per frame → scene LSTM | 77.0 | 73.2 | – |
| B7 | person LSTM → pool all players → scene LSTM | 78.3 | 72.4 | 70.3 (1 group) |
| **B8** | **same, pooled per team (2 groups)** | **91.8** | **92.0** | 81.9 |

Person-level models (9 actions):

| Run | Test acc | Test macro-F1 |
|---|---:|---:|
| person (B3-A, one frame) | 80.4 | 66.8 |
| person LSTM (9 frames) | 82.2 | 70.1 |

## What the baselines show

- **Team-wise pooling is the decisive component: B7 → B8 is +13.5 points** (78.3 → 91.8), the same
  architecture with one `concat` of two pools instead of one pool. `docs/ANALYSIS.md` §2 predicted it
  from the labels alone: with ground-truth actions, a global pool caps the 8-class accuracy near 50%
  because it cannot tell left from right, while a per-team pool lifts it to ~86%.
- All one-group models (B3, B5, B6, B7) sit between 74% and 78%, below the plain image model B1
  (80.0%): pooling every player together throws away exactly what an image model keeps, the side.
- **B4 > B1 here (81.9 vs 80.0), the opposite of the paper (63.1 vs 66.7).** The paper explains its
  inversion by camera motion corrupting whole-frame features. With a 9-frame window that stays inside
  one rally phase (`ANALYSIS.md` §5) and features from the fine-tuned B1, temporal context helps.
- Temporal modelling helps person actions too: the person LSTM gains +3.3 macro-F1 over the
  single-frame classifier (70.1 vs 66.8), mostly on the rare actions.
- Every baseline beats its paper counterpart by 7–19 points; the backbone (ResNet50 vs AlexNet), the
  full 16:9 frame instead of a centre crop, and the label-swapping horizontal flip all contribute.

## V1 — faithful reproduction of the paper (kernels `gar-v1`, `gar-v1b`)

AlexNet fc7 (4096-d), 224x224 inputs, no augmentation, SGD with momentum 0.9 and a fixed learning
rate, person LSTM 3000 nodes, scene FC 3000 + LSTM 2000 nodes, stage-wise training. Paper numbering.

| Paper baseline | Test acc | Test macro-F1 | Paper |
|---|---:|---:|---:|
| B1 — image classification | 63.9 | 63.9 | 66.7 |
| B2 — person classification (ImageNet fc7) | 47.9 | 45.0 | 64.6 |
| B3 — fine-tuned person classification | 68.1 | 62.2 | 68.1 |
| B4 — temporal model, image features | 66.0 | 65.9 | 63.1 |
| B5 — temporal model, person features (ImageNet fc7) | 51.1 | 48.2 | 67.6 |
| B6 — two-stage without LSTM 1 (2 groups) | 85.4 | 86.2 | 74.7 |
| B7 — two-stage without LSTM 2 (2 groups) | 81.9 | 82.7 | 80.2 |
| **Full two-stage model (2 groups, max)** | **85.9** | **86.5** | **81.9** |

Table 6 of the paper (pooling variants of the full model):

| Full model | Test acc | Paper |
|---|---:|---:|
| 1 group, max pool | 74.2 | 70.3 |
| 2 groups, max pool | 85.9 | 81.9 |
| 2 groups, avg pool | 85.2 | 80.7 |
| 4 groups, max pool | 86.2 | 81.5 |

- **The reproduction validates the implementation.** B1, B3 (exactly) and B7 land within 3 points of
  the paper, and every conclusion of the paper reproduces: the two-stage model beats all baselines,
  team pooling is decisive (1 group 74.2 → 2 groups 85.9, paper 70.3 → 81.9), max beats average
  pooling, and 4 sub-groups bring nothing over 2 teams.
- The models on fine-tuned features come out ~4 points *above* the paper (full model 85.9 vs 81.9).
  Likely reasons: training details that cannot be matched exactly (PyTorch instead of Caffe, early
  stopping on val, a learning rate of 1e-3/1e-2 instead of 1e-5) and 9 tracked frames instead of 10.
- **The one gap is the frozen-ImageNet baselines B2 and B5** (47.9 / 51.1 vs 64.6 / 67.6). With raw
  ImageNet fc7 a 1e-2 learning rate diverged (first-epoch loss ~100); 1e-3 trains but stays far below
  the paper. The paper does not detail how these features were normalized or trained, so this part
  is not reproduced. It does not affect the paper's main claims, which rely on fine-tuned features.

## V2 — ablations (kernel `gar-ablation`)

| Run | Change | Test acc | Test macro-F1 | Paper |
|---|---|---:|---:|---:|
| B3, 2 groups | B3 with team-wise pooling, no temporal model | 90.1 | 90.1 | – |
| B5, 2 groups | paper B7 (no scene LSTM) | 90.3 | 90.2 | 80.2 |
| B6, 2 groups | paper B6 (no person LSTM) | 91.2 | 91.5 | 74.7 |
| B8 | full model, 2 groups, max pool | **91.8** | **92.0** | 81.9 |
| B8, avg pool | Table 6 | 90.7 | 90.8 | 80.7 |
| B8, 4 groups | Table 6 | 90.1 | 90.4 | 81.5 |

- **Team-wise pooling explains almost everything.** With 2 groups, even B3 — one frame, no LSTM at
  all — reaches 90.1% (+15.0 over its 1-group version). B5, B6, B8 with 2 groups all land in
  90–92%; the LSTMs add at most ~1.7 points on top.
- Both Table 6 findings reproduce: average pooling is below max pooling (90.7 vs 91.8, paper 80.7 vs
  81.9), and 4 sub-groups do not beat 2 teams (90.1, paper 81.5).
- Unlike the paper, removing the person LSTM (B6, 91.2) costs less than removing the scene LSTM
  (B5, 90.3) here; the gap is small either way.

### Paper B2 / B5 in V2 (kernel `gar-v2b`)

The course leaves out the paper's B2 and B5 because both use frozen ImageNet features ("Don't try
anything that doesn't finetune"). Run with ResNet50 for completeness:

| Paper baseline, ResNet50 | Test acc | Test macro-F1 | Paper (AlexNet) |
|---|---:|---:|---:|
| B2 — frozen ImageNet player features, pooled, softmax | 47.9 | 46.7 | 64.6 |
| B5 — frozen ImageNet player features, pooled, scene LSTM | 57.8 | 53.4 | 67.6 |
| *B3 — fine-tuned player features, pooled* | *75.1* | *71.9* | *68.1* |

Fine-tuning the player CNN is worth +27 points over frozen ImageNet features here, which is the
reason the course skips these two baselines. As in V1, the frozen-feature baselines stay below the
paper's numbers.

## V3 — beyond the paper (kernel `gar-ablation`)

| Run | What | Test acc | Test macro-F1 |
|---|---|---:|---:|
| transformer | actor transformer over 12 players x 9 frames, with box positions | 90.8 | 91.0 |
| transformer_nobox | same, without box positions (rank + time only) | 91.5 | 92.0 |
| transformer_cnx | actor transformer on ConvNeXt-T player features | 91.6 | 91.8 |
| b8_cnx | B8 on ConvNeXt-T player features | 91.1 | 90.7 |
| b1_cnx | B1 with ConvNeXt-T | 81.7 | 83.0 |
| e2e | end-to-end ResNet50 + transformer, group + person losses, 4 frames | 91.5 | 91.8 |
| **e2e_cnx** | **end-to-end ConvNeXt-T + transformer** | **92.7** | **92.8** |

Late fusion (weights chosen on val, test scored once):

| Ensemble | Weights | Test acc | Test macro-F1 |
|---|---|---:|---:|
| B8 + B4 | 0.8 / 0.2 | 92.0 | 92.1 |
| B8 + e2e | 0.5 / 0.5 | 93.0 | 93.2 |
| B8 + e2e + transformer + B4 | 0.2 / 0.3 / 0.2 / 0.3 | 93.1 | 93.5 |
| b8_cnx + transformer_cnx + b1_cnx | 0.4 / 0.2 / 0.4 | 93.4 | 93.6 |
| **B8 + b8_cnx + e2e + e2e_cnx + transformer_cnx + B4** | 0.1 / 0.1 / 0.2 / 0.1 / 0.2 / 0.3 | **93.6** | **94.1** |

- **Best single model: end-to-end ConvNeXt-T, 92.7%** (+0.9 over B8). Training the CNN with the group
  loss helps more than any change of the scene model on frozen features.
- Box positions did not help the transformer (90.8 with, 91.5 without): the left-to-right slot rank
  already encodes the side, and the explicit geometry seems to add noise on 2k training clips.
- ConvNeXt-T player features are slightly worse as a person classifier (78.6 vs 80.4) but slightly
  better for the group transformer (91.6 vs 90.8) and clearly better end to end (92.7 vs 91.5).
- Ensembles add 1–2 points: whole-frame models (B4, b1_cnx) are the most useful partners of the
  player models, because they see what the player crops miss (ball, net, celebration shots).

## V3 — hyper-parameter search with Optuna (kernel `gar-optuna`)

TPE sampler with median pruning, selection on val only; the best configuration is then retrained
with 5 seeds (test evaluated once per seed) and the seeds are averaged.

| Search | Trials | Best val | Test, 5 seeds (mean ± std) | Test, seed ensemble |
|---|---:|---:|---:|---:|
| Actor transformer (ResNet50 features) | 50 | 90.1 | 91.6 ± 0.5 | 92.0 |
| **Actor transformer (ConvNeXt-T features)** | 50 | 91.2 | **92.4 ± 0.4** | **93.6** |
| B8 | 40 | 90.2 | 91.4 ± 0.2 | 91.5 |

Greedy ensemble selection (Caruana) over every trained 8-class model, weights chosen on val:
**93.7% test accuracy, 94.1 macro-F1** (val 93.8), mixing whole-frame models (B1, B1-ConvNeXt, B4),
player models (B3, B6, B8-ConvNeXt, Optuna B8 seeds) and the end-to-end and transformer models.

- The seed spread (±0.2–0.5) is the size of most differences between single models above: gaps
  under ~1 point between two single runs should not be over-interpreted.
- Averaging 5 seeds of one architecture is worth +1.2 points for the ConvNeXt transformer, as much
  as mixing different architectures.

## V3 — detected boxes instead of ground-truth tracks (kernel `gar-detect`)

Each detector (COCO-pretrained, person class, not fine-tuned) runs on the 9 frames of every clip;
the same IoU tracker and the same player selector (trained on 500 train clips, threshold tuned on
400 val clips) are used for all. Recall / precision are measured against the ground-truth players on
the target frame (IoU ≥ 0.5). The models are the ones trained on ground-truth tracks, evaluated
unchanged on the detected players of the 1,337 test clips. Speed is measured on a Kaggle T4.

| Detector | FPS | Recall | Precision | B8 | Transformer | e2e |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | 68.1 | 76.1 | 90.9 | 72.3 | 74.1 | 79.6 |
| YOLO11s | 36.8 | 80.3 | 91.9 | 77.3 | 79.4 | 81.7 |
| YOLO11m | 14.7 | 83.9 | 92.1 | 80.6 | 82.5 | 84.9 |
| **YOLO11x** | 6.0 | **84.1** | **93.2** | **81.5** | 82.7 | **85.3** |
| YOLOv8m | 16.0 | 83.4 | 91.9 | 79.7 | **82.9** | 84.2 |
| RT-DETR-L | 6.1 | 76.2 | 85.1 | 75.2 | 78.1 | 81.5 |
| Faster R-CNN R50-FPN v2 | 5.2 | 82.0 | 89.8 | 78.5 | 81.6 | 84.8 |
| Faster R-CNN MobileNetV3 | 25.1 | 71.0 | 80.4 | 72.8 | 75.0 | 76.7 |
| RetinaNet R50-FPN v2 | 6.9 | 76.2 | 88.9 | 70.6 | 72.6 | 78.3 |
| *ground-truth tracks* | – | 100 | 100 | 91.8 | 90.8 | 91.5 |

- **Best pipeline: YOLO11x + IoU tracker + end-to-end model, 85.3%**, i.e. −6.2 points against the
  ground-truth tracks. YOLO11m is nearly as good (84.9%) at 2.5x the speed; YOLO11n keeps 79.6% at
  68 FPS.
- Group accuracy follows detection recall closely: every missed player is a hole in the team
  representation. Among similar recalls, precision matters less because the selector already filters
  referees and the crowd.
- The end-to-end model is the most robust to imperfect boxes (−6.2 vs −10.3 for B8 with YOLO11x):
  its CNN was trained with the group loss and copes better with loose or shifted crops.
- The YOLO family dominates this setting; RT-DETR-L and RetinaNet, although strong on COCO, find fewer
  of the small, distant players at their default input size.

### YOLO11s fine-tuned on players (kernels `gar-yolo-s`, `gar-demo`)

YOLO11s (COCO weights) fine-tuned on one class, "player", with the tracked boxes of 3 frames per
train clip (6,456 images), 960 px, 15 epochs. Validation on 400 val clips (4,761 players): precision
97.8, recall 95.9, mAP50 98.7, mAP50-95 60.5. Then the same evaluation as the COCO detectors above:

| Detector | FPS | Recall | Precision | B8 | Transformer | e2e | e2e ConvNeXt-T |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11x (COCO, best of the nine) | 6.0 | 84.1 | 93.2 | 81.5 | 82.7 | 85.3 | – |
| **YOLO11s fine-tuned on players** | **38.7** | **96.1** | **96.9** | **90.2** | **89.9** | **89.7** | **92.3** |
| *ground-truth tracks* | – | 100 | 100 | 91.8 | 90.8 | 91.5 | 92.7 |

- **From raw frames, without any ground-truth box, the end-to-end ConvNeXt-T model reaches 92.3%**,
  0.4 points below the same model on the annotated tracks.
- Fine-tuning turns a small detector into the best one by far: recall 84 → 96%, at 6.5x the speed of
  YOLO11x. The COCO detectors box referees, line judges and the crowd and miss players in odd poses
  (a setter in the air, a defender on the floor); the fine-tuned detector learns exactly the
  annotators' notion of a player.

## V3 — test-time augmentation, seeds and final ensemble (kernel `gar-boost`)

Mirror TTA: every clip is also evaluated mirrored (crops flipped, players reversed, `l_*` ↔ `r_*`)
and the two softmax outputs are averaged. Exact for the image and end-to-end models, approximate
for the transformer on cached features. No retraining.

| Model | Test acc | Test acc with TTA |
|---|---:|---:|
| B1 (ResNet50) | 80.0 | 82.3 |
| B1 (ConvNeXt-T) | 81.7 | 82.9 |
| Actor transformer (ResNet50) | 90.8 | 91.5 |
| Actor transformer (ConvNeXt-T) | 91.6 | 92.1 |
| End-to-end ResNet50 | 91.5 | 92.3 |
| End-to-end ConvNeXt-T, 3 seeds (mean ± std) | 92.4 ± 0.3 | 93.0 ± 0.4 |

| Final models | Selection | Test acc | Test macro-F1 |
|---|---|---:|---:|
| End-to-end ConvNeXt-T, 3 seeds + TTA, equal weights | none (fixed in advance) | **93.8** | – |
| **Greedy ensemble over all models** (incl. TTA variants) | models and weights on val | **93.9** | **94.3** |

- TTA is the cheapest gain of the project: +0.5 to +2.3 points for free at inference, largest for
  the whole-frame models.
- The two final numbers agree (93.8 / 93.9): one well-trained architecture averaged over seeds and
  mirror views is as good as the hand-picked mix of 8 models.
- We did not reach 95%. Published results on this benchmark around 94–95% use stronger video
  backbones (I3D, ViT) and/or pose keypoints; with ResNet50/ConvNeXt-T player crops, 9 frames and
  2,152 training clips, ~94% is where this design saturates.
