# Implementation plan

Order, deliverables and compute strategy. Numbers to beat come from the extension paper
(arXiv 1607.02643, Tables 5–8), measured with AlexNet — we use ResNet50, so we should beat them.

## Baseline mapping (course slides ↔ paper)

The course numbering is not the paper's. This is the mapping, with the paper's Volleyball accuracy:

| Course | What it is | Paper equivalent | Paper acc |
|---|---|---|---:|
| B1-tuned | fine-tuned image classifier, middle frame | B1 | **66.7** |
| B3 | person action classifier → pooled crop features → MLP | B3 | **68.1** |
| B4 | image features → LSTM | B4 | **63.1** |
| B5 | LSTM per player → pool → MLP (no scene LSTM) | B7 (2-group) | **80.2** |
| B6 | pooled person features per frame → scene LSTM | B6 (2-group) | **74.7** |
| B7 | full model, one pool over all players | full model, 1 group | **70.3** |
| B8 | full model, pool per team, concat | full model, 2 groups | **81.9** |

Other useful rows from the paper: 4 groups = 81.5 (no better than 2), avg pool = 80.7 vs max 81.9,
1 group avg = 68.5. Best hyper-parameters: person LSTM 3000 nodes / 5 timesteps, scene LSTM
2000 nodes / 10 timesteps.

Note `B4 (63.1) < B1 (66.7)`: a temporal model over whole frames is **worse** than a single frame,
because the camera moves. Reproducing that inversion is a good sanity check that our pipeline is
honest.

## Compute

Local machine: **GTX 1650, 4 GB VRAM**, D: has 414 GB free, **C: has only 3.4 GB free**.

### Fix the disk first

Everything (venv, pip cache, torch weights, dataset, features) goes on D:

```powershell
setx PIP_CACHE_DIR      D:\cache\pip
setx TORCH_HOME         D:\cache\torch
setx HF_HOME            D:\cache\hf
```

### Measured on the GTX 1650 (conda env `gar`, torch 2.8.0+cu126)

| | fp32 | fp16 (autocast) |
|---|---:|---:|
| ResNet50 inference, bs 64 | **250 img/s** | 66 img/s |
| ResNet50 training, bs 16 | **72 img/s**, 2.75 GB | 21 img/s, 1.59 GB |
| ResNet50 training, bs 32 | **74 img/s**, 3.22 GB | – |

⚠️ **Do not use AMP / fp16 on this card.** The GTX 1650 (TU117) has no tensor cores, so autocast
is ~4× *slower*. Train in fp32; bs 32 fits. (On Colab T4 / Kaggle, AMP is the right choice.)
Still store cached features in fp16 — that is storage, not compute.

Resulting estimates:

- B1: 2,152 train images → ~30 s of GPU per epoch.
- Feature extraction, 521,640 crops → ~35 min of GPU (plus JPEG decoding, parallelized).
- B3-A on target-frame crops (~25k train crops) → ~6 min per epoch: local is fine.
- B3-A on all 9 frames (~225k train crops) → ~50 min per epoch: Colab / Kaggle.

### What runs where

| Job | Cost | Where |
|---|---|---|
| Parse annotations, statistics, visualisation | CPU, minutes | local |
| **Feature extraction** (frozen ResNet50, 520k crops) | ~1–2 h GPU | **local (1650)** |
| B1 fine-tune (4,830 images) | small | local, bs 16 + AMP |
| B3-A person classifier (58k crops, target frame only) | medium | local overnight, or Colab |
| B3-A on all 9 frames (520k crops) | heavy | Colab / Kaggle |
| All LSTM baselines **on cached features** | tiny | local (even CPU) |
| End-to-end challenge (12 players × 9 frames = 108 crops/sample) | needs ≥12 GB | **Colab T4 / Kaggle P100** |

The key move: **the 1650 is enough for everything except end-to-end fine-tuning**, because once the
features are cached, B4–B8 are just small LSTMs over 2048-d vectors.

### Getting data to Colab/Kaggle

Do **not** upload 62 GB and do **not** read thousands of small files from a mounted Drive — it is
extremely slow. Instead:

1. Extract features locally → `features/` is **2.1 GB in fp16** for all crops of 9 frames.
2. Upload that one archive to Drive (or as a Kaggle dataset).
3. On Colab: copy the archive to `/content` **then** unzip. Never train off `/content/drive`.

If a job really needs images (B1 fine-tune, end-to-end), build a slim package first: only the 9
frames per clip that we use, re-encoded at a smaller size — a few GB instead of 62.

Also check whether the Volleyball dataset is already hosted as a public Kaggle dataset; if it is,
Kaggle attaches it with zero upload.

### Colab / Kaggle practicalities

- Colab free: T4 16 GB, sessions die after ~12 h or on idle. **Checkpoint every epoch to Drive**,
  and make training resumable, otherwise a disconnect costs the whole run.
- Kaggle: ~30 GPU-hours/week, P100 or 2×T4, 9 h per session, no random disconnects — more reliable
  than free Colab for long runs.
- Always `torch.cuda.amp` (mixed precision): ~2× speed and half the memory on both.

### "Can I use Colab's GPU from my local PC?"

No. The only official bridge is the opposite direction: **Colab local runtime**
(`jupyter_http_over_ws`) lets the Colab notebook UI execute on *your* machine and *your* GPU — it
does not give you Google's GPU locally.

What actually works for "code locally, compute remotely":

| Option | How | Notes |
|---|---|---|
| **VS Code Remote-SSH** to a rented GPU box (Lightning AI, RunPod, vast.ai, Paperspace) | local VS Code, remote GPU | closest to what you want; free tiers exist, verify current limits |
| **Kaggle API** (`kaggle kernels push`) | write code locally, push, it runs on Kaggle | good for batch jobs, no interactivity |
| **Colab local runtime** | `pip install jupyter_http_over_ws`, connect | uses YOUR 1650, Colab is only the editor |
| Sync + run | `rsync`/git push, run on the remote | simplest, always works |

Recommendation: local 1650 for everything up to B6, and one Kaggle/Colab notebook for the two heavy
jobs (person classifier on all frames, end-to-end model).

## The helper scripts

`volleyball_annot_loader.py`, `boxinfo.py`, `extract_features.py` are the instructor's. Read them,
then fix these before use:

1. **They need `volleyball_tracking_annotation`, which is not in our 62 GB zip.** Download it from
   `github.com/mostafa-saad/deep-activity-rec` (it is small), or generate tracks ourselves
   (see `ANALYSIS.md` §7 — a greedy IoU tracker over 9 frames is realistic).
2. **Two different box formats.** `annotations.txt` is `X Y W H`, the tracking files are
   `x1 y1 x2 y2` plus `player_ID … lost grouping generated category`. Do not mix them.
3. `load_tracking_annot` keeps 9 frames with `boxes_info[5:][:-6]`. Verify what the 20-ish frames per
   clip actually span before trusting that window; print the frame ids once.
4. `extract_features.py` has `if True: cv2.imshow(...)` inside the crop loop — it blocks on every
   crop. Remove it.
5. `dnn_repr.numpy()` fails on GPU tensors → `.cpu().numpy()`, and save in fp16.
6. The `np.save` line is commented out, so the script currently computes and throws away.
7. `models.resnet50(pretrained=True)` is deprecated → `weights=ResNet50_Weights.IMAGENET1K_V1`.
8. It forwards 12 crops at a time. Batch across frames (e.g. 128 crops) to actually use the GPU.
9. `output_root` says `image-level` while `image_level=False`. Keep feature paths honest, we will
   have several variants.

## Phases

### Phase 1 — data (done)

Parser, statistics, visualisation, observations. See `DATASET.md` and `ANALYSIS.md`.
Remaining: port the throwaway scripts into `src/gar/data/annotations.py`, and get tracks.

### Phase 2 — infrastructure

- `datasets.py`: three modes (whole frame / crops of one frame / clip of T frames × K players).
- Deterministic seeding, video-level splits from `splits.py`.
- Train loop + eval producing accuracy, macro-F1, confusion matrix, saved under `results/<exp>/`.
- Feature extraction + fp16 cache.
- **Deliverable**: `python -m gar.train --config configs/b1.yaml` runs end to end.

### Phase 3 — baselines, in this order

Each step reuses the previous one. Stop and write down what you learned before moving on.

| # | Step | Needs | Target |
|---|---|---|---|
| 1 | **B1-tuned** — ResNet50 fine-tuned on the middle frame, 8 classes | target frames only | > 66.7 |
| 2 | **B3-A** — ResNet50 fine-tuned on crops, 9 actions | target frame boxes | report macro-F1, not accuracy |
| 3 | **B3-B/C** — cache crop features, max-pool per frame, MLP → 8 classes | B3-A | > 68.1 |
| 4 | **B4** — B1 features over 9 frames → LSTM | 9 frames/clip | expect **< B1** (paper: 63.1) |
| 5 | **B6** — pooled person features per frame → scene LSTM | B3-A + tracks | > 74.7 |
| 6 | **B5** — player LSTM → last hidden → pool → MLP | B3-A + tracks | > 80.2 |
| 7 | **B7** — player LSTM → pool all → scene LSTM | B5 | ~70 if one group |
| 8 | **B8** — same, pooled per team, concat | B7 | > 81.9, the best model |

Steps 1–3 need **no tracking annotations** — only the target frame. That is why they come first.

### Phase 4 — ablation and analysis

- Results table for all 8 baselines: accuracy + macro-F1.
- Confusion matrices of B7 vs B8 side by side — the left/right confusion should collapse.
- Pooling: max vs avg vs sum; 1 vs 2 vs 4 groups (the paper says 4 does not help).
- LSTM sizes and timesteps (paper: person 5 / scene 10 is best).
- 3 seeds on the important models, report mean ± std, and bootstrap by video.

### Phase 5 — beyond the paper

1. Single end-to-end network, two losses (`L_group + λ·L_person`) — the course's challenge.
2. YOLO / Faster R-CNN detection + IoU or ByteTrack tracking, replacing the given boxes. Compare
   accuracy with GT boxes vs detected boxes: that number is the honest "in the wild" result.
3. Attention or a Transformer encoder in place of max-pooling; a GNN over players
   (Ibrahim & Mori 2018, then GroupFormer / DIN).

## Decisions fixed now

- **T = 9 frames** for every baseline (the course's choice). The paper uses 5 person / 10 scene
  timesteps; we can test that in Phase 4. Never more than ~10 — see `ANALYSIS.md` §5–6.
- **Backbone ResNet50**, ImageNet weights, fine-tuned (never frozen for the first training of a
  stage — the course is explicit: "don't try anything that doesn't finetune").
- **Selection on val**, test touched once per model, at the end.
- **Metrics**: accuracy + macro-F1 + confusion matrix, always.
