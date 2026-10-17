# Volleyball Dataset — verified notes

Everything here was measured directly from `D:\volleyball.zip` (62.0 GB zip, 62.3 GB uncompressed,
202,984 entries), not copied from the paper or the README.

## Layout

```
videos/
  Info.txt                    # official description + splits + label counts
  <video_id>/                 # 0 .. 54
    annotations.txt
    annotations.txt~          # stale backup, see "Gotchas"
    <clip_id>/                # = target frame id
      <frame_id>.jpg          # exactly 41 frames: target-20 .. target+20
```

- 55 videos, 4,830 clips, 198,030 jpg (4,830 × 41), mean 307 KB per jpg.
- Clips per video: min 35 (v39), median 89, max 161 (v3).
- **No tracking annotations in this zip.** Only the target frame of each clip is annotated.

## Annotation format

```
{frame_id}.jpg {group_activity} {X Y W H action} {X Y W H action} ...
```

⚠️ `Info.txt` documents the player block as `{Action Class} X Y W H`. **The real order is
`X Y W H {Action}`** — the action is last. Verified on all 4,830 lines.

`X Y W H` is a top-left corner plus size, in pixels of the source video.

## Integrity checks (all pass)

| Check | Result |
|---|---|
| Annotated frames parsed | 4,830 |
| Malformed lines | 0 |
| Duplicate (video, clip) | 0 |
| Annotated clips without an image dir | 0 |
| Clips whose frame count ≠ 41 | 0 |
| Clips missing their target frame | 0 |
| Image dirs with no annotation | 0 |
| Window always centred on target (−20..+20) | yes |
| Boxes with w ≤ 0 or h ≤ 0 | 0 |
| Boxes outside image bounds | 6 (clip them) |

## Labels

### Group activities (8) — 4,830 clips

| Label (as written in the files) | Count | % |
|---|---:|---:|
| `l-pass` | 826 | 17.1 |
| `r-pass` | 801 | 16.6 |
| `r_set` | 644 | 13.3 |
| `l-spike` | 642 | 13.3 |
| `l_set` | 632 | 13.1 |
| `r_spike` | 623 | 12.9 |
| `l_winpoint` | 367 | 7.6 |
| `r_winpoint` | 295 | 6.1 |

⚠️ **Inconsistent spelling**: `pass` and `l-spike` use a hyphen, while `r_spike`, `*_set` and
`*_winpoint` use an underscore. Always normalize `-` → `_` before building the label map.

### Person actions (9) — 56,327 boxes, 11.66 per frame

| Action | Count | % |
|---|---:|---:|
| standing | 38,686 | 68.7 |
| moving | 5,120 | 9.1 |
| waiting | 3,600 | 6.4 |
| blocking | 2,458 | 4.4 |
| digging | 2,333 | 4.1 |
| setting | 1,332 | 2.4 |
| falling | 1,241 | 2.2 |
| spiking | 1,216 | 2.2 |
| jumping | 341 | 0.6 |

Extreme imbalance: `standing` is 68.7 % and `jumping` 0.6 % (113× ratio). The interesting actions
(spiking, setting) are exactly the rare ones. Use class weights / balanced sampling when training
the person-action classifier, and report macro-F1, not only accuracy.

## Splits (official, from `Info.txt` — by video, no frame leakage)

| Split | Videos | Clips | % |
|---|---:|---:|---:|
| train | 24 | 2,152 | 44.6 |
| val | 15 | 1,341 | 27.8 |
| test | 16 | 1,337 | 27.7 |

Train is only 44.6 % of the data — smaller than usual, keep it as is for comparability.
Per-class distribution is stable across the three splits (each class within ~±1.5 points of its
global share), so no re-balancing is needed.

## Geometry

- **Two resolutions**: 1280×720 for 47 videos, **1920×1080 for videos 2, 37, 38, 39, 40, 41, 44, 45**.
  Present in every split (train 11.4 %, val 5.3 %, test 12.6 % of clips).
  → Always normalize box coordinates by the frame's own width/height.
- Box size: median 54×123 px, aspect h/w p5=1.26, median 2.26, p95=3.76 → crops are tall; a plain
  224×224 resize distorts them.
- Players per frame: 12 (73.8 %), 11 (20.5 %), 10 (4.0 %), 9 (1.3 %), 8 (0.3 %), 7 (2 frames).
  → K is **at most** 12, not exactly 12. Pad and mask.

## Gotchas

1. **Player order inside a line is arbitrary** — only 1.7 % of lines happen to be sorted by x.
   Sort by box centre x before any per-player modelling.
2. **Team split is not "first 6 / last 6"**: only 73.8 % of frames have 12 players, and a mid-image
   cut gives splits like 7/4 or 8/3. Sort by x and cut at `n//2`, or cut at the largest x gap
   (median gap between the 6th and 7th player is 96 px, but the p5 is 44 px and one frame has a
   gap ≤ 0). Whatever rule is used, it must be applied **once per clip** and kept fixed across
   the T frames, otherwise a player can switch teams mid-sequence.
3. **`annotations.txt~` are stale backups.** Each of the 11 backup files differs from its `.txt` in
   exactly one line, where the old version has 13 players and the final one 12 — a duplicate box was
   removed. This explains the small mismatches with the counts printed in `Info.txt`
   (standing 38,696 vs 38,686 measured, waiting/moving/`l_set` off by one): **`Info.txt` was
   computed before this cleanup**. Ignore the `.txt~` files; trust the `.txt`.
4. **Only the 12 players are annotated** — referees, coaches and the crowd are not. A detector
   (YOLO / Faster R-CNN) will return them too and they must be filtered.
5. **Neighbouring clips overlap in time**: 2,936 of 4,775 consecutive clip pairs are less than 41
   frames apart (minimum gap 5). Their 41-frame windows share images, so two training samples can be
   nearly identical. Same video ⇒ same split, so it is not test leakage, but it inflates the
   effective training-set size.
6. **No ball annotation**, no pose, no track ids in this zip.

## Temporal window

41 frames are available per clip. `Info.txt`: *"In our work, we used 5 before and 4 after frames"*
— that is 5 + target + 4 = **10 frames**, while the paper says 9 timesteps. Pick one convention,
write it in the config and keep it identical across all baselines. Current choice: `num_frames: 9`.

## What is missing for the player-level LSTM

The person-level LSTM (B5, B7, B8) needs the same player tracked across the T frames, but this zip
only annotates the target frame. Two options:

1. Download the official `volleyball_tracking_annotation` from the authors' repo.
2. Generate tracks ourselves: a detector (YOLO / Faster R-CNN) + a tracker (ByteTrack / DeepSORT / a
   simple IoU tracker), seeded from the ground-truth boxes on the target frame and propagated
   backwards/forwards.

Option 2 is more work, but it is a genuine modernization of the 2016 pipeline (which used Dlib's
DSST tracker) and removes the dependency on extra downloads.

## Performance budget (measured on this machine, reading straight from the zip)

| | Value |
|---|---|
| Read one jpg from the zip | 4.9 ms (≈68 MB/s) |
| Read + decode + resize to 224 | 29.7 ms |
| One pass over 9 frames/clip (43,470 frames) | ≈21.5 min of pure IO/decode |
| Crops for 9 frames × 12 players | 521,640 |
| Feature cache, crops, fp16 | 2.14 GB |
| Feature cache, images, fp16 | 0.18 GB |

→ Decoding, not the GPU, is the bottleneck. Extract features **once** and cache them; only B1
fine-tuning has to touch the jpgs repeatedly.

## Sanity check that the labels mean what we think

Mean normalized x of the key actor per activity (`setting` for `*_set`, `spiking` for `*_spike`,
`digging` for `*_pass`):

| Activity | Key actor mean x |
|---|---:|
| l-pass | 0.34 |
| l-spike | 0.37 |
| l_set | 0.41 |
| r_set | 0.59 |
| r_spike | 0.62 |
| r-pass | 0.66 |

Left activities have their key actor on the left half and vice versa, as expected — the annotations
are consistent.
