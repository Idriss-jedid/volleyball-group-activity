# What the data says about the problem

`DATASET.md` describes the files. This document is about what the annotations imply for the model:
what is learnable, where the ceiling is, and which architectural choice actually matters.

Every number below was measured on this dataset. The oracle experiments use the **ground-truth**
person actions and boxes as input to a random forest, train on the official train videos and report
on the official test videos. They are upper bounds on the information carried by the *labels* —
real CNN features carry more (appearance, jersey, ball), so a real model can go past them.

## 1. The task splits cleanly into two sub-problems

| Predict | Input: GT actions, no spatial info | Input: GT actions split left / right |
|---|---:|---:|
| **kind** (set / spike / pass / winpoint) | **94.4 %** | 94.3 % |
| **side** (left / right) | **51.7 %** (= chance) | **89.4 %** |
| full 8 classes | 49.4 % | 88.4 % |

Read this twice:

- **Knowing who does what is enough to get the kind of the activity** — 94 % with nine discrete
  labels and nothing else. Spatial information adds nothing here.
- **Side is a pure spatial problem.** With the same actions and no notion of where they happen,
  side is a coin flip. All of the information lives in *which half of the court* the action is in.

The 8-class task is `kind × side`, so a model that nails the kind and guesses the side lands at
about 50 %. That is precisely what a global pooling over people gives.

## 2. Simulating the pooling layer explains the whole B7 → B8 gap

Each player is treated as a one-hot vector over the 9 actions — this is exactly what the pooling
layer of the paper sees, minus the appearance features.

| Pooling scheme | Dim | Test acc | Macro-F1 |
|---|---:|---:|---:|
| max-pool over all players (≈ B3 / B7) | 9 | 49.0 % | 44.3 % |
| sum-pool over all players | 9 | 50.3 % | 47.9 % |
| **max-pool per team (≈ B8)** | 18 | **85.9 %** | 82.3 % |
| sum-pool per team | 18 | 88.6 % | 86.1 % |
| max-pool per team + team x-spread | 20 | 86.2 % | 82.9 % |

- Global pooling over all players caps the *label* information at ~50 %, but reaches **95.1 %** on
  the 4-way kind. The architecture is not weak at recognizing the activity; it is blind to the side.
- Splitting into two teams before pooling recovers ~36 points. This is the single most valuable
  architectural decision in the whole project, and it costs one `sort by x` plus one `concat`.
- Max vs sum barely differ on labels (max is slightly worse). The paper's claim that max beats
  average is about *learned features*, not about label counts — worth re-testing ourselves.

That real B7 reaches ~80 % in the paper, not 50 %, means the CNN features smuggle spatial and
appearance information (jersey colour, court position, camera geometry) past the pooling layer. The
gap between 50 % and 80 % is the part the person-action labels cannot explain.

## 3. The realistic ceiling

With GT actions **and** GT positions (43 hand-made features), the test accuracy is **90.1 %**
(macro-F1 87.7 %). With positions only and no actions, 59.5 %.

So ~90 % is roughly what a *perfect* person-action recognizer plus layout gives. Published models
that report 93–95 % are necessarily using signals the action labels do not contain: motion,
appearance, the ball, pose. Setting a target above 90 % is reasonable, but only with features, not
by improving the person classifier alone.

### Where the remaining errors are

Confusion matrix of the 90.1 % oracle (rows = true):

|  | l_pass | l_set | l_spike | l_win | r_pass | r_set | r_spike | r_win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **l_pass** | 210 | 0 | 0 | 2 | 8 | 0 | 0 | 6 |
| **l_set** | 1 | 162 | 3 | 0 | 0 | 2 | 0 | 0 |
| **l_spike** | 0 | 2 | 170 | 0 | 5 | 0 | 2 | 0 |
| **l_winpoint** | 1 | 0 | 0 | 80 | 8 | 0 | 0 | **13** |
| **r_pass** | 9 | 0 | 0 | 10 | 190 | 0 | 1 | 0 |
| **r_set** | 2 | 7 | 0 | 0 | 3 | 178 | 1 | 1 |
| **r_spike** | 5 | 0 | 7 | 0 | 0 | 1 | 160 | 0 |
| **r_winpoint** | 3 | 0 | 0 | **28** | 1 | 0 | 0 | 55 |

`r_winpoint` → `l_winpoint` is the single worst cell (28 of 87). The reason is in the data: in
winpoint clips **94.5 % of players are labelled `standing`** and 3.8 % `falling`. Both teams stand;
the label depends on *who is celebrating*, which the nine action classes cannot express. Winpoint
side will only be solved from appearance — and it is also the rarest class (6.1 % and 7.6 %).

Set ↔ pass, the confusion the paper reports, is small here (7 and 3 cells) because the GT actions
`setting` and `digging` are clean. In a real model it will be larger, since those two actions are
themselves the hard ones to classify.

## 4. The dataset is not biased in the easy ways

| Shortcut | Accuracy it would give |
|---|---:|
| Always predict the global majority class (`l_pass`) | 16.9 % |
| **Oracle**: always predict the majority class of *that video* | 19.9 % |
| Left vs right overall balance | 51.1 % left |

Per-video class share is 20.4 % on average (uniform = 12.5 %), max 27 %. So there is no venue
shortcut: a model cannot get far by recognising the arena or the scoreboard. Good news — it means a
high score has to come from the actual activity.

## 5. Temporal structure: the rally grammar

Consecutive annotated clips less than 60 frames apart follow the volleyball rally:

| Transition | Count |
|---|---:|
| l_set → l_spike | 436 |
| r_set → r_spike | 419 |
| l_pass → l_set | 394 |
| r_pass → r_set | 373 |
| l_spike → r_pass | 283 |
| r_spike → l_pass | 282 |

`pass → set → spike → (other team) pass`. Two consequences:

- Clip labels inside a video are **not independent**; a sequence model over whole clips would be
  able to exploit this. That is outside the paper's setting, but it is a legitimate extension.
- More importantly, it is a **hazard for the temporal window**. Gaps between consecutive clips:
  p1 = 5, p5 = 10, p25 = 25, median 35 frames. And:

| Clip pairs closer than | Share | Of those, **different** label |
|---|---:|---:|
| 10 frames | 2.1 % | 16 % |
| 20 frames | 13.3 % | 79 % |
| 41 frames | 61.5 % | 85 % |

A 41-frame window (the full clip) overlaps a neighbouring clip that carries a **different** label
most of the time. Using the whole available window would feed the next phase of the rally into the
current sample. The paper's 9–10 frame window is not arbitrary; it is about the largest window that
stays inside one phase.

## 6. Broadcast cuts — only in `winpoint`

Mean absolute frame difference across the window, 25 clips per class:

| Window | Clips containing a shot cut |
|---|---:|
| ±20 frames (whole clip) | 8 % overall — **34 % of winpoint clips**, 0 % of every other class |
| ±10 frames | 4 % overall — 14 % of winpoint |
| −5 .. +4 (our T = 9) | 0.5 % overall — 2 % of winpoint |

Example: `videos/50/38655` (`r_winpoint`) cuts at offset +6 from a wide court shot to a close-up of
the celebrating team (`docs/figures/winpoint_cut.jpg`). After a cut, the ground-truth boxes are
meaningless and any tracker breaks.

**Conclusion: keep T ≤ 10.** If we ever want a longer window, we must detect cuts first (a simple
frame-difference threshold works — the cut frames are 4× the median difference and above 12).

Motion also explains why the window is asymmetric. Averaged over clips, `*_pass` has its highest
motion **before** the target frame (≈5.5 at offset −20 falling to ≈2.0 at +8), while `*_spike` is
flat. Five frames before and four after captures the approach, which is where the discriminative
motion is.

## 7. Tracking is feasible without downloading anything

The zip has no track ids, but overlapping clips give a free measurement: two annotated frames G
frames apart, matched greedily by box centre.

| Gap G | Median displacement per frame | Median IoU after G frames | IoU > 0.5 |
|---:|---:|---:|---:|
| 5 | 4.7 px | 0.42 | 39 % |
| 10 | 4.2 px | 0.21 | 18 % |
| 15 | 3.4 px | 0.13 | 13 % |
| 20 | 2.8 px | 0.12 | 10 % |

A player moves ~4.4 px per frame (p90 = 10 px, p99 = 16 px) against a median box of 54 × 123 px.
Frame-to-frame IoU is therefore high and a greedy IoU tracker chained frame by frame will hold over
9 frames. What does **not** work is matching two frames 10+ apart directly — IoU drops to 0.21.

So: seed from the ground-truth boxes on the target frame and propagate ±4 frames with a detector +
IoU association, or use ByteTrack. Hard cases are the fast actions (spike, dig) and crossings at the
net, where two players overlap.

## 8. What this implies for our baselines

1. **B8 is not an optional extra** — it is the only baseline that can pass ~50 % on label
   information alone. Expect the largest single jump there.
2. **The person-action classifier only has to be good at the rare classes.** Getting `standing`
   right (68.7 % of boxes) buys nothing; `setting`, `spiking`, `digging` decide the activity kind.
   Train it with class weights and select on macro-F1.
3. **Winpoint needs appearance.** Do not expect any action-based model to separate `l_winpoint` from
   `r_winpoint`; keep the crops' colour information and do not over-augment it away.
4. **Never flip horizontally without swapping the label**, and note that flipping is otherwise the
   ideal augmentation here since the two classes sets are mirror images.
5. **T = 9 (−5..+4), fixed for every baseline.** Justified by the gap analysis and the cut analysis.
6. **Report macro-F1 alongside accuracy**, and remember the test set is correlated: 60 % of
   consecutive test clips share frames with their neighbour, so a binomial confidence interval on
   1,337 clips is too optimistic. Bootstrap by video.
