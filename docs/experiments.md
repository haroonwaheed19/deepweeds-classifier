# Experiment log

Dated notes, including the things that did not work. Numbers here come from the committed
`reports/metrics/*.json` files.

## 2026-10-23: P1 data audit (`python -m weedcls.data.inspect`)

**What the data is.** 17,509 JPEGs, all 256x256 RGB, ~27 KB each, no unreadable files, no
duplicate filenames (matches the paper's count). Labels are one CSV row per image:
`Filename,Label,Species`, labels 0-7 are the eight weeds and 8 is `Negative`. Licence is CC BY 4.0.

**Class balance.** Negative 9,106 (52.0%); each weed 1,009-1,125. Largest/smallest is 9.0x, but
the eight weed classes are balanced among themselves (1.1x), so the imbalance is really
"weed vs not-weed", not between species.

**Sessions are tangled up with class.** Filenames carry a timestamp (`YYYYMMDD-HHMMSS-k.jpg`).
There are 60 capture dates from 2016-09-28 to 2018-03-22, and most dates contain only 1-3 classes
(14 dates are Negative-only). Number of dates each class appears on:

| class | dates | biggest single-date share |
|---|---|---|
| Parkinsonia | 2 | 67% |
| Parthenium | 3 | 51% |
| Siam weed | 3 | 74% |
| Prickly acacia | 6 | 49% |
| Rubber vine | 6 | 53% |
| Lantana | 14 | 49% |
| Snake weed | 20 | 16% |
| Chinee apple | 22 | 30% |
| Negative | 34 | 16% |

Consequence: a random image split puts frames from the same session (same light, same background,
same colour cast) on both sides, so a model can score well by recognising the session. And a
strict one-date-per-split design is impossible for Parkinsonia (2 dates cannot fill 3 splits).

**The suffix number is not noise.** The trailing `-k` (0-4) is strongly class-correlated: 860 of
1,016 Snake weed images (85%) have k=0, against 358 of 9,106 Negatives (3.9%); k=0 contains no
Parkinsonia, Parthenium or Rubber vine at all, and k=4 is 27 images (26 Parthenium). I don't
know what k encodes (the dataset page doesn't say), so I won't use it as a feature, and I'll
check in P5 whether errors cluster on it.

**Near-duplicates are not a big problem.** pHash (64-bit) finds 96 pairs at Hamming distance <=4
and 110 at <=6, **zero** across classes, **zero** across dates. So the "cross-class duplicates as
label-noise evidence" idea from the plan found nothing. Duplicates all sit inside a single
date, so grouping by session already keeps them together.

**What the pictures show** (`reports/figures/sample_grid.png`): the plant is often small, partial
or buried in clutter (several Parkinsonia frames show almost no visible plant); there are heavy
shadows and very dark frames; and classes differ in colour cast (Prickly acacia / Parthenium look
blue-grey, Rubber vine bright green), which is another session-shortcut risk.

**Decision for P2.** Group by (date, hour): 195 groups, every class has >=14 groups
(date+30min would give 303 groups, >=21). Largest group is 542 images. This is weaker than
holding out whole dates, because for Parkinsonia/Parthenium/Siam weed train and test will share
a date but not a time block. I will state this limitation in the README instead of claiming
site-level generalisation.

## 2026-10-23: P2 splits and preprocessing

**First attempt, rejected.** `StratifiedGroupKFold` (6 outer folds for test, 5 inner for val,
best of 300 seeds) on (date, hour) groups. Result was too lumpy: test came out at 11.6% of the
data, Parkinsonia had **16** test images and Snake weed 70 val images. A few groups hold 500+
images, so fold-based balancing can't do better.

**What I use instead.** Whole (date, hour) groups are assigned to train/val/test by hill-climbing
on the squared deviation of every (split, class) share from 70/15/15 (swap/move moves, 40
restarts x 6000 steps, best restart kept; `weedcls.data.splits`). 193 groups (195 before merging
pHash near-duplicate pairs that straddled an hour boundary).

| class | train | val | test |
|---|---|---|---|
| Chinee apple | 784 | 164 | 177 |
| Lantana | 745 | 160 | 159 |
| Negative | 6373 | 1362 | 1371 |
| Parkinsonia | 722 | 155 | 154 |
| Parthenium | 721 | 145 | 156 |
| Prickly acacia | 750 | 156 | 156 |
| Rubber vine | 725 | 142 | 142 |
| Siam weed | 752 | 164 | 158 |
| Snake weed | 708 | 154 | 154 |
| **total** | **12280 (70.1%)** | **2602 (14.9%)** | **2627 (15.0%)** |

Every class is 14.1-15.3% of val and test. Groups: 106 train / 55 val / 32 test. Test has only
32 groups, so its score is noisy; I will put bootstrap CIs on it (P5) rather than trust a point
estimate.

**Honest limit.** The split keeps hour-long blocks apart, not whole days: 26 dates appear in both
train and val, 17 in train and test. For the rare classes (Parkinsonia, 2 dates) this means the
test measures "same fields, different time of day", not "new field". Scores on this split are
therefore still somewhat optimistic about truly new sites.

**Normalisation.** Train-split channel mean/std = (0.380, 0.394, 0.384) / (0.233, 0.232, 0.230),
darker than ImageNet (0.485, 0.456, 0.406). Pretrained models keep ImageNet stats (that is what
their weights expect); the from-scratch CNN uses the dataset stats.

**Augmentation** (`reports/figures/augmentation_examples.png`): random resized crop scale
0.7-1.0, horizontal flip + random 90 degree rotation (top-down camera, orientation is
meaningless), colour jitter brightness/contrast 0.3, saturation 0.1, hue 0.02 (hue kept tiny
because leaf colour is a species cue). The brightest jittered samples look a bit blown out;
if the training curves show under-fitting I'll lower it to 0.2.

**Imbalance handling, decided now, compared in P3/P4.** Class-weighted cross-entropy with
inverse-sqrt frequency weights (average weight over samples = 1) versus a
`WeightedRandomSampler`; the winner is picked by val macro-F1.

## 2026-10-23: P3 M0 floor and M2 linear probe (all numbers are VALIDATION; test untouched)

**M0, always predict "Negative".** Accuracy 0.523, balanced accuracy 0.111, macro-F1 **0.076**,
weed recall 0. A model that does nothing gets 52% accuracy, which is why macro-F1 is the
headline metric here and accuracy is only reported.

**M2, frozen ImageNet features + logistic regression.** Embeddings cached once per backbone
(MobileNetV3-L ~7 min, EfficientNet-B0 ~11 min on this laptop; the benchmark had predicted 5 and 8,
the gap is the DataLoader sharing 2 cores). 24 configs (2 backbones x C in {3e-4, 3e-3, 3e-2, 0.3}
x class weighting {none, inverse-sqrt, balanced}), ~2.5 s per fit
(`reports/metrics/linear_probe_sweep.csv`).

| | val macro-F1 | val acc | weed recall | false-alarm rate |
|---|---|---|---|---|
| best: EfficientNet-B0, C=0.003, no weights | **0.706** | 0.759 | 0.812 | 0.184 |
| best MobileNetV3-L (C=0.03, no weights) | 0.682 | 0.738 | 0.851 | 0.232 |
| EfficientNet-B0, C=0.03, balanced weights | 0.655 | 0.670 | 0.930 | 0.413 |

Things worth knowing:

- **Class weighting lowers macro-F1 but raises weed recall.** "Balanced" weights push weed
  recall to 0.93-0.96 at the cost of a 41-66% false-alarm rate; no weighting sits at 0.81 recall
  and 18% false alarms. So the choice depends on the cost of a miss vs a wasted spray, not on a
  single number. For selection I stay with macro-F1 but I'll tune a decision threshold in P4
  and report the recall/false-alarm trade-off.
- **EfficientNet-B0 beats MobileNetV3-L by 0.024 macro-F1.** With 55 validation groups that is
  within what I'd expect from noise, so I'm not reading it as a real ranking. And B0 costs
  36 min/epoch to fine-tune vs 15.6 (benchmark), which breaks my compute gate. Fine-tuning
  therefore goes ahead with MobileNetV3-L (decision based on cost, with a small accuracy price).
- **Leakage gap.** Best config re-scored on the paper's random 60/20/20 split: macro-F1 0.806 and
  accuracy 0.851, versus 0.706 and 0.759 on the grouped split. A random split overstates
  performance by about **10 macro-F1 points** for the identical model and features, which is
  the reason the grouped split exists.
- Transfer learning is justified before spending any fine-tuning compute: frozen features alone
  reach 0.71 macro-F1 in seconds; the from-scratch CNN has to beat this to earn its place
  (result below once trained).

## 2026-10-23: P3 small-CNN sweep (30% of train, 4 epochs each, val macro-F1)

6 runs: lr {1e-3, 3e-3} x imbalance {weighted loss, WeightedRandomSampler, none}
(`reports/metrics/small_cnn_sweep.csv`; ~2 min/epoch at this subset size).

| lr | imbalance | val macro-F1 | weed recall | false alarm |
|---|---|---|---|---|
| 3e-3 | weights | **0.362** | 0.681 | 0.400 |
| 3e-3 | sampler | 0.360 | 0.820 | 0.558 |
| 1e-3 | weights | 0.359 | 0.701 | 0.436 |
| 1e-3 | sampler | 0.342 | 0.738 | 0.553 |
| 1e-3 | none | 0.304 | 0.365 | 0.139 |
| 3e-3 | none | 0.271 | 0.336 | 0.121 |

- Weighted loss and sampler tie on macro-F1 (0.362 vs 0.360); the sampler buys recall by paying
  in false alarms. I'm going with the weighted loss: same score, no resampling of the data.
- No imbalance handling is clearly worse for this model (0.27-0.30): it drifts to "Negative" and
  misses ~2/3 of weeds. Note this is the opposite of the linear probe, where weighting cost
  macro-F1. The probe starts from strong features, so it doesn't need the push; an under-trained
  CNN does.
- lr 3e-3 vs 1e-3: 0.003 apart for weights, which I treat as a tie; 3e-3 chosen as the better of
  the two tested values. These are 4-epoch runs on a subset, so they say little about the
  best LR at 16 epochs.
- Overall level is low (0.36) after 4 short epochs. The full run uses all training data for up to
  16 epochs.

## 2026-10-23: P3 M1 small CNN from scratch, full run (validation)

0.32M-parameter CNN (4 conv-BN-ReLU-pool blocks + GAP, 128px, dataset normalisation), AdamW lr 3e-3
with cosine decay, weighted loss, 16 epochs on all 12,280 training images, ~5.2 min/epoch (the benchmark
predicted 3.9 for training alone; validation adds ~0.5). Best epoch 15 (`reports/metrics/m1_small_cnn_*`).

| | val macro-F1 | val acc | weed recall | false alarm |
|---|---|---|---|---|
| M0 majority | 0.076 | 0.523 | 0.00 | 0.00 |
| **M1 small CNN** | **0.558** | 0.591 | 0.902 | 0.461 |
| M2 probe (EfficientNet-B0, frozen) | 0.706 | 0.759 | 0.812 | 0.184 |

- **Transfer learning is justified.** Frozen ImageNet features + logistic regression, which costs
  ~11 minutes of embedding extraction and seconds of fitting, beat a CNN trained for ~80 minutes
  by 15 macro-F1 points. With 1,000 images per weed class, a scratch model has too little data
  to learn what a leaf looks like.
- The CNN's weak spot is false alarms: 46% of Negative images get flagged as weeds (Negative
  recall 0.54, precision 0.86). Its weighted loss made it trigger-happy; the probe with no
  weights was more balanced.
- Per-class F1: Siam weed 0.86 and Lantana 0.68 are fine; Parthenium 0.36, Prickly acacia 0.43,
  Chinee apple 0.44 are weakest. Those are the ones with the most session-specific
  appearance (blue-grey cast for Parthenium/Prickly acacia), so a model trained from scratch has
  little to go on beyond colour and texture.
- Training was noisy: val loss swung 1.1-2.1 between epochs, because lr 3e-3 is aggressive for
  a BatchNorm network with batches of 64, and validation images come from different
  sessions than the training ones. Macro-F1 still trended up (0.26 -> 0.56) with no early stop. I did not
  chase more epochs: the question M1 had to answer is "does scratch training compete?", and it doesn't.

## 2026-10-23: P4 fine-tuning sweep, MobileNetV3-Large @224 (30% of train, 3 epochs: 1 head-only + 2 unfrozen)

ImageNet normalisation, weighted loss, label smoothing 0.1, backbone LR = 0.1 x head LR
(`reports/metrics/finetune_sweep.csv`).

| lr (head) | last N blocks trainable | val macro-F1 | weed recall | false alarm | min/epoch |
|---|---|---|---|---|---|
| 1e-3 | **8** | **0.683** | 0.907 | 0.325 | 3.3 |
| 1e-3 | 4 | 0.662 | 0.928 | 0.418 | 2.8 |
| 1e-3 | 2 | 0.625 | 0.916 | 0.462 | 2.5 |
| 3e-4 | 4 | 0.620 | 0.921 | 0.491 | 2.8 |
| 3e-4 | 2 | 0.595 | 0.918 | 0.554 | 2.5 |

- Both knobs move the same way: higher LR and more unfrozen blocks are better, monotonically.
  After the first 4 runs the best point was on the edge of the grid, so I added `last 8`, which
  won again. I stopped there for budget reasons (all-layer training measured 15.6 min/epoch);
  deeper unfreezing might still help, and I haven't tested it.
- Only 3 epochs on 3.7k images, so these are far from converged, but they rank the settings.
  Differences of ~0.02 are within noise; the 0.09 spread from lr 3e-4/last 2 to lr 1e-3/last 8 is not.
- Unfreezing 4 -> 8 blocks cost only +0.5 min/epoch on the 30% subset (2.8 -> 3.3). Backward
  only runs through the unfrozen stages; the frozen early stages still pay their forward cost.
- Full run (all 12,280 train images, lr 1e-3, last 8, 8 epochs, patience 3) started.

## 2026-10-23: P4 M3 fine-tuned MobileNetV3-Large, full run (validation; test still untouched)

Config `reports/metrics/m3_config.yaml`: 224px, ImageNet norm, weighted loss, label smoothing 0.1,
head lr 1e-3 / backbone 1e-4, cosine, epoch 0 head-only then last 8 of 17 backbone blocks
unfrozen (1.24M trainable in the head-only epoch, 4.09M of 4.21M total once 8 blocks are unfrozen), all 12,280 train images, 8 epochs, ~10 min/epoch
(head-only epoch 6 min). No early stop. Curve in `reports/metrics/m3_training_curve.csv`.

| | val macro-F1 | val acc | weed recall | false alarm |
|---|---|---|---|---|
| M0 majority | 0.076 | 0.523 | 0.00 | 0.00 |
| M1 small CNN (scratch) | 0.558 | 0.591 | 0.90 | 0.46 |
| M2 probe (EfficientNet-B0, frozen) | 0.706 | 0.759 | 0.81 | 0.18 |
| **M3 fine-tuned MobileNetV3-L** | **0.801** | 0.836 | 0.93 | 0.17 |
| M3 + temperature + Negative bias | **0.816** | 0.857 | 0.85 | 0.08 |

Val macro-F1 was still climbing at epoch 8 (0.711, 0.759, 0.761, 0.770, 0.780, 0.795, 0.801 from
epoch 1), so 8 epochs under-trains this model a little. I kept the run as is: it already
answers "does fine-tuning beat the probe enough to justify the cost" (+9.5 macro-F1 points).
More epochs is the first thing I'd buy with more compute.

**Calibration** (fit on val logits only, `scripts/calibrate.py`, `reports/metrics/calibration.json`):
- Temperature T = 0.72. Below 1, so the model is *under*-confident (label smoothing and class
  weights flatten the softmax). ECE 0.107 -> 0.017. Reliability plot: `reports/figures/reliability_val.png`.
- Negative-logit bias +1.8 (applied after dividing by T). Weighted training makes the model
  trigger-happy on weeds; the bias moves the operating point: macro-F1 0.801 -> 0.816, false alarms
  17% -> 7.7%, but weed recall 93% -> 85%. The curve is flat between bias 0.4 and 2.8
  (macro-F1 0.804-0.812) so the exact value isn't fragile; the choice is really recall vs
  wasted spray. The API returns `weed_probability` so a client can pick its own threshold.
  ECE after the bias is 0.034 (bias slightly de-calibrates, accepted).
- "Uncertain" flag: confidence < 0.619, chosen so predictions above it are 90% accurate on val;
  it flags ~10% of images (10.7% on a 300-image sample through the ONNX predictor).
  Tried a 0.85 accuracy target first: threshold came out at 0.23 and flagged nothing, because
  overall accuracy already exceeds 0.85, so a stricter target was needed to be useful.

**Per-class val F1 (after calibration):** Parkinsonia 0.93, Siam weed 0.93, Lantana 0.91,
Negative 0.90, Snake weed 0.87, Rubber vine 0.85, Chinee apple 0.76, **Prickly acacia 0.63,
Parthenium 0.57**. Same two classes were also the weakest for the scratch CNN, so it is not
a model-capacity issue. To investigate in P5 (both have few capture dates: 6 and 3, and a blue-grey
colour cast).

**Export check.** ONNX export (opset 17) matches PyTorch to 6.7e-6 max abs logit difference.
File is 16.8 MB. The torch-free `Predictor` agrees with the PyTorch pipeline on 300/300 val
images (same temperature/bias), 9 ms per image on this CPU (`artifacts/model_card.json`
holds T, bias, threshold, normalisation, classes).

## 2026-10-23: P5 test-set evaluation and error analysis

All thresholds (temperature 0.72, Negative bias +1.8, uncertainty cut-off 0.619) were fit on
validation. Test = 2,627 images, 32 capture groups. Run via `python -m weedcls.eval.evaluate`;
95% CIs come from a **group bootstrap** (1000 resamples of whole capture groups, since frames in a
group are near-copies). The first run printed sklearn warnings: some resamples contain no image of
a rare class, and my macro-F1 scored that class as 0, biasing the intervals down. I fixed
`compute_metrics` to average over classes present in the resample and re-ran. Point estimates
were identical; the same final models were scored twice, nothing was re-tuned in between.

| model | macro-F1 [95% CI] | acc | weed recall | false alarm | params | CPU ms/img |
|---|---|---|---|---|---|---|
| M0 majority | 0.076 [0.029, 0.105] | 0.522 | 0.00 | 0.000 | - | - |
| M1 small CNN (scratch) | 0.584 [0.418, 0.661] | 0.670 | 0.850 | 0.263 | 0.32M | 10 (torch, 128px) |
| M2 EfficientNet-B0 frozen + LR | 0.710 [0.564, 0.764] | 0.773 | 0.755 | 0.126 | (frozen backbone) | - |
| M3 fine-tuned MobileNetV3-L, raw | 0.815 [0.663, 0.881] | 0.863 | 0.904 | 0.094 | 4.21M | 29 torch / 9 ONNX |
| **M3 + temperature + Negative bias (shipped)** | **0.811 [0.659, 0.880]** | 0.869 | 0.830 | 0.038 | 4.21M | 9 (ONNX, 16.8 MB) |

Paired group-bootstrap differences in macro-F1 (same resamples for both models):
- M3 vs M2: **+0.103 [0.051, 0.162]**; M3 vs M1: **+0.232 [0.157, 0.305]**. Both exclude 0.
- M3 calibrated vs raw: -0.002 [-0.031, +0.035]. **The +0.015 val gain from the Negative bias did not
  carry over to test.** It is an operating-point shift (false alarms 9.4% -> 3.8%, weed recall
  90.4% -> 83.0%), not an accuracy gain, and I will describe it that way.
- The CI on any single model is wide (M3: 0.66-0.88) because there are only 32 test groups and
  some classes sit in one or two. Val (0.816) and test (0.811) agree, which is reassuring, but
  rankings between close models would not be trustworthy at this test size.

Leakage gap (P3, same probe): 0.806 on the paper's random split vs 0.706 on mine: +0.10.

### Where it fails (344 test errors; `reports/metrics/error_breakdown.json`)

- **Parthenium is the main failure.** Test recall 20.5% (val 49%). 86 of the 214 missed weeds are
  Parthenium; 33 more Parthenium images are called Prickly acacia. All 156 test Parthenium images
  come from a single session (2017-09-06, 10:00-12:00) and no training Parthenium comes from that
  date. The split search put that whole session in test.
- **Session shift is the dominant mechanism, not a Parthenium quirk.** Over all weed test images,
  species recall is **0.893 when the class's capture date also appears in train (n=847) and 0.509
  when it doesn't (n=409)**; accuracy CIs [0.85, 0.93] vs [0.27, 0.80]
  (`reports/metrics/date_shift_analysis.json`). Per class: Lantana 0.96 vs 0.37, Snake weed 0.75 vs
  0.45, Chinee apple 0.92 vs 0.78, but Rubber vine 0.82 vs 1.00 (93 images), so it is not
  universal. Caveat: "unseen date" is confounded with class (Parthenium is unseen and also hard),
  so this is strong evidence, not proof. It also means my hour-level split is optimistic for
  classes whose test dates overlap train dates (Parkinsonia 0.97, Prickly acacia 0.89): their
  test sessions share a date with training.
- **Misses vs false alarms.** 214 weeds called Negative (17% of weed images), 52 Negatives called
  a weed (3.8%), and only 78 weed images (7.5% of those found) given the wrong species. So the
  model is much better at "which weed" than at "weed or not" when the weed is subtle.
- **Confident wrong answers.** Median confidence of an error is 0.79 (quartiles 0.58 / 0.91). The
  uncertainty flag (< 0.619) catches 31% of errors while flagging 3.8% of correct predictions.
  Useful, but most errors are confident ones.

Manual failure-mode tagging of 60 randomly sampled errors (`reports/error_tags.csv`, tags assigned
by eye from `reports/figures/error_sheet_*.png`, so subjective):

| tag | share | meaning |
|---|---|---|
| A | 25% (15) | small or sparse weed in leaf litter / dead grass, called Negative (9 are Parthenium) |
| B | 22% (13) | one weed species confused with another (9 Parthenium -> Prickly acacia/Chinee apple) |
| C | 20% (12) | true Negative with a non-target look-alike plant called a weed |
| F | 17% (10) | weed clearly visible but called Negative |
| D | 10% (6) | harsh shadow, overexposure or colour cast |
| E | 7% (4) | no clear weed visible (possible label noise) |

Grad-CAM on 12 of these (`reports/figures/gradcam_errors.png`, a small sample): for Parthenium called
Negative the evidence for "Negative" sits on dead leaves and litter, not on the small plant; for
weeds called Negative on sand the attention goes to the sand background; for look-alike false
alarms the model attends to the pinnate leaves themselves, so that one is about class
definition, not background.

False-alarm cluster: 33 of 52 false alarms are Negatives predicted as Chinee apple, 25 of them from
one session (2018-01-09, 466 test Negatives, 5.4% of them hit). The training set holds 1,009
Negatives from that same date, so this is not missing coverage. It is a small share of one
session whose leaflets look like Chinee apple at this scale.

Suffix `-k` and date accuracy are confounded with class, so I only note: suffix 0 has 64%
accuracy (n=251, mostly Snake weed), suffix 4 22% (n=18, Parthenium).

### Label noise (train, out-of-fold probe, `reports/metrics/label_noise_oof.json`)

Grouped 5-fold out-of-fold predictions on the 12,280 training images: 282 (2.3%) are confidently
contradicted (model p > 0.9 for another class; 92 are Negatives). I looked at the 40 most
contradicted: 7 (17.5%) are likely mislabels (a weed label with no target visible: person's
shadow, grass only, bare twigs), 13 (32.5%) look right and the model is wrong (often thrown by a
pink or blue colour cast), 20 (50%) are ambiguous. If the true mislabel share among all 282 is no
higher than in the top 40, that is under 50 images, **roughly 0.4% of train or less**; the top-40
is not a random sample so treat that as a loose ceiling. I did not remove or relabel anything:
the effect is too small to matter next to the session shift. Earlier finding (P1) still stands:
zero cross-class near-duplicates.

### Improvement hypotheses (each tied to an observation above; none run yet)

1. **Session shift (Parthenium 0.21, unseen-date recall 0.51 vs 0.89).** The model uses session
   appearance (lighting, colour cast, background) as a shortcut. Try stronger photometric
   augmentation (larger colour jitter, random grayscale, random shadow) and per-image colour
   normalisation (grey-world white balance). Test by retraining and reporting recall on the
   "unseen-date" subset with its CI. Longer-term fix is more sessions per class, which no
   algorithm replaces.
2. **Small weed in litter (tag A, 25% of errors).** A 224px resize plus global pooling dilutes a
   plant that covers a few percent of the frame. Try native 256px input and 2x2 overlapping-crop
   inference with max-pooling of weed probability. Test on the Parthenium and Snake weed misses.
3. **Look-alike Negatives (tag C, 20%).** The Negative class is a catch-all. Try hard-negative
   mining (upweight training Negatives the out-of-fold model gets wrong) or a separate
   "other plant" class. Expect fewer Negative -> Chinee apple false alarms.
4. **Operating point.** Weed recall 83% vs 17% missed is a business choice. Sweep the
   Negative-bias to match a target recall and report the false-alarm cost; the API already
   exposes `weed_probability`.
5. **Longer training.** Val macro-F1 was still rising at epoch 8 of the M3 run.
