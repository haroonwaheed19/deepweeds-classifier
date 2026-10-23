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
