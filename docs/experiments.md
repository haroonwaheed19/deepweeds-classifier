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
