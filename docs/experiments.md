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
