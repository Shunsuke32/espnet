# Reference results

Accuracy is exact language-set accuracy. Individual-language classifiers use
top-1 for FLEURS and unordered top-2 for CS corpora; atomic-pair classifiers
expand one predicted class. ASR uses the generated sequence's language set,
not forced top-2. Ordered sequence metrics are named explicitly.
Percentages below are historical references, not verified fresh-server
regression targets. Confirm the exact experiment, checkpoint, reference IDs,
and exclusion policy before using a row in a reproduction comparison.

## LID1 classifier baselines

The FLEURS-only model uses ordinary language classes and takes softmax top-2 on
two-language tests.  Pair models treat each canonical pair (for example,
`ara-eng`) as one atomic class; scoring expands that class back to its unordered
two-language set.

| evaluation set | split | n | FLEURS-only top-2 | FLEURS+CS-FLEURS pair | CS-all pair |
|---|---|---:|---:|---:|---:|
| FLEURS | all/seen | 77,808 | unverified | 97.30% | 97.38% |
| CS-FLEURS all | all | 49,297 | 8.33% | 30.04% | 30.12% |
| CS-FLEURS all | seen | 20,154 | 10.70% | 73.47% | 73.66% |
| CS-FLEURS all | unseen | 29,143 | 6.69% | 0.00% | 0.00% |
| CS read | all | 5,818 | 13.42% | 0.19% | 0.69% |
| CS read | seen | 5,307 | 14.60% | 0.21% | 0.75% |
| CS read | unseen | 511 | 1.17% | 0.00% | 0.00% |
| CS XTTS1 | seen | 11,321 | 1.71% | 99.58% | 99.68% |
| CS XTTS2 | unseen | 17,226 | 0.51% | 0.00% | 0.00% |
| CS MMS | all | 14,932 | 20.37% | 23.59% | 23.58% |
| CS MMS | seen | 3,526 | 33.66% | 99.89% | 99.86% |
| CS MMS | unseen | 11,406 | 16.26% | 0.00% | 0.00% |

## LID2 vs final LID3 AAM/Sub-center+BCE

| evaluation set | split | n | LID2 soft-target KL | LID3 AAM+BCE |
|---|---|---:|---:|---:|
| FLEURS | all/seen | 77,808 | 91.35% | 96.50% |
| CS-FLEURS all | all | 49,297 | 52.04% | 59.71% |
| CS-FLEURS all | seen | 20,154 | 95.20% | 96.64% |
| CS-FLEURS all | unseen | 29,143 | 22.20% | 34.17% |
| CS read | all | 5,818 | 75.52% | 80.49% |
| CS read | seen | 5,307 | 82.65% | 88.13% |
| CS read | unseen | 511 | 1.57% | 1.17% |
| CS XTTS1 | seen | 11,321 | 99.65% | 99.61% |
| CS XTTS2 | unseen | 17,226 | 0.00% | 0.00% |
| CS MMS | all | 14,932 | 66.84% | 90.24% |
| CS MMS | seen | 3,526 | 99.80% | 99.89% |
| CS MMS | unseen | 11,406 | 56.65% | 87.25% |
| CS-YODAS | seen | 2,905 | 91.33% | 86.88% |

## FLEURS+CS-FLEURS model comparison

| evaluation set | split | n | LID2 | ASR d256 |
|---|---|---:|---:|---:|
| FLEURS | all/seen | 77,808 | 91.35% | 95.55% |
| CS read | all | 5,818 | 75.52% | 0.00% |
| CS read | seen | 5,307 | 82.65% | 0.00% |
| CS read | unseen | 511 | 1.57% | 0.00% |
| CS XTTS1 | seen | 11,321 | 99.65% | 99.74% |
| CS XTTS2 | unseen | 17,226 | 0.00% | 0.00% |
| CS MMS | all | 14,932 | 66.84% | 45.31% |
| CS MMS | seen | 3,526 | 99.80% | 99.83% |
| CS MMS | unseen | 11,406 | 56.65% | 28.45% |
| CS all | all | 49,297 | 52.04% | 36.63% |
| CS all | seen | 20,154 | 95.20% | 73.49% |
| CS all | unseen | 29,143 | 22.20% | 11.13% |

Run `evaluation/paired_significance.py` on the generated per-utterance details
for every claimed model comparison.  Aggregate percentages alone are not valid
inputs to a paired test.
