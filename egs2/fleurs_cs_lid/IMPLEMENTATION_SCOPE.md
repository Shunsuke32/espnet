# Implementation scope

## Included

- Canonical mapping of all 102 FLEURS labels and guarded canonical
  `nlsyms.txt` generation.
- Pinned FLEURS, CS-FLEURS, and six-pair CS-YODAS acquisition.
- Deterministic CS-FLEURS global 98:2 and optional class-wise 9:1 splits.
- CS-YODAS per-language, video-grouped 80:10:10 split and 70-second train/dev
  cap.
- Shared train/dev duration filtering and unfiltered evaluation manifests.
- LID1 atomic pair classes with deterministic `other-eng` ordering.
- Data-count-derived global effective-batch and pass budgets.
- ASR attention-only unordered two-token minimum-permutation loss.
- LID2 one-hot/0.5-soft targets and batch-mean KL loss.
- LID3 multi-hot targets, plain BCE, and AAM/Sub-center/Inter-TopK BCE with
  positive-class weighting.
- Saved-logit top-k and threshold scoring, exact set accuracy, sequence exact
  accuracy, prediction cardinality, seen/unseen and per-language-set summaries.
- Paired t-test, exact McNemar test, and paired-bootstrap confidence interval.
- Focused CPU tests for targets, losses, permutation loss, data preparation,
  evaluation, and statistics.

## Deliberately excluded

The old development branch contained many useful exploratory artifacts, but
they are not required to reproduce the final paper models or tables and are not
published here:

- Generated configs, budget JSON/TSV files, data/dump directories, checkpoints,
  logs, plots, spreadsheets, and dated GPU launch/resume scripts.
- Conformer experiments and official-FLEURS-like CTC/InterCTC variants.
- MMS-300M, four/six-layer comparison models, LID Transformer, and
  MfaConformer experiments.
- First-decoder-token forced top-2 diagnostics.
- Calibration experiments (prior bias, class bias, vector scaling, and
  class-specific thresholds).
- Attention, embedding, occlusion, prototype-geometry, and sub-center
  exploratory visualization scripts.
- CSDIALOG, Miami-Bangor, SEAME, CAFE, and SADiLaR preparation experiments.
- Test-only/all-candidate atomic pair labels.
- Historical v9/v10 handoff trees and one-off repair scripts.

No files under `egs2/ml_superb2` are part of this branch's CS-LID change.  The
branch is based directly on current upstream ESPnet and is pushed only to the
author's fork; no upstream pull request is created.
