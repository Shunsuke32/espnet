# LID classifiers

The historical LID1, LID2 and LID3 systems share this recipe directory. Model
construction and target preprocessing use `espnet2.tasks.lid.LIDTask` and the
standard LID template. Select the system with `--lid_config`:

| Config in `conf/` | Historical name | Target | Training sampler / epochs |
|---|---|---|---|
| `train_fleurs_lid_mms_ecapa.yaml` | LID1 | Single language or atomic pair class | category / catbel, 30 |
| `train_lidseq_mms_ecapa_softtarget.yaml` | LID2 | One-hot or 0.5/0.5 distribution; KL | sequence / sorted, 15 |
| `train_mms_ecapa_multilabel_aam_bce_posw50.yaml` | LID3 AAM | One-hot or two-hot; BCE with positive weight 50 | category / catbel, 30 |

All three use unfrozen MMS-1B, the learned S3PRL layer mixture, utterance MVN,
ECAPA-TDNN, channel-attentive statistics pooling, and the 192-dimensional
RawNet3 projector. The hard, KL and AAM+BCE heads have three sub-centers per
class, margin 0.5, scale 30 and Inter-TopK negative margins. The old plain
Linear+BCE ablation is not included in this recipe.

## Data profiles

`--profile fleurs_only`, `mixed` and `csall` choose FLEURS,
FLEURS+CS-FLEURS and FLEURS+CS-FLEURS+CS-YODAS respectively. Preparation writes
both ordinary-language and atomic-pair views. The hard config reads
`utt2lang`; the other configs read `utt2langs`. For a mixed hard model,
`ara-eng` is one class; `eng-ara` is not an additional class. Evaluation-only
pair classes are never added to its output inventory.

Historical defaults (global batch / accumulation; total effective batch 32):

| System | FLEURS-only | FLEURS+CS-FLEURS | CS-all |
|---|---|---|---|
| LID1 hard | 8 / 4 | 8 / 4 | 4 / 8 |
| LID2 KL | 8 / 4 (no selected historical run) | 8 / 4 | 8 / 4 |
| LID3 AAM+BCE | 16 / 2 | 4 / 8 | 8 / 4 |

Mixed/CS-all iterations per epoch are respectively 3880/8240 for LID1,
7760/8240 for LID2 (15 epochs), and 7760/4120 for LID3 (30 epochs).
LID1/LID3 use 30 epochs. All have approximately three nominal passes; the
wrapper recalculates iterations if the selected batch or data count changes.

Prepare data once using `local/data.sh`, with the download roots described in
the parent README. To use existing prepared views, start at Stage 3.

```bash
# Example workflow; choose the exact historical config/budget before a rerun.
CUDA_VISIBLE_DEVICES=0,1 ./run.sh \
  --profile mixed --stage 3 --stop_stage 5 --ngpu 2 \
  --lid_config conf/train_lidseq_mms_ecapa_softtarget.yaml \
  --train_batch_size 8 --accum_grad 4
```

Stage 1 prepares manifests, Stage 2 handles optional speed perturbation,
Stage 3 formats audio, Stage 4 collects statistics and Stage 5 trains.
The default stats directory is specific to the experiment tag, so FLEURS,
mixed and CS-all cannot accidentally share incompatible speech shapes.

Historical data membership is bound to the selected model/profile:

- Hard/AAM+BCE FLEURS-only or mixed uses `raw`. After Stage 3, a manifest-only
  duration view is written under `dump/old_duration/raw/<train-or-valid-set>`.
  It uses actual formatted sample counts, with strict `1s < length < 30s`.
  The original source and formatted manifests/audio are left unchanged.
- Sorted KL mixed uses `raw_copy`, preserving its historical direct-data
  membership. All CS-all profiles also use `raw_copy`. These use the original
  six-decimal metadata duration converted to integer 16-kHz samples, not a new
  waveform-derived selection. `sample_count_policy.json` records the policy.
- Train/valid source caps remain 30s for FLEURS/CS-FLEURS and 70s for CS-YODAS.
  Test data are not filtered by this training-view step.

This distinction matters for one real Persian FLEURS recording: its metadata
says 1.0230625s, but its waveform is 0.9590625s. Historical raw runs excluded
it; historical raw_copy runs retained it. A uniform new waveform filter would
silently change the original CS-all experiment. The recipe does not hard-code
an exclusion for that utterance; it reproduces the relevant length policy.
Use `--dumpdir` to choose another root; the same relative layout applies.

The wrapper computes a global batch budget from the prepared training count.
`batch_size * accum_grad` is 32; GPU count does not multiply it. ECAPA training
requires at least two utterances per GPU because of BatchNorm. The sorted KL
15-epoch schedule preserves the update budget of the 30-epoch version by
doubling iterations per validation epoch. Category sampling can omit or repeat
examples, so its nominal three-pass budget is not a guarantee that every
unique recording is sampled exactly three times.

Historical saved configs remain the authority for exact reruns, including
sampler, scheduler, number of updates and evaluation checkpoint. A newly
computed budget is not automatically identical to every historical experiment.
For an existing experiment, explicitly supply its `--lid_tag`, `--lid_stats_dir`
and `--dumpdir`; the new defaults do not discover legacy directory names or
modify the live historical experiments. Resume only with the matching
source/formatted inputs and duration policy. Filtered views are reused only
when their recorded input identities still match.
Use its confirmed config with `--auto_training_budget false
--enforce_training_policy false` when preserving a historical schedule rather
than generating a fresh one. Starting after Stage 1 with missing manifests
fails instead of silently downloading or regenerating training data.

## Evaluation

Use `local/evaluate.sh` after Stage 5. It saves scores once and then computes
top-k and threshold results. Ordinary language heads use top-1 on FLEURS and
top-2 on CS data. Atomic-pair classifiers take one class and expand a predicted
pair for unordered language-set evaluation.

KL and hard cosine heads use softmax; BCE heads use sigmoid. Inference never
applies a ground-truth-dependent angular margin. Threshold selection on a test
set is an exploratory curve, not a held-out calibration procedure.

See `local/evaluate.sh --help` for the complete invocation. For an exact
comparison, pass the actual training manifest/inventory and an explicit model
checkpoint. `checkpoint.pth` contains the last training state, which is not
necessarily the validation-best model.
For new hard/AAM raw runs, pass `dump/old_duration/raw/<train-set>` as
`--train_data_dir`; for raw_copy runs, pass `dump/raw_copy/<train-set>`.
Test inputs remain under `dump/raw` or `dump/raw_copy`, respectively.
