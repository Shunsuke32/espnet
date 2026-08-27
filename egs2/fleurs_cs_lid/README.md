# FLEURS / CS-FLEURS language identification

This recipe contains the code used for the paper's FLEURS, CS-FLEURS, and
six-pair CS-YODAS experiments.  It follows the normal ESPnet `run.sh` + template
layout and does not contain data, checkpoints, generated configs, dated launch
scripts, or result workbooks.

## Systems

| directory | target | model / loss |
|---|---|---|
| `lid1` | one class; a CS pair is an atomic class such as `ara-eng` | MMS-1B + ECAPA-TDNN + AAMSoftmax/Sub-center/Inter-TopK |
| `lid2` | one-hot for FLEURS, `0.5/0.5` for CS | same backend + soft-target KL |
| `lid3` | one-hot or two-hot independent labels | plain Linear+BCE or AAM/Sub-center/Inter-TopK+BCE |
| `asr1` | one or two generated language tokens | MMS-1B + 24-layer Transformer encoder + 4-layer decoder |

All language symbols are canonical ISO 639-3-style labels (`eng`, `ara`,
`cmn`, ...).  Raw FLEURS labels such as `en_us` and ASR prompts such as
`[en_us]` are rejected from `nlsyms.txt`.

## Environment

Start from this fork's `cs-lid` branch and follow ESPnet's normal installation:

```bash
cd tools
./setup_miniforge.sh miniforge espnet 3.10
make TH_VERSION=2.9.1
. ./activate_python.sh
cd ..
```

The reference run used Python 3.10.14, PyTorch/Torchaudio 2.9.1+cu126,
`datasets` 4.8.5, `soundfile` 0.13.1, `s3prl` 0.4.18, `pycountry` 26.2.16,
and SciPy 1.15.3.  See [SOFTWARE_VERSIONS.md](SOFTWARE_VERSIONS.md).
Data acquisition also requires `git-lfs`, `curl`, `tar`, and `sha256sum`.
Initialize Git LFS once with `git lfs install`; the CS-FLEURS downloader then
checks out the pinned dataset commit and materializes its LFS objects.

## Data acquisition

Run from `egs2/fleurs_cs_lid`.  All destinations are configurable; no `/home`
or `/data3` path is assumed.

```bash
./local/download_cs_fleurs.sh --root downloads/cs-fleurs
./local/download_extract_cs_yodas.sh --root downloads/cs-yodas

cd lid1
./local/data.sh \
  --fleurs_download_dir ../downloads/fleurs \
  --fleurs_cache_dir ../downloads/cache \
  --cs_root ../downloads/cs-fleurs \
  --cs_yodas_root ../downloads/cs-yodas
```

Pinned inputs:

- `google/fleurs`: `70bb2e84b976b7e960aa89f1c648e09c59f894dd`
- `byan/cs-fleurs`: `0cdbf166c5517ae4b6eb1c54248522eedec53017`
- `byan/cs-yodas`: `e51028041b403f63c99ac91a4af040e72d0cad0e`

The CS-YODAS downloader verifies the six metadata SHA-256 values before data
preparation.  It selects only `ara/cmn/fra/hin/jpn/rus + eng` records.

The FLEURS download materializes stable audio paths in addition to the
Hugging Face cache.  On the reference machine these occupied about 299 GB and
501 GB respectively at peak; CS-FLEURS occupied about 77 GB.  The cache may be
removed only after confirming every TSV `path` points into the materialized
audio tree.  The recipe never removes caches automatically.

## Data policy

- FLEURS uses the official train/validation/test split.
- The paper CS-FLEURS profile uses deterministic global-hash train/dev `98:2`
  (`--cs_split_mode global_hash --cs_dev_ratio 0.02`).
- The optional improved validation profile is class-wise `9:1`
  (`--cs_split_mode classwise_hash --cs_dev_ratio 0.1`); it is not the paper
  split and must be reported as a different experiment.
- CS-YODAS is split per base language by `video_id`, 80/10/10, with seed
  `cs-yodas-v1`; a video cannot cross splits.
- FLEURS and CS-FLEURS train/valid keep `1.0 <= duration < 30.0` seconds.
- CS-YODAS train/valid keep duration `< 70.0` seconds.
- Test sets are not duration-filtered during preparation.

`data/local/` contains split, duration, overlap, label-map, and inventory audit
files.  `local/verify_lid_data.sh` must pass before training.

## Training

The wrappers generate `conf/generated/` from the actual post-filter training
count.  ESPnet's `batch_size` is global across DDP workers.  Every paper run
keeps `batch_size * accum_grad = 32`; `num_iters_per_epoch` is chosen so total
exposure is approximately three or five complete passes.

Representative commands, after data preparation:

```bash
# LID1 FLEURS-only (3 passes / 30 epochs)
cd lid1
CUDA_VISIBLE_DEVICES=0,1 ./run.sh --stage 3 --stop_stage 5 --ngpu 2 \
  --train_set train_fleurs_lid --valid_set valid_fleurs_lid \
  --train_batch_size 8 --accum_grad 4

# LID1 FLEURS+CS-FLEURS atomic-pair classifier
CUDA_VISIBLE_DEVICES=0,1 ./run.sh --stage 3 --stop_stage 5 --ngpu 2 \
  --train_set train_lid_pair_cs --valid_set valid_lid_pair_cs \
  --train_batch_size 8 --accum_grad 4

# LID2 final soft-target KL model (sorted, 3 passes / 15 epochs)
cd ../lid2
CUDA_VISIBLE_DEVICES=0,1 ./run.sh --profile mixed --ngpu 2

# LID3 final AAM/Sub-center+BCE model
cd ../lid3
CUDA_VISIBLE_DEVICES=0,1 ./run.sh --profile mixed --ngpu 2 \
  --lid_config conf/train_mms_ecapa_multilabel_aam_bce_posw50.yaml \
  --train_batch_size 4 --accum_grad 8

# LID3 FLEURS-only AAM/Sub-center+BCE model
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --profile fleurs_only --ngpu 4 \
  --lid_config conf/train_mms_ecapa_multilabel_aam_bce_posw50.yaml \
  --train_batch_size 16 --accum_grad 2

# LID3 CS-all profile, including <=70-second CS-YODAS training data
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --profile csall --ngpu 4 \
  --lid_config conf/train_mms_ecapa_multilabel_aam_bce_posw50.yaml \
  --train_batch_size 8 --accum_grad 4
```

ASR-style runs use a frozen phase followed by a new low-LR unfrozen run whose
model weights are initialized from the stated frozen checkpoint.  Optimizer and
scheduler state are intentionally restarted in phase 2.

```bash
cd ../asr1

# Small unordered frozen prefix: 10 epochs / about 1 pass. Keep 9epoch.pth.
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 1 --stop_stage 11 --ngpu 4 \
  --train_mode mixed --asr_config conf/train_lidseq_mms_transformer24_order_insensitive_min.yaml \
  --train_batch_size 8 --accum_grad 4 \
  --target_passes 1 --budget_max_epoch 10 --warmup_ratio 0.3

# Small unordered phase 2, initialized from the historical frozen epoch 9.
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 10 --stop_stage 11 --ngpu 4 \
  --train_mode mixed \
  --asr_config conf/train_lidseq_mms_transformer24_order_insensitive_min_unfrozen_lr5e6.yaml \
  --pretrained_model exp/asr_<frozen-tag>/9epoch.pth \
  --train_batch_size 8 --accum_grad 4

# Large frozen model, 5 passes / 30 epochs.
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 1 --stop_stage 11 --ngpu 4 \
  --train_mode mixed --target_passes 5 \
  --asr_config conf/train_lidseq_mms_transformer24_d512_order_insensitive_min_frozen_lr1e_5.yaml \
  --train_batch_size 8 --accum_grad 4

# Large phase 2 from the frozen epoch-29 valid-loss-best weights.
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 10 --stop_stage 11 --ngpu 4 \
  --train_mode mixed --target_passes 5 \
  --asr_config conf/train_lidseq_mms_transformer24_d512_order_insensitive_min_unfrozen_lr1e_5.yaml \
  --pretrained_model exp/asr_<large-frozen-tag>/29epoch.pth \
  --train_batch_size 4 --accum_grad 8
```

The small frozen prefix uses a `0.3` warmup ratio so its 10-epoch schedule has
the same 2,910 warmup updates as the historical run from which `9epoch.pth` was
selected.  Small-model runs do not use early stopping.  The large configs retain
`patience: 10`, matching their saved training configs.

For the order-sensitive ablation, use `train_lidseq_mms_transformer24.yaml`
then initialize `train_lidseq_mms_transformer24_order_sensitive_unfrozen_lr5e6.yaml`
from the frozen valid-loss-best checkpoint.  For CS-all ASR, add
`--train_mode csall --cs_yodas_root ../downloads/cs-yodas` and use the same
frozen-to-unfrozen protocol.

## Evaluation

ASR decoding and unordered/ordered scoring are stages 12-13 of `asr1/run.sh`.
The wrapper defaults to `valid.loss.best.pth`.  To reproduce the reported
checkpoint choices exactly, evaluate epoch 15 for the small unordered phase-2
run and epoch 9 for the large phase-2 run.  The large run was initialized from
frozen epoch 29 and was stopped after phase-2 epoch 11; its reported
valid-loss-best checkpoint was epoch 9.  Use explicit `--inference_asr_model`
paths for these fixed historical selections rather than relying on a symlink
that can change if training continues.
The LID systems save raw logits once, then recompute top-k and threshold metrics
without another model forward pass.  See each local scorer's `--help`.

After scoring, generate seen/unseen and language-set tables with:

```bash
python3 evaluation/summarize_details.py \
  --details <system.details.tsv> \
  --ref_utt2langs lid1/data/<test-set>/utt2langs \
  --train_utt2langs lid1/data/train_lidseq/utt2langs \
  --output_prefix <output-prefix>
```

Use `train_lidseq_csall_yodas/utt2langs` for a CS-all model.  "Seen" means the
same unordered canonical language set occurs in that training manifest.

Paired significance on identical utterances:

```bash
python3 evaluation/paired_significance.py \
  --system_a <model-a.details.tsv> --system_a_name LID2 \
  --system_b <model-b.details.tsv> --system_b_name LID3 \
  --output <comparison.json>
```

This reports the requested paired t-test, exact McNemar test, and deterministic
paired-bootstrap 95% confidence interval.  Expected paper values are recorded
in [RESULTS.md](RESULTS.md).

## Stage design

The official LID template uses Stage 1 data preparation, Stage 2 speed
perturbation, Stage 3 formatting, Stage 4 statistics, Stage 5 training, and
later inference/scoring stages.  Unlike the ASR template's Stage 4, the LID
template has no dedicated remove-long/short-data stage.  An older local version
inserted a new duration-filter stage and renumbered every later stage.  That is
not retained: it breaks normal ESPnet stage semantics and resume commands.

Duration filtering is now deterministic source-data preparation inside the
existing Stage 1.  This also guarantees that ASR, LID1, LID2, and LID3 see the
same source-specific utterance inventory before ASR's common-bound guard is
applied.  Tests remain unfiltered.  The only generic template extension is the
default-preserving `--lid_label_file` option (`utt2lang` by default), needed for
LID2/LID3's `utt2langs` targets.

See [IMPLEMENTATION_SCOPE.md](IMPLEMENTATION_SCOPE.md) for every included and
excluded component.
