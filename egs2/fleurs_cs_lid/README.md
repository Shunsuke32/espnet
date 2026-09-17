# FLEURS / CS-FLEURS language identification

This recipe contains the code used for the paper's FLEURS, CS-FLEURS, and
six-pair CS-YODAS experiments.  It follows the normal ESPnet `run.sh` + template
layout and does not contain data, checkpoints, generated configs, dated launch
scripts, or result workbooks.

## Systems

| directory | target | model / loss |
|---|---|---|
| `lid1` | one class; a CS pair is an atomic class such as `ara-eng` | MMS-1B + ECAPA-TDNN + AAMSoftmax/Sub-center/Inter-TopK |
| `lid1` (LID2 config) | one-hot for FLEURS, `0.5/0.5` for CS | same backend + soft-target KL |
| `lid1` (LID3 config) | one-hot or two-hot independent labels | AAM/Sub-center/Inter-TopK+BCE, positive weight 50 |
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

For a source-only trial on a different server, start with
[REMOTE_QUICKSTART.md](REMOTE_QUICKSTART.md). It includes audio relocation,
offline data preparation, and the standard ASR/LID stage numbers.

Use existing downloaded datasets whenever available. The training wrappers
default to `--skip_fleurs_download true`; missing inputs fail instead of being
downloaded. The current review performs **no downloads** and creates no new
full audio dump. All roots are configurable; no `/home` or `/data3` location
is required.

From a fresh code checkout, rebuild manifests from the original, already
downloaded FLEURS TSV/audio and CS dataset metadata/audio:

```bash
cd egs2/fleurs_cs_lid/lid1
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
./local/data.sh --skip_fleurs_download true \
  --fleurs_tsv_root "${FLEURS_TSV_ROOT:?set the existing TSV directory}" \
  --fleurs_audio_root "${FLEURS_AUDIO_ROOT:?set the materialized audio root}" \
  --cs_root "${CS_FLEURS_ROOT:?set the existing CS-FLEURS directory}" \
  --cs_yodas_root "${CS_YODAS_ROOT:?set the existing CS-YODAS directory}"
```

This creates new manifests in the checkout, not in the source corpora. It does
not require copying historical `data/` or `dump/`. The same `local/data.sh`
entrypoint is available in `asr1`. Omit `--cs_yodas_root` for mixed-only data.
FLEURS TSVs are the acquisition output; CS manifests are rebuilt from original
JSONL metadata. The CPU review also tests FLEURS parquet-to-TSV materialization
with synthetic inputs; it does not redownload the real parquet files.
The explicit FLEURS audio root remaps old absolute TSV paths to the selected
materialized tree without changing source files or historical utterance IDs.

Only on a different server with no source data and enough disk space, use the
following **optional online acquisition** first, from `egs2/fleurs_cs_lid`:

```bash
./local/download_cs_fleurs.sh --root downloads/cs-fleurs
./local/download_extract_cs_yodas.sh --root downloads/cs-yodas

cd lid1
./local/data.sh \
  --skip_fleurs_download false \
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
- No new class-wise `9:1` split is included. Historical runs that used that
  split are not reproduced by substituting the old `98:2` split.
- CS-YODAS is split per base language by `video_id`, 80/10/10, with seed
  `cs-yodas-v1`; a video cannot cross splits.
- FLEURS and CS-FLEURS train/valid keep `1.0 <= duration < 30.0` seconds.
- CS-YODAS train/valid keep duration `< 70.0` seconds (70.0 itself is excluded).
- Test sets are not duration-filtered during preparation.

`data/local/` contains split, duration, overlap, label-map, and inventory audit
files.  `local/verify_lid_data.sh` must pass before training.

## Training

The wrappers generate `conf/generated/` from the actual post-filter training
count.  ESPnet's `batch_size` is global across DDP workers.  Every paper run
keeps `batch_size * accum_grad = 32`; `num_iters_per_epoch` is chosen so total
exposure is approximately three or five nominal passes. Category-balanced
sampling is not an exhaustive traversal; some records may repeat or be omitted.
The ASR phase defaults below follow the confirmed d256 unordered family.
The initial frozen checkpoint for a new unfrozen phase remains a user choice;
the recipe never silently chooses epoch 9, epoch 16, latest, or best weights.
LID settings and the exact historical source inventories are described in
the per-directory READMEs. Synthetic CPU checks are not a claim of identical
GPU training results.

Representative commands, after data preparation:

On a new server, start statistics collection with `--nj 1` and increase only
after checking RAM: independent MMS statistics jobs can each construct a full
frontend. This preparation parallelism is separate from training batch size
and does not multiply the optimization budget.

```bash
# LID1 FLEURS-only (3 nominal passes / 30 epochs)
cd lid1
CUDA_VISIBLE_DEVICES=0,1 ./run.sh --stage 3 --stop_stage 5 --ngpu 2 \
  --profile fleurs_only \
  --train_batch_size 8 --accum_grad 4

# LID1 FLEURS+CS-FLEURS atomic-pair classifier
CUDA_VISIBLE_DEVICES=0,1 ./run.sh --stage 3 --stop_stage 5 --ngpu 2 \
  --profile mixed \
  --train_batch_size 8 --accum_grad 4

# LID2 final soft-target KL model (sorted, 3 passes / 15 epochs)
CUDA_VISIBLE_DEVICES=0,1 ./run.sh --profile mixed --ngpu 2 \
  --lid_config conf/train_lidseq_mms_ecapa_softtarget.yaml \
  --train_batch_size 8 --accum_grad 4

# LID3 final AAM/Sub-center+BCE model
CUDA_VISIBLE_DEVICES=0,1 ./run.sh --profile mixed --ngpu 2 \
  --lid_config conf/train_mms_ecapa_multilabel_aam_bce_posw50.yaml \
  --train_batch_size 4 --accum_grad 8

# LID3 FLEURS-only AAM/Sub-center+BCE model
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --profile fleurs_only --ngpu 4 \
  --lid_config conf/train_mms_ecapa_multilabel_aam_bce_posw50.yaml \
  --train_batch_size 16 --accum_grad 2

# LID3 CS-all profile, including CS-YODAS training data shorter than 70 seconds
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --profile csall --ngpu 4 \
  --lid_config conf/train_mms_ecapa_multilabel_aam_bce_posw50.yaml \
  --train_batch_size 8 --accum_grad 4
```

ASR-style runs use a frozen phase followed by a new low-LR unfrozen run.
Optimizer and scheduler state are intentionally restarted in phase 2.
Both stages keep the learned MMS layer mixture trainable; only
`frontend.upstream` is frozen in phase 1. Prepare the corresponding source
manifests first, using the offline command above from `asr1`.

| Data | Phase | LR | Batch / accum | Iters/epoch | Epochs | Warmup updates |
|---|---|---:|---|---:|---:|---:|
| FLEURS+CS-FLEURS | frozen | 1e-3 | 8 / 4 | 3880 | 10 | 2910 |
| FLEURS+CS-FLEURS | unfrozen | 5e-6 | 8 / 4 | 3880 | 30 | 2910 |
| CS-all | frozen | 1e-3 | 4 / 8 | 8240 | 30 | 3090 |
| CS-all | unfrozen | 5e-6 | 4 / 8 | 8240 | 30 | 3090 |

The mixed frozen prefix is approximately one pass, matching the first ten
epochs of the original parent run. Each other phase is approximately three
passes. Counts are regenerated if the global batch or actual inventory changes.

```bash
cd ../asr1

# Mixed frozen prefix; data already prepared, no acquisition.
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 3 --stop_stage 11 --ngpu 4 --nj 1 \
  --train_mode mixed --training_phase frozen --asr_tag mixed_frozen

# Mixed phase 2: set MIXED_FROZEN_WEIGHTS to your explicitly chosen epoch file.
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 10 --stop_stage 11 --ngpu 4 --nj 1 \
  --train_mode mixed --training_phase unfrozen --asr_tag mixed_unfrozen \
  --pretrained_model "${MIXED_FROZEN_WEIGHTS:?select frozen model-only weights}"

# CS-all frozen; raw_copy references existing verified 16-kHz audio.
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 3 --stop_stage 11 --ngpu 4 --nj 1 \
  --train_mode csall --training_phase frozen --asr_tag csall_frozen

# CS-all phase 2; retain the same prepared data/dump as its frozen phase.
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 10 --stop_stage 11 --ngpu 4 --nj 1 \
  --train_mode csall --training_phase unfrozen --asr_tag csall_unfrozen \
  --pretrained_model "${CSALL_FROZEN_WEIGHTS:?select frozen model-only weights}"

```

Resume an interrupted phase with its original options/tag and
`--stage 11 --stop_stage 11 --resume true`, without `--pretrained_model`.
New-phase launches reject existing experiment directories; resume checks the
saved config and preserves optimizer/scheduler state. Do not overwrite a
frozen run with an unfrozen config. No early stopping is used. The recipes do
not include order-sensitive, d512/new-9:1, or old plain Linear+BCE experiments.

## Evaluation

ASR decoding and unordered/ordered scoring are stages 12-13 of `asr1/run.sh`.
The wrapper defaults to `valid.loss.best.pth`. Use an explicitly confirmed
`--inference_asr_model` to reproduce a historical selection. A best-checkpoint
symlink can change when training continues; an extracted last checkpoint must
never be substituted for a missing best checkpoint.
All LID variants use `lid1/local/evaluate.sh` to save raw logits once, then
recompute top-k and threshold metrics without another model forward pass.
KL/hard-language logits use softmax; BCE logits use sigmoid, not softmax.
Atomic-pair classifiers use top-1 class prediction expanded into two languages.
See [lid1/README.md](lid1/README.md) for the evaluation interface. CS-FLEURS
subsets are inferred once; CS-all is their disjoint aggregate, not another
inference pass. Threshold sweeps on test data are diagnostic curves, not
validation-selected operating points. No calibration is included.

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
paired-bootstrap 95% confidence interval. Historical values are recorded
in [RESULTS.md](RESULTS.md), which is a historical reference, not a verified
fresh-server reproduction report.

## Stage design

The official LID template uses Stage 1 data preparation, Stage 2 speed
perturbation, Stage 3 formatting, Stage 4 statistics, Stage 5 training, and
later inference/scoring stages.  Unlike the ASR template's Stage 4, the LID
template has no dedicated remove-long/short-data stage.  An older local version
inserted a new duration-filter stage and renumbered every later stage.  That is
not retained: it breaks normal ESPnet stage semantics and resume commands.

Duration filtering starts in deterministic source-data preparation inside
Stage 1. After Stage 3, LID raw runs build a separate actual-length manifest
view before Stage 4 statistics; historical raw_copy routes preserve their
metadata-based membership. ASR retains its standard Stage 4 guard. Source
manifests have the same source-specific policy, but historical final raw and
raw_copy training membership can differ at duration boundaries. Tests remain
unfiltered. The generic template extensions are the
default-preserving `--lid_label_file` option (`utt2lang` by default), needed for
LID2/LID3's `utt2langs` targets, and an optional `--lid_stats_dir` so recipes can
keep statistics for different target inventories separate.

See [IMPLEMENTATION_SCOPE.md](IMPLEMENTATION_SCOPE.md) for every included and
excluded component.

## Offline CPU Checks

From the repository root, with its ESPnet environment active, run:

```bash
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
python3 test/egs2/fleurs_cs_lid/test_cpu_workflow.py --work-dir /tmp/lid-cpu-check
python3 test/egs2/fleurs_cs_lid/test_asr_cpu_workflow.py --work-dir /tmp/asr-cpu-check
python3 test/egs2/fleurs_cs_lid/test_asr_wrapper_cpu_workflow.py --work-dir /tmp/asr-wrapper-cpu-check
```

Use new, nonexistent output directories. Commands, intermediate data,
checkpoints, and `report.json` are retained there for review. The LID check
uses the recipe/template stages, tests the three selected target/head combinations, and
checks saved-logit rescoring. The ASR check exercises the actual task/trainer
and beam-inference APIs for both loss orderings; it does not cover all shell
stages. The additional ASR wrapper check covers raw/raw_copy formatting,
statistics, weights-only second-phase initialization, one update, decoding
and scoring through the real shell stages. The tests deliberately use
synthetic audio and a small frontend, not
MMS, so they test workflow correctness rather than published accuracy. YODAS
synthetic inputs exercise the preparation APIs, while its production CLI
continues to enforce the official metadata checksums.

On a read-only environment, point `NUMBA_CACHE_DIR` and `MPLCONFIGDIR` to
writable scratch directories. Unit tests live in `test/egs2/fleurs_cs_lid` and
the relevant `test/espnet2` modules. No calibration is required for any check.
