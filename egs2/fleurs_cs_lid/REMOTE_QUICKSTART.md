# Trial on another server

Use this reviewed source tree, not an old `cs-lid` checkout with only its YAMLs
replaced. The ASR PIT option and the dense-target LID heads need the matching
`espnet2` implementation. No upstream pull request is needed.

Clone the dedicated branch from the research fork:

```bash
git clone --branch cs-lid-repro --single-branch https://github.com/Shunsuke32/espnet.git
cd espnet
git rev-parse HEAD  # Record this commit for the experiment.
```

The branch contains code, configs and tests, not corpora or checkpoints.
Existing experiments on the older `cs-lid` branch are not modified.

## Environment and existing inputs

Activate the other server's ESPnet Python environment. Follow
`SOFTWARE_VERSIONS.md` for dependencies; the current CPU checks reuse an
existing environment and do not establish a clean dependency installation.
From the checkout root, expose this checkout rather than another ESPnet:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import espnet2; print(espnet2.__file__)'
```

Set the following to existing readable inputs. Do not point output directories
at the corpora. No fresh dataset download is needed for this route.

```bash
export FLEURS_TSV_ROOT=/mounted/datasets/fleurs/all_materialized
export FLEURS_AUDIO_ROOT="$FLEURS_TSV_ROOT/audio"
export CS_FLEURS_ROOT=/mounted/datasets/cs-fleurs
export CS_YODAS_ROOT=/mounted/datasets/cs-yodas
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

`FLEURS_TSV_ROOT` must contain `train.tsv`, `dev.tsv`, and `test.tsv`.
`FLEURS_AUDIO_ROOT` contains `<raw-language>/<train|validation|test>/<file>`.
The explicit audio root remaps old absolute TSV paths without editing those
TSVs or changing the historical utterance IDs. It does not copy audio and
never falls back to an old path when a requested relocated file is missing.
Relative TSV paths are resolved against the TSV directory. If two audio roots
are plausible, the explicit root is required.
For roots containing spaces, use the quoted `local/data.sh` command below,
not wrapper Stage 1. The wrappers reject whitespace there because ESPnet's
template serializes data options as a shell word list; after direct data
preparation, start `run.sh` at Stage 3.

CS-FLEURS needs the original `*/metadata.jsonl` and audio, including XTTS train
and the four test subsets. CS-YODAS needs `metadata/{ara,cmn,fra,hin,jpn,rus}.jsonl`
and `audio/` with the original `wav_path` layout. The six metadata hashes must
match the recorded revision. Archives alone are not the extracted audio tree.

The MMS-1B snapshot must also already be cached or available as a local model
directory. See `SOFTWARE_VERSIONS.md` to pin it; dataset preparation does not
download model weights. A missing cache is an error in offline mode.

## Data first, no training

In a fresh checkout, run the same entrypoint from each recipe you will use:

```bash
cd egs2/fleurs_cs_lid/asr1  # or ../lid1 for all three classifier heads
./local/data.sh --skip_fleurs_download true \
  --fleurs_tsv_root "$FLEURS_TSV_ROOT" \
  --fleurs_audio_root "$FLEURS_AUDIO_ROOT" \
  --cs_root "$CS_FLEURS_ROOT" --cs_yodas_root "$CS_YODAS_ROOT" \
  --duration_missing_policy error
```

This writes manifests in the recipe's new `data/`, not audio copies. Omit
`--cs_yodas_root` for a mixed-only trial. The policy is the old CS-FLEURS
global-hash 98:2 split, not class-wise 9:1. Train/valid caps are 30 seconds for
FLEURS/CS-FLEURS and 70 seconds for CS-YODAS (exclusive upper limits); test
manifests are not duration-filtered. Historical raw/raw_copy boundary handling
is documented in `README.md`.

## Training and evaluation stages

| Operation | ASR | Unified LID |
|---|---:|---:|
| Data preparation | 1 | 1 |
| Audio formatting | 3 | 3 |
| Length filtering | 4, plus source-specific preparation | 1 and post-3 recipe view |
| Statistics | 10 | 4 |
| Training | 11 | 5 |
| Evaluation | 12-13 | `local/evaluate.sh` |

Do not reuse a legacy local LID `--stage 6` launch: it meant training in the
older renumbered template, not in this tree. Unchanged prepared inputs can be
reused by skipping data preparation. Missing inputs never trigger a download.

The ASR phase examples and exact budget table are in `README.md`. Select
`--train_mode mixed|csall` and `--training_phase frozen|unfrozen`; a new unfrozen
phase needs an explicitly chosen model-only frozen checkpoint and a new tag.
`checkpoint.pth` is for resuming the same phase, not for initializing a new
phase or replacing a missing best checkpoint. Keep global batch times
accumulation at 32; GPU count does not multiply the global batch.

New ASR YAMLs use `model_conf.pit_loss: true` and
`model_conf.pit_loss_reduction: min`. Historical `lidseq_order_insensitive_*`
keys remain accepted; contradictory aliases fail. PIT here supports only one
or two language tokens, attention loss only. It does not change logged token
accuracy into unordered set accuracy.

For LID, select the hard, KL, or AAM+BCE YAML listed in `lid1/README.md`.
Do not replace their samplers, label files, or epoch schedules with one shared
default. Model publication and its native ESPnet stages are documented in
`MODEL_PUBLICATION.md`.

## Offline CPU smoke check

From the checkout root, using the active environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 test/egs2/fleurs_cs_lid/test_asr_wrapper_cpu_workflow.py \
  --work-dir /tmp/cs-lid-asr-smoke-new
```

The output directory must be new. This uses synthetic audio and a tiny frontend
to check formatting, statistics, training, fresh-state phase initialization,
decoding, and scoring without downloading MMS. Passing it is not a claim of
full-size MMS training accuracy or GPU memory fit.
