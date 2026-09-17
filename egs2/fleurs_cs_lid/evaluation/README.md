# Saved-score LID evaluation

From `lid1`, with ESPnet's Python environment activated:

```bash
local/evaluate.sh \
  --model_file "$EXPERIMENT/$CHECKPOINT" \
  --config_file "$EXPERIMENT/config.yaml" \
  --train_data_dir "$FROZEN_TRAIN_DATA" \
  --test_sets "test_fleurs_lid test_cs_all" \
  --data_dir "$FORMATTED_DATA" \
  --output_dir "$EVALUATION_OUTPUT" \
  --ngpu 0 --batch_size 1
```

Select the experiment, saved config, and exact checkpoint explicitly. This CLI
does not choose historical experiments, change splits, train, or replace a
requested checkpoint with a best/latest file. Use only the confirmed OLD 98:2
training manifests for the paper workflow. `checkpoint.pth` is supported by
extracting its `model` dictionary and loading strictly. It is the last saved
training state, not necessarily the validation-best model.

Loading always tries `weights_only=True` first. Current ESPnet reporter state
serializes to safe native types, but older snapshots may contain NumPy objects,
`timedelta`, or custom reporter classes. If the safe loader refuses such objects,
the command fails by default. **Only for a checkpoint you personally trust**, add
`--trusted_checkpoint` to permit retrying with `weights_only=False`. Pickle can
execute arbitrary code during this retry. There is no automatic unsafe fallback
or environment-variable opt-in. A used fallback is logged and marked
`:trusted_pickle` in `metadata.checkpoint_format`. Structural/state-key errors
still fail, without partially initialized parameters.

`--train_data_dir` must contain the checkpoint-matching `lang2utt` and
`utt2langs`, not a newly generated or test inventory. An experiment-side
`lang2utt` snapshot beside the config takes precedence; supplied training and
available config inventories must agree in exact class order. If no snapshot
exists, the CLI warns and freezes the explicitly supplied training inventory.
The original historical class order cannot be inferred from checkpoint tensors.
Training pair views may store a single atomic class such as `ara-eng` in
`utt2langs`; only this training inventory is expanded into canonical individual
language sets for seen/unseen grouping. Ordinary test references still require
one or two separate language labels and reject atomic tokens.
Inputs are snapshotted under the evaluation output, never written back to the
experiment or data directories.
An obsolete absolute `lang2utt` path in a moved config need not exist: model
construction does not read it, and inference disables target preprocessing and
uses the evaluation snapshot in memory. An available old inventory is still
checked for matching class order. Other assets referenced by the config, such as
frontend weights or normalization statistics, must remain accessible separately.

Each test directory needs formatted, utterance-level `wav.scp` and `utt2langs`.
Piped/segmented audio must first pass ESPnet audio formatting. Relative audio
paths are resolved against the invocation directory when present there, and
otherwise against their manifest directory. Batch size 1 is the conservative
default for variable-length speech.

## Outputs

- `scores.npz`: unique utterance IDs, frozen labels, logits, float64 probabilities,
  and JSON metadata. Includes checkpoint/config/training/input hashes and the
  activation and training loss scale. No fitted probability transforms are used.
- `<set>.details.tsv`: per-utterance references, predictions, and exact decisions.
- `<set>.json` and `<set>.summary.tsv`: exact accuracy, reference cardinality,
  language-set breakdown, and seen/unseen **pairs** from frozen training targets.
- KL/BCE: `<set>.threshold_sweep.tsv` plus `softmax_threshold_sweep.png`,
  `sigmoid_threshold_sweep.png`, or `logit_threshold_sweep.png` as applicable.
- `evaluation.json`: provenance, scored views, exclusions, and overall results.

Hard individual-language heads and KL/BCE heads use top1 for one-language
FLEURS references and top2 for two-language CS references. A hard head containing
atomic pair classes always selects one class and expands it, including on
FLEURS. Rankings use logits with class-order tie breaking, avoiding sigmoid
saturation. AAM/KL inference uses scaled cosine with sub-center max pooling and
no target-dependent margins. Plain BCE retains Linear bias. Plot baselines use
these exact same top-k decisions.

KL keeps softmax thresholds 0..1 by 0.01 and raw-logit thresholds 0..30 by 0.5.
BCE keeps 0, 0.001, 0.002, 0.005 and 0.01..1 by 0.01. `--thresholds` and
`--logit_thresholds` override these grids. Thresholding uses `>=`, allows empty
or oversized sets, and never forces a top-k fallback or selects a test optimum.

When all four standard CS-FLEURS subsets are selected, `test_cs_all` is generated
as their union. Requesting `test_cs_all` expands to those subsets when all four
directories exist. An existing aggregate's references and IDs must match the
union; its separately formatted audio is not decoded. No CS aggregate is inferred
from a partial collection of subsets.

## Reuse and Alignment

Repeating the command reuses matching `scores.npz`, without loading the model.
`--score_only` additionally forbids inference when the cache is absent. Reuse
checks model/config hashes, class order, training manifests, input audio manifest,
batch size, score finiteness, and probability consistency. Changed provenance
requires a new output directory. Reference views are revalidated on every run.

Missing, extra, and duplicate IDs fail. The only exclusion mechanism is an explicit
global `--exclusions exclusions.tsv`, with a header `utt<TAB>reason` and a nonempty
reason per ID. Excluded IDs are removed from both audio and references, unused
exclusions fail, and every reason is recorded in `evaluation.json`.

The details files also work with `summarize_details.py` and
`paired_significance.py`. Set comparisons ignore reference token order;
`seq_exact` comparisons retain order. The ASR decoder/scorer remains separate.

## ASR Aggregate Details

The LID CLI's automatic union does not merge ASR decode/detail directories.
For existing ASR subset scores, generate the aggregate without decoding again.
Run from `lid1`:

```bash
python3 ../asr1/local/score_lidseq.py \
  --ref "$DATA/test_cs_all/utt2langs" \
  --details_inputs \
    "$ASR_DECODE_DIR/test_cs_read_test/lidseq_details.tsv" \
    "$ASR_DECODE_DIR/test_cs_xtts_test1/lidseq_details.tsv" \
    "$ASR_DECODE_DIR/test_cs_xtts_test2/lidseq_details.tsv" \
    "$ASR_DECODE_DIR/test_cs_mms_test/lidseq_details.tsv" \
  --out "$EVALUATION_OUTPUT/asr_cs_all.json" \
  --details_out "$ASR_CS_ALL_DETAILS"
```

Then summarize the resulting details:

```bash
python3 ../evaluation/summarize_details.py \
  --details "$ASR_CS_ALL_DETAILS" \
  --ref_utt2langs "$DATA/test_cs_all/utt2langs" \
  --train_utt2langs "$FROZEN_TRAIN_DATA/utt2langs" \
  --correctness_column set_exact \
  --output_prefix "$EVALUATION_OUTPUT/asr_cs_all_unordered"
```

Repeat with `--correctness_column seq_exact` and a distinct output prefix for
ordered accuracy. The merger must retain one header and every utterance exactly
once; the summary rejects duplicate IDs, missing/extra references, and mismatched
reference language sets. `pair_seen_status` reports only CS pairs; the legacy
`seen_status` also includes single-language sets. Aggregate generation preserves
the original hypotheses and recalculates both ordered and unordered decisions;
it does not just concatenate previous correctness columns.

## Common Training Partition

The unified LID CLI defines pair seen/unseen from the model's actual supplied
training references. For a FLEURS-only model, every CS pair is therefore unseen.
To compare that baseline on a mixed model's common seen/unseen partition, run
`summarize_details.py` with the mixed model's `--train_utt2langs` explicitly and
report that partition source. Do not mislabel it as FLEURS-only training
exposure, and do not change the frozen class inventory used for inference.
