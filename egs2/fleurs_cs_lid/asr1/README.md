# ASR-style language-set generator

Targets are canonical one- or two-token sequences such as `<eng>` and
`<ara> <eng>`.  The paper model uses MMS-1B S3PRL features, learned multi-layer
weighted sum, utterance MVN, a 24-layer Transformer encoder, a four-layer
Transformer decoder, and attention loss only (`ctc_weight: 0`).
The encoder/decoder width is 256, with 4 attention heads and FFN width 1024.
Label smoothing remains 0.1, matching the selected historical d256 runs.

## Training Phases

`run.sh --train_mode mixed|csall --training_phase frozen|unfrozen` selects:

| Data / phase | Config in `conf/` |
|---|---|
| Mixed frozen | `train_lidseq_mms_transformer24_order_insensitive_min.yaml` |
| Mixed unfrozen | `train_lidseq_mms_transformer24_order_insensitive_min_unfrozen_lr5e6.yaml` |
| CS-all frozen | `train_lidseq_mms_transformer24_order_insensitive_min_csall.yaml` |
| CS-all unfrozen | `train_lidseq_mms_transformer24_order_insensitive_min_csall_unfrozen_lr5e6.yaml` |

Mixed defaults to batch 8 / accumulation 4; CS-all to 4 / 8 and `raw_copy`.
The global effective batch is 32 regardless of GPU count. Mixed frozen uses
10 validation epochs, about one pass, and LR 1e-3. CS-all frozen uses 30 epochs,
about three passes, and LR 1e-3. Both unfrozen phases use 30 epochs, about three
passes, and LR 5e-6. `run.sh` recalculates iterations and warmup from the data
count; see the parent README for the historical counts and launch examples.

A new unfrozen phase requires an explicit `--pretrained_model` pointing to
model-only weights from the chosen frozen epoch. There is no default epoch or
automatic best-checkpoint selection. Use a new `--asr_tag`; do not resume the
frozen experiment for this low-LR second phase. The parent `checkpoint.pth`
contains optimizer/reporter state and is not accepted as model-only weights.
If a best-checkpoint symlink is explicitly supplied, it is resolved to its
numbered target at launch so a later symlink update cannot change that choice.

To resume the same phase, use the same mode, phase, batch, tag and generated
config, with `--stage 11 --stop_stage 11 --resume true` and no
`--pretrained_model`. Missing data or checkpoints fail instead of regenerating
them. The wrappers skip downloading by default. No training command in the
CPU review runs MMS or uses a GPU.
Training overrides belong in the explicit YAML or wrapper options, not
`--asr_args`, which is rejected for training to prevent a second unchecked
config, output directory, loss, or scheduler from replacing the validated run.

## Loss and Evaluation

The canonical configuration is:

```yaml
model_conf:
  pit_loss: true
  pit_loss_reduction: min
```

This fork's permutation-invariant training (PIT) implementation supports one
or two language tokens and an attention-only decoder, not general arbitrary-
length PIT. The old `lidseq_order_insensitive_loss` and
`lidseq_order_insensitive_reduction` keys remain accepted for historical saved
configs. Conflicting old/new values are rejected. Disabled PIT is still the
default for ordinary ESPnet ASR. The rename does not change the loss formula.

The unordered loss evaluates the original and swapped two-token targets with
teacher forcing and takes the lower loss independently for each utterance.
Single-token targets are unchanged.  The normal training `valid.acc` remains
order-sensitive; final paper accuracy comes from `local/score_lidseq.py`.

`run.sh --stage 12 --stop_stage 13` decodes and writes for each set:

- `lidseq_score.json` with sequence and unordered exact accuracy.
- `lidseq_details.tsv` with per-utterance sequence/set/length decisions.

To score an existing decode directory:

```bash
./local/score_lidseq_decodes.sh \
  --decode_dir exp/asr_<tag>/<inference-tag> \
  --test_sets "test_fleurs_lid test_cs_read_test test_cs_xtts_test1 test_cs_xtts_test2 test_cs_mms_test"
```

The second unfrozen phase uses `--pretrained_model`; this initializes model
weights only and deliberately restarts Adam and WarmupLR.  `--resume true` is
for resuming the same phase's `checkpoint.pth`, including optimizer/scheduler
state, and is not equivalent.
CS-FLEURS all is aggregated from the four disjoint CS subset scores without
another inference pass. Use an explicit `--inference_asr_model` for a chosen
historical checkpoint. Ordered scores can still be reported, but no
order-sensitive training recipe is included.
