# ASR-style language-set generator

Targets are canonical one- or two-token sequences such as `<eng>` and
`<ara> <eng>`.  The paper model uses MMS-1B S3PRL features, learned multi-layer
weighted sum, utterance MVN, a 24-layer Transformer encoder, a four-layer
Transformer decoder, and attention loss only (`ctc_weight: 0`).

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
  --test_sets "test_fleurs_lid test_cs_read_test test_cs_xtts_test1 test_cs_xtts_test2 test_cs_mms_test test_cs_all"
```

The second unfrozen phase uses `--pretrained_model`; this initializes model
weights only and deliberately restarts Adam and WarmupLR.  `--resume true` is
for resuming the same phase's `checkpoint.pth`, including optimizer/scheduler
state, and is not equivalent.
