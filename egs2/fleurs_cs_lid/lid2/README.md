# LID2 soft-target classifier

LID2 keeps the LID1 MMS/ECAPA/AAM/Sub-center/Inter-TopK architecture.  Its only
target/loss change is one-hot targets for FLEURS, `0.5/0.5` distributions for
two-language speech, and `KL(target || softmax(logits))` with PyTorch
`batchmean` reduction.

The final paper run uses length-sorted fixed-count batches, global batch 8,
gradient accumulation 4, and 7,760 iterations x 15 epochs.  `run.sh` derives
the exact iteration count from the prepared inventory.

Save logits once:

```bash
python3 local/lid_softmax_logits.py \
  --lid_train_config exp/lid_<tag>/config.yaml \
  --lid_model_file exp/lid_<tag>/valid.accuracy.best.pth \
  --lang2utt dump/raw/train_lidseq/lang2utt \
  --wav_scp dump/raw/test_cs_all/wav.scp \
  --ref_utt2langs data/test_cs_all/utt2langs \
  --output exp/lid_<tag>/eval/test_cs_all.predictions.tsv \
  --results exp/lid_<tag>/eval/test_cs_all.json \
  --logits_output exp/lid_<tag>/eval/test_cs_all.logits.npz \
  --apply_loss_scale --ngpu 1
```

Recompute top-k and threshold sweeps without inference:

```bash
python3 local/score_lid2_logits_threshold_sweep.py \
  --logits_npz exp/lid_<tag>/eval/test_cs_all.logits.npz \
  --ref_utt2langs data/test_cs_all/utt2langs \
  --output_prefix exp/lid_<tag>/eval/test_cs_all
```

The saved AAM logits include scale 30 when `--apply_loss_scale` is supplied.
This does not affect top-k but is required to reproduce the reported softmax
threshold curves.
