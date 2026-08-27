# LID3 sigmoid multi-label classifier

LID3 uses multi-hot targets and sigmoid probabilities.  Two heads are retained
because both appear in paper comparisons:

- `train_mms_ecapa_multilabel_bce.yaml`: Linear + BCE baseline.
- `train_mms_ecapa_multilabel_aam_bce_posw50.yaml`: cosine AAM, three
  sub-centers, Inter-TopK, BCE, and `pos_weight=50` final model.

Save raw logits and predictions:

```bash
python3 local/lid_sigmoid_topk.py \
  --lid_train_config exp/mixed/lid_<tag>/config.yaml \
  --lid_model_file exp/mixed/lid_<tag>/valid.accuracy.best.pth \
  --lang2utt dump/mixed/raw/train_lidseq/lang2utt \
  --wav_scp dump/mixed/raw/test_cs_all/wav.scp \
  --ref_utt2langs data/test_cs_all/utt2langs \
  --output exp/mixed/lid_<tag>/eval/test_cs_all.predictions.tsv \
  --results exp/mixed/lid_<tag>/eval/test_cs_all.json \
  --logits_output exp/mixed/lid_<tag>/eval/test_cs_all.logits.npz \
  --topk 2 --score_k 2 --ngpu 1
```

Score all thresholds and create the paper plot:

```bash
python3 local/score_lid3_sigmoid_logits_threshold_sweep.py \
  --logits_npz exp/mixed/lid_<tag>/eval/test_cs_all.logits.npz \
  --ref_utt2langs data/test_cs_all/utt2langs \
  --output_prefix exp/mixed/lid_<tag>/eval/test_cs_all \
  --save_sigmoid_npz

python3 local/plot_lid3_sigmoid_threshold_sweep.py \
  --score_dir exp/mixed/lid_<tag>/eval \
  --output_prefix exp/mixed/lid_<tag>/eval/threshold_sweep
```

Run the scorer once per FLEURS/CS-FLEURS/CS-YODAS data directory; do not
concatenate independently repeated inference outputs.
