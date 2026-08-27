# LID1 hard-target classifier

`run.sh` uses the standard ESPnet LID template.  FLEURS labels are ordinary
single classes.  In `train_lid_pair_cs*`, a two-language set is one atomic,
order-normalized class (`ara-eng`, never both `ara-eng` and `eng-ara`).

The final backend is MMS-1B with trainable upstream weights, learned S3PRL
multi-layer weighted sum, ECAPA-TDNN, channel-attentive statistics pooling,
RawNet3 192-dimensional projection, and AAMSoftmax with three sub-centers and
Inter-TopK negatives.

Top-k evaluation:

```bash
./local/score_lid_topk.sh \
  --ngpu 1 \
  --lid_exp exp/lid_<tag> \
  --train_set train_fleurs_lid \
  --test_sets "test_fleurs_lid test_cs_read_test test_cs_xtts_test1 test_cs_xtts_test2 test_cs_mms_test test_cs_all" \
  --save_full_output true
```

For an atomic-pair model set `--train_set train_lid_pair_cs` and
`--prediction_mode atomic_top1`.  The script writes per-utterance exact
decisions for significance tests.
