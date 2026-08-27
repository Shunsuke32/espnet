import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


RECIPE = Path(__file__).resolve().parents[3] / "egs2" / "fleurs_cs_lid"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_lid2_saved_scores_are_pickle_free_and_order_insensitive(tmp_path):
    scorer = load_module(
        "lid2_saved_scorer",
        RECIPE / "lid2" / "local" / "score_lid2_logits_threshold_sweep.py",
    )
    labels = ["ara", "eng", "jpn"]
    utt_ids = ["single", "pair"]
    logits = np.asarray([[0.0, 4.0, -2.0], [3.0, 2.0, -4.0]], dtype=np.float32)
    archive = tmp_path / "logits.npz"
    np.savez_compressed(
        archive,
        utt_ids=np.asarray(utt_ids),
        labels=np.asarray(labels),
        logits=logits,
    )
    refs = tmp_path / "utt2langs"
    refs.write_text("single eng\npair eng ara\n", encoding="utf-8")
    prefix = tmp_path / "scores"
    old_argv = sys.argv
    try:
        sys.argv = [
            "score_lid2",
            "--logits_npz",
            str(archive),
            "--ref_utt2langs",
            str(refs),
            "--output_prefix",
            str(prefix),
        ]
        scorer.main()
    finally:
        sys.argv = old_argv

    derived = np.load(
        prefix.with_suffix(".logits_and_softmax_probs.npz"), allow_pickle=False
    )
    assert derived["utt_ids"].tolist() == utt_ids
    with prefix.with_suffix(".cardinality_matched_topk.details.tsv").open(
        encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert [row["exact"] for row in rows] == ["1", "1"]


def test_paired_significance_and_seen_summary(tmp_path):
    details_a = tmp_path / "a.tsv"
    details_b = tmp_path / "b.tsv"
    details_a.write_text(
        "utt\tref\tpred\texact\n"
        "u1\teng\teng\t1\n"
        "u2\tara eng\tara\t0\n"
        "u3\tjpn eng\tjpn\t0\n",
        encoding="utf-8",
    )
    details_b.write_text(
        "utt\tref\tpred\texact\n"
        "u1\teng\teng\t1\n"
        "u2\tara eng\teng ara\t1\n"
        "u3\tjpn eng\tjpn\t0\n",
        encoding="utf-8",
    )
    significance = tmp_path / "significance.json"
    subprocess.run(
        [
            sys.executable,
            str(RECIPE / "evaluation" / "paired_significance.py"),
            "--system_a",
            str(details_a),
            "--system_b",
            str(details_b),
            "--bootstrap_samples",
            "100",
            "--output",
            str(significance),
        ],
        check=True,
    )
    result = json.loads(significance.read_text(encoding="utf-8"))
    assert result["num_paired_utterances"] == 3
    assert result["accuracy_difference_b_minus_a"] == 1 / 3

    train = tmp_path / "train.utt2langs"
    refs = tmp_path / "refs.utt2langs"
    train.write_text("t1 eng\nt2 ara eng\n", encoding="utf-8")
    refs.write_text("u1 eng\nu2 eng ara\nu3 jpn eng\n", encoding="utf-8")
    prefix = tmp_path / "summary"
    subprocess.run(
        [
            sys.executable,
            str(RECIPE / "evaluation" / "summarize_details.py"),
            "--details",
            str(details_b),
            "--ref_utt2langs",
            str(refs),
            "--train_utt2langs",
            str(train),
            "--output_prefix",
            str(prefix),
        ],
        check=True,
    )
    rows = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))["rows"]
    seen = next(row for row in rows if row["group"] == "seen")
    unseen = next(row for row in rows if row["group"] == "unseen")
    assert seen["n"] == 2 and seen["correct"] == 2
    assert unseen["n"] == 1 and unseen["correct"] == 0
