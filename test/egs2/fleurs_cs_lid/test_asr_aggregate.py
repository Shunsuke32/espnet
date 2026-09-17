import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


PATH = (
    Path(__file__).resolve().parents[3]
    / "egs2/fleurs_cs_lid/asr1/local/score_lidseq.py"
)
spec = importlib.util.spec_from_file_location("asr_aggregate_scorer", PATH)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def test_aggregate_retains_order_and_duplicate_predictions(tmp_path):
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    header = "utt\tref\thyp\tseq_exact\tset_exact\tlen_exact\n"
    first.write_text(header + "a\tara eng\teng ara\t0\t1\t1\n")
    second.write_text(header + "b\tara eng\teng eng\t0\t0\t1\n")
    refs = {"a": ("ara", "eng"), "b": ("ara", "eng")}
    hyp = scorer.read_detail_hypotheses([first, second], refs, {})
    assert hyp["b"] == ("eng", "eng")
    result = scorer.score(refs, hyp)
    assert result["sequence_exact_accuracy"] == 0
    assert result["unordered_set_exact_accuracy"] == 0.5
    with pytest.raises(ValueError, match="duplicate utterance"):
        scorer.read_detail_hypotheses([first, first], refs, {})
    with pytest.raises(ValueError, match="reference mismatch"):
        scorer.read_detail_hypotheses([first], {"a": ("eng", "ara")}, {})


def test_missing_hypotheses_do_not_publish_scores(tmp_path):
    ref, hyp = tmp_path / "ref", tmp_path / "hyp"
    ref.write_text("a ara eng\nb ara eng\n")
    hyp.write_text("a <ara> <eng>\n")
    out, details = tmp_path / "metrics.json", tmp_path / "details.tsv"
    result = subprocess.run(
        [
            sys.executable,
            str(PATH),
            "--ref",
            str(ref),
            "--hyp",
            str(hyp),
            "--out",
            str(out),
            "--details_out",
            str(details),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing hypotheses" in result.stderr
    assert not out.exists() and not details.exists()
