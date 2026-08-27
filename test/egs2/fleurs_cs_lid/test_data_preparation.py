import importlib.util
import json
import sys
from pathlib import Path

import pytest


RECIPE = Path(__file__).resolve().parents[3] / "egs2" / "fleurs_cs_lid"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fleurs_prep():
    return load_module(
        "fleurs_cs_lid_data_prep",
        RECIPE / "lid1" / "local" / "prepare_fleurs_cs_lid_data.py",
    )


@pytest.fixture(scope="module")
def yodas_prep():
    return load_module(
        "fleurs_cs_yodas_data_prep",
        RECIPE / "local" / "prepare_cs_yodas_views.py",
    )


def test_canonical_labels_and_nlsyms_guard(fleurs_prep):
    mapping = fleurs_prep.read_label_map(None)
    assert fleurs_prep.canonicalize_label("en_us", mapping) == "eng"
    assert fleurs_prep.canonicalize_label("cmn_hans_cn", mapping) == "cmn"
    fleurs_prep.validate_nlsyms(["<ara>", "<eng>"])
    with pytest.raises(ValueError, match="raw FLEURS"):
        fleurs_prep.validate_nlsyms(["<en_us>"])


def test_paper_cs_split_is_deterministic_and_disjoint(fleurs_prep):
    examples = [
        fleurs_prep.Example(
            uttid=f"cs_{i:04d}",
            wav=f"{i}.wav",
            labels=("ara", "eng"),
            speaker=f"spk{i}",
            source="cs_fleurs",
            subset="xtts/train",
        )
        for i in range(500)
    ]
    train_a, valid_a, _ = fleurs_prep.split_cs_train_valid_global_hash(
        {"xtts/train": examples}, 0.02
    )
    train_b, valid_b, _ = fleurs_prep.split_cs_train_valid_global_hash(
        {"xtts/train": list(reversed(examples))}, 0.02
    )
    assert {x.uttid for x in train_a} == {x.uttid for x in train_b}
    assert {x.uttid for x in valid_a} == {x.uttid for x in valid_b}
    assert {x.uttid for x in train_a}.isdisjoint(x.uttid for x in valid_a)


def test_source_specific_duration_policy(fleurs_prep):
    def example(uttid, duration):
        return fleurs_prep.Example(
            uttid=uttid,
            wav=f"{uttid}.wav",
            labels=("eng",),
            speaker=uttid,
            source="fleurs",
            subset="test",
            duration_sec=duration,
        )

    raw = {
        "train_lidseq": [
            example("short", 0.9),
            example("keep", 29.9),
            example("long", 30.0),
        ],
        "test_fleurs_lid": [example("test-long", 300.0)],
    }
    kept, removed = fleurs_prep.apply_duration_filter(
        raw,
        min_train_sec=1.0,
        max_train_sec=30.0,
        min_eval_sec=-1.0,
        max_eval_sec=0.0,
        missing_policy="error",
    )
    assert [x.uttid for x in kept["train_lidseq"]] == ["keep"]
    assert [x.uttid for x in kept["test_fleurs_lid"]] == ["test-long"]
    assert {x[1].uttid for x in removed} == {"short", "long"}


def test_yodas_video_split_pair_order_and_train_cap(yodas_prep):
    assert yodas_prep.is_exact_base_english(
        "ara", {"languages": ["Arabic", "English"]}
    )
    assert yodas_prep.is_exact_base_english(
        "ara", {"languages": ["English", "Arabic"]}
    )
    examples = []
    for lang in ("ara", "cmn"):
        for video in range(20):
            for segment in range(2):
                duration = 75.0 if segment == 0 else 10.0
                examples.append(
                    yodas_prep.Example(
                        uttid=f"{lang}-{video}-{segment}",
                        wav="x.wav",
                        labels=(lang, "eng"),
                        speaker="spk",
                        source="cs_yodas",
                        subset=lang,
                        raw_label_value="",
                        raw_label_parts=(lang, "eng"),
                        duration_sec=duration,
                        num_samples=None,
                        metadata={"video_id": f"{lang}-video-{video}"},
                    )
                )
    split = yodas_prep.split_yodas(examples, 0.8, 0.1, "cs-yodas-v1")
    yodas_prep.ensure_no_split_leak(split)
    train, dropped = yodas_prep.duration_cap(split["train"], 70.0)
    assert all(x.duration_sec < 70.0 for x in train)
    assert all(x.duration_sec >= 70.0 for x in dropped)
    assert any(x.duration_sec >= 70.0 for x in split["test"])
    assert yodas_prep.canonical_pair(("eng", "ara")) == "ara-eng"
    assert yodas_prep.canonical_pair(("ara", "eng")) == "ara-eng"


def test_end_to_end_fleurs_cs_fleurs_preparation(fleurs_prep, tmp_path):
    tsv_root = tmp_path / "fleurs"
    tsv_root.mkdir()
    header = "id\tpath\tlang_id_name\tspeaker\tduration\n"
    rows = {
        "train": [
            "1\t/audio/en-train.wav\ten_us\tspk1\t2.0\n",
            "2\t/audio/ar-train.wav\tar_eg\tspk2\t3.0\n",
            "3\t/audio/too-long.wav\ten_us\tspk3\t30.0\n",
        ],
        "dev": [
            "4\t/audio/en-dev.wav\ten_us\tspk4\t2.0\n",
            "5\t/audio/ar-dev.wav\tar_eg\tspk5\t2.0\n",
        ],
        "test": [
            "6\t/audio/en-test.wav\ten_us\tspk6\t120.0\n",
            "7\t/audio/ar-test.wav\tar_eg\tspk7\t2.0\n",
        ],
    }
    for split, split_rows in rows.items():
        (tsv_root / f"{split}.tsv").write_text(
            header + "".join(split_rows), encoding="utf-8"
        )

    cs_root = tmp_path / "cs-fleurs"
    for subset in ("xtts/train", "read/test"):
        subset_dir = cs_root / subset
        subset_dir.mkdir(parents=True)
        records = []
        count = 20 if subset == "xtts/train" else 2
        for index in range(count):
            records.append(
                {
                    "id": f"{subset.replace('/', '-')}-{index}",
                    "file_name": f"audio/{index}.wav",
                    "language": "Arabic + English",
                    "duration": 4.0 if index else 31.0,
                }
            )
        (subset_dir / "metadata.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    outdir = tmp_path / "data"
    fleurs_prep.main(
        [
            "--outdir",
            str(outdir),
            "--fleurs_tsv_root",
            str(tsv_root),
            "--fleurs_config",
            "all",
            "--cs_root",
            str(cs_root),
            "--cs_train_subsets",
            "xtts/train",
            "--cs_eval_subsets",
            "read/test",
            "--cs_dev_ratio",
            "0.1",
            "--cs_split_mode",
            "classwise_hash",
            "--token_format",
            "angle",
            "--duration_missing_policy",
            "error",
        ]
    )

    nlsyms = (outdir / "nlsyms.txt").read_text(encoding="utf-8").splitlines()
    assert "<eng>" in nlsyms
    assert "<ara>" in nlsyms
    assert "<ara-eng>" not in nlsyms
    assert "<en_us>" not in nlsyms

    pair_targets = (outdir / "train_lid_pair_cs" / "utt2lang").read_text(
        encoding="utf-8"
    )
    assert " ara-eng\n" in pair_targets
    assert " eng-ara\n" not in pair_targets
    assert "too-long" not in (outdir / "train_lidseq" / "wav.scp").read_text(
        encoding="utf-8"
    )
    assert "en-test" in (outdir / "test_fleurs_lid" / "wav.scp").read_text(
        encoding="utf-8"
    )


def test_fixed_batch_budget_keeps_updates_when_halving_epochs():
    budget_module = load_module(
        "fleurs_cs_lid_fixed_batch_budget",
        RECIPE / "lid1" / "local" / "make_fixed_batch_config.py",
    )
    common = {
        "num_utts": 310_000,
        "batch_size": 8,
        "accum_grad": 4,
        "effective_batch_size": 32,
        "ngpu": 2,
        "target_passes": 3.0,
        "warmup_ratio": 0.1,
        "round_updates_per_epoch_to": 10,
    }
    budget_30 = budget_module.compute_budget(max_epoch=30, **common)
    budget_15 = budget_module.compute_budget(max_epoch=15, **common)

    assert budget_30["num_iters_per_epoch"] == 3_880
    assert budget_15["num_iters_per_epoch"] == 7_760
    assert budget_30["total_optimizer_updates"] == 29_100
    assert budget_15["total_optimizer_updates"] == 29_100
    assert budget_30["actual_passes"] == budget_15["actual_passes"]


def test_small_asr_frozen_prefix_reproduces_historical_budget():
    budget_module = load_module(
        "fleurs_cs_lid_frozen_prefix_budget",
        RECIPE / "asr1" / "local" / "make_fixed_batch_config.py",
    )
    budget = budget_module.compute_budget(
        num_utts=310_000,
        batch_size=8,
        accum_grad=4,
        effective_batch_size=32,
        ngpu=4,
        max_epoch=10,
        target_passes=1.0,
        warmup_ratio=0.3,
        round_updates_per_epoch_to=10,
    )

    assert budget["num_iters_per_epoch"] == 3_880
    assert budget["total_optimizer_updates"] == 9_700
    assert budget["warmup_steps"] == 2_910
    assert budget["actual_passes"] == pytest.approx(1.0012903225806453)


def test_fixed_batch_config_preserves_model_specific_patience():
    budget_module = load_module(
        "fleurs_cs_lid_patience_budget",
        RECIPE / "asr1" / "local" / "make_fixed_batch_config.py",
    )
    budget = budget_module.compute_budget(
        num_utts=1_000,
        batch_size=8,
        accum_grad=4,
        effective_batch_size=32,
        ngpu=2,
        max_epoch=30,
        target_passes=3.0,
        warmup_ratio=0.1,
        round_updates_per_epoch_to=10,
    )
    config = {
        "patience": 10,
        "scheduler": "warmuplr",
        "scheduler_conf": {"warmup_steps": 1},
    }
    generated = budget_module.patch_config(
        config,
        budget,
        task="asr",
        batch_type="sorted",
        drop_last_iter=True,
    )
    assert generated["patience"] == 10
