import importlib.util
import csv
import hashlib
import json
import os
import subprocess
import types
import sys
from dataclasses import replace
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


@pytest.fixture(scope="module")
def fleurs_download():
    return load_module(
        "fleurs_cs_download",
        RECIPE / "lid1" / "local" / "create_fleurs_lid_dataset.py",
    )


@pytest.fixture(scope="module")
def duration_view():
    return load_module(
        "fleurs_duration_view", RECIPE / "local/finalize_duration_view.py"
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
    assert yodas_prep.is_exact_base_english("ara", {"languages": ["Arabic", "English"]})
    assert yodas_prep.is_exact_base_english("ara", {"languages": ["English", "Arabic"]})
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
            "0.02",
            "--cs_split_mode",
            "global_hash",
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


def test_only_old_split_is_available(fleurs_prep):
    with pytest.raises(SystemExit):
        fleurs_prep.get_parser().parse_args(["--cs_split_mode", "classwise_hash"])
    with pytest.raises(ValueError, match="OLD"):
        fleurs_prep.split_cs_train_valid_global_hash({}, 0.1)
    with pytest.raises(ValueError, match="OLD"):
        fleurs_prep.main(["--cs_dev_ratio", "0.1"])


def test_materialized_ids_relocate_without_changing_old_values(fleurs_prep):
    name = "31_13412724805436051564_4df8c99c467bba1a.wav"
    suffix = f"fa_ir/train/{name}"
    old = f"{fleurs_prep.OLD_FLEURS_AUDIO_NAMESPACE}/{suffix}"
    expected = "fleurs_fa_ir_31_31_13412724805436051564_4df8c99c467bba1a_b425e27140d9"
    assert fleurs_prep.fleurs_uttid("fa_ir", 31, old) == expected
    for root in ("/new-server/audio", "/different/custom-audio-root", "relative/audio"):
        assert fleurs_prep.fleurs_uttid("fa_ir", 31, f"{root}/{suffix}") == expected
    # Arbitrary nonmaterialized TSV paths retain the legacy hashing contract.
    path = "/audio/recording.wav"
    digest = hashlib.sha1(path.encode()).hexdigest()[:12]
    assert (
        fleurs_prep.fleurs_uttid("en_us", 1, path)
        == f"fleurs_en_us_1_recording_{digest}"
    )


def write_fleurs_path_tsv(path, wav, **fields):
    row = {
        "id": "31",
        "path": str(wav),
        "lang_id_name": "fa_ir",
        "speaker": "original speaker",
        "raw_transcription": "original text",
        "duration": "1.0230625",
        "num_samples": "16369",
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    return row


@pytest.mark.parametrize(
    "relative",
    [
        "audio/fa_ir/train/31_13412724805436051564_4df8c99c467bba1a.wav",
        "audio/recording with spaces.wav",
    ],
)
def test_relative_fleurs_paths_are_tsv_root_relative(fleurs_prep, tmp_path, relative):
    import wave

    root = tmp_path / "TSV root with spaces"
    audio = root / relative
    audio.parent.mkdir(parents=True)
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 32000)
    row = write_fleurs_path_tsv(
        root / "train.tsv", relative, duration="", num_samples=""
    )
    before = {path: path.read_bytes() for path in (audio, root / "train.tsv")}
    (ex,) = fleurs_prep.load_fleurs_tsv_split(
        root, "train", fleurs_prep.read_label_map(None), 0, True
    )
    assert ex.wav == str(audio.resolve())
    assert ex.uttid == fleurs_prep.fleurs_uttid("fa_ir", "31", relative)
    assert (ex.duration_sec, ex.num_samples) == (2.0, 32000)
    assert ex.metadata["source_metadata"] == row
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("old_exists", [False, True])
def test_explicit_fleurs_relocation_preserves_source_and_old_ids(
    fleurs_prep, tmp_path, old_exists
):
    root, audio_root = tmp_path / "original TSVs", tmp_path / "new audio root"
    old_root = tmp_path / "old inaccessible server"
    name = "31_13412724805436051564_4df8c99c467bba1a.wav"
    old_rows = {}
    for hf_split, tsv_split in (
        ("train", "train"),
        ("validation", "dev"),
        ("test", "test"),
    ):
        suffix = Path("fa_ir") / hf_split / name
        wav = audio_root / suffix
        wav.parent.mkdir(parents=True)
        wav.write_bytes(b"existing new-server audio")
        if old_exists:
            old = old_root / suffix
            old.parent.mkdir(parents=True)
            old.write_bytes(b"different old-server file; must never be selected")
        old_rows[hf_split] = write_fleurs_path_tsv(
            root / f"{tsv_split}.tsv", old_root / suffix
        )
    provenance = {
        "revision": fleurs_prep.PAPER_FLEURS_REVISION,
        "tsv_sha256": {p.name: fleurs_prep.sha256_file(p) for p in root.glob("*.tsv")},
    }
    (root / "source.json").write_text(json.dumps(provenance))
    files = [
        p
        for directory in (root, audio_root, old_root)
        for p in directory.rglob("*")
        if p.is_file()
    ]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
    out = tmp_path / "outputs"
    fleurs_prep.main(
        [
            "--fleurs_tsv_root",
            str(root),
            "--fleurs_audio_root",
            str(audio_root),
            "--outdir",
            str(out),
            "--duration_missing_policy",
            "error",
        ]
    )
    expected_train = (
        "fleurs_fa_ir_31_31_13412724805436051564_4df8c99c467bba1a_b425e27140d9"
    )
    for hf_split, out_split in (
        ("train", "train"),
        ("validation", "valid"),
        ("test", "test"),
    ):
        row = old_rows[hf_split]
        expected = fleurs_prep.fleurs_uttid("fa_ir", "31", row["path"])
        if hf_split == "train":
            assert expected == expected_train
        folder = out / f"{out_split}_fleurs_lid"
        assert (
            folder / "wav.scp"
        ).read_text().strip() == f"{expected} {audio_root / 'fa_ir' / hf_split / name}"
        record = json.loads((folder / "labels.jsonl").read_text())
        assert record["source_metadata"] == row
        assert record["speaker"] == "fleurs_original speaker"
        assert record["duration_sec"] == 1.0230625
        assert (folder / "utt2num_samples").read_text().strip() == f"{expected} 16368"
    assert all(
        (p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items()
    )
    args = fleurs_prep.get_parser().parse_args(
        ["--fleurs_tsv_root", str(root), "--fleurs_audio_root", str(audio_root)]
    )
    audit = fleurs_prep.source_provenance(args)
    assert audit["fleurs_source"] == provenance
    assert audit["fleurs_audio_root"] == str(audio_root.resolve())


def test_fleurs_relocation_never_falls_back_to_old_audio(fleurs_prep, tmp_path):
    root = tmp_path / "TSVs"
    suffix = Path("fa_ir/train/31_13412724805436051564_4df8c99c467bba1a.wav")
    old = tmp_path / "old" / suffix
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old audio")
    new_root = root / "audio"
    new = new_root / suffix
    new.parent.mkdir(parents=True)
    write_fleurs_path_tsv(root / "train.tsv", old)
    args = (root, "train", fleurs_prep.read_label_map(None), 0, True)
    # Even a partially copied local tree must not silently fall back to old audio.
    with pytest.raises(ValueError, match="ambiguous.*--fleurs_audio_root"):
        fleurs_prep.load_fleurs_tsv_split(*args)
    with pytest.raises(FileNotFoundError, match="relocated FLEURS audio"):
        fleurs_prep.load_fleurs_tsv_split(*args, audio_root=new_root)
    new.write_bytes(b"new audio")
    with pytest.raises(ValueError, match="ambiguous"):
        fleurs_prep.load_fleurs_tsv_split(*args)
    (ex,) = fleurs_prep.load_fleurs_tsv_split(*args, audio_root=new_root)
    assert ex.wav == str(new)
    # Original materialized TSV/audio in a single root still needs no remapping.
    write_fleurs_path_tsv(root / "train.tsv", new)
    (same,) = fleurs_prep.load_fleurs_tsv_split(*args)
    assert same.wav == str(new) and same.uttid == ex.uttid


@pytest.mark.parametrize(
    "wav",
    [
        "../outside.wav",
        "audio/../../outside.wav",
        "/old/../escape/file.wav",
        "audio/x\nwrong.wav",
        "audio/x\twrong.wav",
        "audio/x\x00.wav",
        "command |",
        "https://example.invalid/x.wav",
        r"audio\..\outside.wav",
    ],
)
def test_fleurs_audio_relocation_rejects_unsafe_paths(fleurs_prep, tmp_path, wav):
    with pytest.raises(ValueError, match="unsafe"):
        fleurs_prep.resolve_fleurs_tsv_audio(wav, tmp_path, "fa_ir", "train")


@pytest.mark.parametrize(
    "suffix",
    [
        "en_us/train/31_13412724805436051564_4df8c99c467bba1a.wav",
        "fa_ir/test/31_13412724805436051564_4df8c99c467bba1a.wav",
        "fa_ir/train/arbitrary.wav",
    ],
)
def test_fleurs_explicit_root_requires_materialized_identity(
    fleurs_prep, tmp_path, suffix
):
    with pytest.raises(ValueError, match="matching materialized"):
        fleurs_prep.resolve_fleurs_tsv_audio(
            "/old/" + suffix, tmp_path, "fa_ir", "train", tmp_path
        )


@pytest.mark.parametrize("explicit", [False, True])
def test_fleurs_relocation_rejects_symlink_escape(fleurs_prep, tmp_path, explicit):
    root = tmp_path / "root"
    suffix = Path("fa_ir/train/31_13412724805436051564_4df8c99c467bba1a.wav")
    dest = root / suffix
    dest.parent.mkdir(parents=True)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    dest.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes root"):
        fleurs_prep.resolve_fleurs_tsv_audio(
            str(Path("/old") / suffix) if explicit else str(suffix),
            root,
            "fa_ir",
            "train",
            root if explicit else None,
        )


def test_fleurs_audio_root_cannot_be_ignored(fleurs_prep, tmp_path):
    with pytest.raises(ValueError, match="active --fleurs_tsv_root"):
        fleurs_prep.main(
            ["--fleurs_audio_root", str(tmp_path), "--outdir", str(tmp_path / "out")]
        )
    assert not (tmp_path / "out").exists()


def test_audio_reuse_checks_content(fleurs_download, tmp_path):
    row = {"id": 7, "audio": {"path": "recording.wav", "bytes": b"original audio"}}
    kwargs = dict(raw_lang="en_us", hf_split="train", audio_root=tmp_path)
    path = Path(fleurs_download.materialize_audio(row, **kwargs))
    assert fleurs_download.materialize_audio(row, **kwargs) == str(path)
    path.write_bytes(b"corrupt audio")
    with pytest.raises(ValueError, match="content mismatch"):
        fleurs_download.materialize_audio(row, **kwargs)
    assert path.read_bytes() == b"corrupt audio"


@pytest.mark.parametrize("splits", [("train",), ("train", "dev", "test")])
def test_partial_tsv_presence_never_implies_complete_language(
    fleurs_download, tmp_path, splits
):
    for split in splits:
        (tmp_path / f".{split}.tsv.tmp.123").write_text(
            "lang_id_name\nen_us\n", encoding="utf-8"
        )
    with pytest.raises(RuntimeError, match="cannot resume safely"):
        fleurs_download.check_partial_tsvs(tmp_path)


def test_source_text_and_speaker_survive_pair_and_yodas_views(
    fleurs_download, fleurs_prep, yodas_prep, tmp_path
):
    source_row = {
        "id": 7,
        "path": "/audio/original.wav",
        "raw_transcription": "original\u2009words",
        "transcription": "original words",
        "num_samples": 32000,
    }
    row = fleurs_download.google_fleurs_row(source_row, raw_lang="en_us", lang_id=0)
    with (tmp_path / "train.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fleurs_download.FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    examples = fleurs_prep.load_fleurs_tsv_split(
        tmp_path, "train", fleurs_prep.read_label_map(None), 0, True
    )
    fleurs_prep.write_data_dir("source", examples, tmp_path, "angle")
    loaded = yodas_prep.read_existing_data_dir(tmp_path / "source")
    assert loaded[0].speaker == examples[0].speaker == "fleurs_7"
    assert (
        loaded[0].metadata["source_metadata"]["raw_transcription"]
        == "original\u2009words"
    )
    assert loaded[0].metadata["source_metadata"]["source_path"] == "/audio/original.wav"
    pair = yodas_prep.to_pair_examples(loaded)
    yodas_prep.write_data_dir("pair", pair, tmp_path)
    reread = yodas_prep.read_existing_data_dir(tmp_path / "pair")
    assert reread[0].speaker == "fleurs_7"
    assert (
        reread[0].metadata["source_metadata"] == loaded[0].metadata["source_metadata"]
    )
    assert (tmp_path / "pair" / "text").read_text().endswith(" <eng>\n")


def test_symbols_and_atomic_training_inventory_are_train_only(fleurs_prep, tmp_path):
    ex = fleurs_prep.Example(
        "train", "x.wav", ("eng", "ara"), "speaker", "cs_fleurs", "xtts/train"
    )
    pair = fleurs_prep.to_pair_class_examples([ex])[0]
    assert pair.labels == ("ara-eng",)
    dirs = {
        "train_lidseq": [ex],
        "test_lidseq": [replace(ex, uttid="test", labels=("rus", "jpn"))],
    }
    fleurs_prep.write_nlsyms(dirs, "angle", tmp_path / "nlsyms.txt")
    assert (tmp_path / "nlsyms.txt").read_text().splitlines() == ["<ara>", "<eng>"]
    dirs["train_lid_pair_cs"] = [pair]
    fleurs_prep.write_inventory(tmp_path, dirs, fleurs_prep.read_label_map(None))
    classes = json.loads((tmp_path / "local" / "training_classes.json").read_text())
    assert classes == {"train_lidseq": ["ara", "eng"], "train_lid_pair_cs": ["ara-eng"]}


@pytest.mark.parametrize(
    "raw,expected",
    [("fil", "tgl"), ("msa", "zlm"), ("ori", "ory"), ("nor", "nob"), ("zh", "cmn")],
)
def test_legacy_aliases(fleurs_prep, raw, expected):
    assert (
        fleurs_prep.canonicalize_label(raw, fleurs_prep.read_label_map(None))
        == expected
    )


def test_legacy_source_boundaries_and_metadata_precedence(fleurs_prep):
    ex = fleurs_prep.Example("u", "unused.wav", ("eng",), "s", "fleurs", "train")
    durations = [0.999999, 1.0, 1.000001, 29.999999, 30.0]
    rows = [replace(ex, uttid=str(d), duration_sec=d) for d in durations]
    kept, _ = fleurs_prep.duration_filter_for_set(
        "train_lidseq",
        rows,
        min_train_sec=1,
        max_train_sec=30,
        min_eval_sec=-1,
        max_eval_sec=0,
        missing_policy="error",
    )
    assert [e.duration_sec for e in kept] == [1.0, 1.000001, 29.999999]
    # OLD source metadata and post-format sample filtering are distinct contracts.
    duration, samples = fleurs_prep.infer_duration_from_row(
        {"duration": 1.0230625, "num_samples": 16369}, "unused.wav", 16000
    )
    assert (duration, samples) == (1.0230625, 16369)


def test_yodas_exact_70_boundary_and_serialized_rounding(yodas_prep, tmp_path):
    base = yodas_prep.Example(
        "u", "x.wav", ("ara", "eng"), "s", "cs_yodas", "ara", "", (), 70.0, None
    )
    examples = [
        replace(base, uttid=str(d), duration_sec=d) for d in (69.999, 70.0, 70.001)
    ]
    kept, dropped = yodas_prep.duration_cap(examples, 70)
    assert [e.duration_sec for e in kept] == [69.999]
    assert [e.duration_sec for e in dropped] == [70.0, 70.001]
    yodas_prep.write_data_dir("train", kept, tmp_path)
    assert (tmp_path / "train" / "utt2dur").read_text() == "69.999 69.999000\n"
    assert yodas_prep.parse_context_duration("video_ara_000001000_000071000")[1] == 70.0


def test_metadata_checksums_are_enforced(yodas_prep, tmp_path):
    (tmp_path / "ara.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        yodas_prep.verify_metadata(tmp_path)


def test_tsv_revision_and_content_checks(fleurs_prep, tmp_path):
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}.tsv").write_text("id\tpath\n")
    args = fleurs_prep.get_parser().parse_args(["--fleurs_tsv_root", str(tmp_path)])
    source = {
        "revision": fleurs_prep.PAPER_FLEURS_REVISION,
        "tsv_sha256": {
            p.name: fleurs_prep.sha256_file(p) for p in tmp_path.glob("*.tsv")
        },
    }
    (tmp_path / "source.json").write_text(json.dumps(source))
    assert fleurs_prep.source_provenance(args)["fleurs_source"] == source
    (tmp_path / "train.tsv").write_text("changed\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        fleurs_prep.source_provenance(args)
    source.pop("tsv_sha256")
    (tmp_path / "source.json").write_text(json.dumps(source))
    assert fleurs_prep.source_provenance(args)["fleurs_source"] == source
    source["revision"] = "wrong-revision"
    (tmp_path / "source.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="revision mismatch"):
        fleurs_prep.source_provenance(args)


def test_fresh_pinned_acquisition_with_in_memory_source(
    fleurs_download, fleurs_prep, monkeypatch, tmp_path
):
    calls = []

    class Split(list):
        def cast_column(self, name, feature):
            assert name == "audio"
            assert feature.decode is False
            return self

    def load_dataset(dataset, config, **kwargs):
        assert dataset == "google/fleurs"
        assert kwargs["revision"] == fleurs_prep.PAPER_FLEURS_REVISION
        calls.append(kwargs["split"])
        return Split(
            [
                {
                    "id": 7,
                    "audio": {"path": "recording.wav", "bytes": b"audio bytes"},
                    "num_samples": 32000,
                    "raw_transcription": "Original sentence",
                }
            ]
        )

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            Audio=lambda **kwargs: types.SimpleNamespace(**kwargs),
            load_dataset=load_dataset,
        ),
    )
    fleurs_download.write_google_fleurs(
        configs=["en_us"],
        out_root=tmp_path,
        cache_dir=str(tmp_path / "cache"),
        subsample_per_lang=0,
        resume_partial=True,
        materialize_audio_files=True,
        audio_root=None,
        revision=fleurs_prep.PAPER_FLEURS_REVISION,
    )
    assert calls == ["train", "validation", "test"]
    args = fleurs_prep.get_parser().parse_args(["--fleurs_tsv_root", str(tmp_path)])
    provenance = fleurs_prep.source_provenance(args)
    assert set(provenance["fleurs_source"]["tsv_sha256"]) == {
        "train.tsv",
        "dev.tsv",
        "test.tsv",
    }
    assert not list(tmp_path.glob(".*.tsv.tmp.*"))


def test_asr_uses_canonical_data_code():
    for name in (
        "data.sh",
        "create_fleurs_lid_dataset.py",
        "prepare_fleurs_cs_lid_data.py",
    ):
        assert (RECIPE / "asr1/local" / name).resolve() == (
            RECIPE / "lid1/local" / name
        ).resolve()
    assert (RECIPE / "asr1/local/verify_lid_data.sh").resolve() == (
        RECIPE / "local/verify_lid_data.sh"
    ).resolve()


def test_downloader_validates_pinned_nested_layout_without_network(tmp_path):
    root = tmp_path / "cs-fleurs"
    (root / ".git").mkdir(parents=True)
    for subset in ("read/test", "xtts/train", "xtts/test1", "xtts/test2", "mms/test"):
        path = root / subset / "metadata.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")
    commands = tmp_path / "bin"
    commands.mkdir()
    git = commands / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        "  *'status --porcelain'*) exit 0 ;;\n"
        "  *'fetch origin '*|*'checkout --detach '*|*'lfs pull'*) exit 0 ;;\n"
        "  *'rev-parse HEAD'*) echo 0cdbf166c5517ae4b6eb1c54248522eedec53017 ;;\n"
        "  *) exit 99 ;;\n"
        "esac\n"
    )
    git.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = subprocess.run(
        ["bash", str(RECIPE / "local/download_cs_fleurs.sh"), "--root", str(root)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "0cdbf166c5517ae4b6eb1c54248522eedec53017" in result.stderr


def test_new_split_rejected_before_data_acquisition():
    result = subprocess.run(
        ["bash", "local/data.sh", "--cs_split_mode", "classwise_hash"],
        cwd=RECIPE / "lid1",
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "only the OLD" in result.stderr


def test_verifier_checks_recording_paths_containing_spaces(fleurs_prep, tmp_path):
    import wave

    audio = tmp_path / "audio files"
    audio.mkdir()
    for split in ("train", "dev", "test"):
        path = audio / f"{split}.wav"
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b"\0\0" * 32000)
        with (tmp_path / f"{split}.tsv").open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["id", "path", "lang_id_name", "duration"])
            writer.writerow([split, str(path), "en_us", "2.0"])
    data = tmp_path / "data"
    fleurs_prep.main(
        [
            "--fleurs_tsv_root",
            str(tmp_path),
            "--outdir",
            str(data),
            "--token_format",
            "angle",
        ]
    )
    command = [
        "bash",
        "local/verify_lid_data.sh",
        "--data_dir",
        str(data),
        "--require_cs",
        "false",
    ]
    result = subprocess.run(
        command, cwd=RECIPE / "lid1", capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (audio / "train.wav").unlink()
    result = subprocess.run(
        command, cwd=RECIPE / "lid1", capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "unreadable audio path" in result.stderr


def test_raw_copy_sidecars_keep_historical_persian(yodas_prep, duration_view, tmp_path):
    ex = yodas_prep.Example(
        "persian",
        "/readonly/original.wav",
        ("fas",),
        "speaker",
        "fleurs",
        "train",
        "fa_ir",
        ("fa_ir",),
        1.0230625,
        16369,
    )
    other = replace(ex, uttid="other", labels=("eng",), duration_sec=2.0)
    yodas_prep.write_data_dir("train_lidseq", [ex, other], tmp_path)
    source = tmp_path / "train_lidseq"
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    assert yodas_prep.read_kv(source / "utt2dur")["persian"] == "1.023062"
    assert yodas_prep.read_kv(source / "utt2num_samples")["persian"] == "16368"
    copied = tmp_path / "copy-view"
    report = duration_view.prepare_view(
        source, source, mode="raw_copy", output_dir=copied
    )
    assert report["num_kept"] == 2
    assert yodas_prep.read_kv(copied / "wav.scp") == yodas_prep.read_kv(
        source / "wav.scp"
    )
    assert not list(copied.glob("*.wav"))
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    with pytest.raises(FileExistsError):
        duration_view.prepare_view(source, source, mode="raw_copy", output_dir=copied)
    reused = duration_view.prepare_view(
        source, source, mode="raw_copy", output_dir=copied, reuse_existing=True
    )
    assert reused["num_kept"] == 2
    (copied / "utt2num_samples").write_text("other 32000\npersian 15345\n")
    with pytest.raises(ValueError, match="stale or modified"):
        duration_view.prepare_view(
            source, source, mode="raw_copy", output_dir=copied, reuse_existing=True
        )

    formatted = tmp_path / "formatted"
    formatted.mkdir()
    (formatted / "wav.scp").write_bytes((source / "wav.scp").read_bytes())
    (formatted / "utt2num_samples").write_text("other 32000\npersian 15345\n")
    actual = tmp_path / "raw-view"
    report = duration_view.prepare_view(
        formatted, source, mode="raw", output_dir=actual
    )
    assert report["num_kept"] == 1
    assert report["dropped"]["persian"]["num_samples"] == 15345
    for name in (
        "wav.scp",
        "text",
        "utt2spk",
        "utt2lang",
        "utt2langs",
        "utt2category",
        "utt2dur",
        "utt2num_samples",
    ):
        assert set(yodas_prep.read_kv(actual / name)) == {"other"}
    assert list(yodas_prep.read_labels_jsonl(actual / "labels.jsonl")) == ["other"]
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}


def test_duration_view_explicit_boundaries_and_no_fallback(
    yodas_prep, duration_view, tmp_path
):
    base = yodas_prep.Example(
        "x", "x.wav", ("eng",), "spk", "fleurs", "train", "en_us", (), 2.0, None
    )
    rows = [
        replace(base, uttid="at1", duration_sec=1.0),
        replace(base, uttid="at30", duration_sec=30.0),
        replace(base, uttid="below30", duration_sec=29.999),
        replace(
            base,
            uttid="at70",
            duration_sec=70.0,
            source="cs_yodas",
            labels=("ara", "eng"),
        ),
        replace(
            base,
            uttid="below70",
            duration_sec=69.999,
            source="cs_yodas",
            labels=("ara", "eng"),
        ),
    ]
    yodas_prep.write_data_dir("train_lidseq", rows, tmp_path)
    source = tmp_path / "train_lidseq"
    report = duration_view.prepare_view(source, source, mode="raw_copy")
    assert report["num_kept"] == 2
    assert set(report["dropped"]) == {"at1", "at30", "at70"}
    checked = subprocess.run(
        [
            sys.executable,
            str(RECIPE / "local/finalize_duration_view.py"),
            "--input-dir",
            str(source),
            "--mode",
            "raw_copy",
            "--audit-only",
            "--require-no-drops",
        ],
        text=True,
        capture_output=True,
    )
    assert checked.returncode != 0 and "duration exclusions" in checked.stderr
    (source / "utt2num_samples").write_text("at1 999\n")
    with pytest.raises(ValueError, match="disagrees"):
        duration_view.prepare_view(source, source, mode="raw_copy")
    (source / "utt2num_samples").unlink()
    with pytest.raises(FileNotFoundError):
        duration_view.prepare_view(source, source, mode="raw")
    with pytest.raises(ValueError, match="never test"):
        duration_view.prepare_view(source, tmp_path / "test_lidseq", mode="raw_copy")


def test_offline_data_entry_does_not_try_acquisition(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "local/data.sh",
            "--skip_fleurs_download",
            "true",
            "--fleurs_tsv_root",
            str(tmp_path / "missing"),
            "--outdir",
            str(tmp_path / "out"),
        ],
        cwd=RECIPE / "lid1",
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "skip FLEURS TSV creation" in result.stdout
    assert "missing or empty FLEURS TSV" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(
    os.environ.get("CS_LID_REAL_DATA_SMOKE") != "1",
    reason="existing local sources only",
)
def test_bounded_real_offline_sources(fleurs_prep, yodas_prep, duration_view, tmp_path):
    import soundfile as sf

    required = (
        "CS_LID_REFERENCE_ROOT",
        "FLEURS_TSV_ROOT",
        "CS_FLEURS_ROOT",
        "CS_YODAS_ROOT",
    )
    if any(not os.environ.get(key) for key in required):
        pytest.skip("real smoke needs " + ", ".join(required))
    live, tsv_root, cs_root, yodas_root = (Path(os.environ[key]) for key in required)
    tsv = tmp_path / "fleurs"
    tsv.mkdir()
    persian_name = "31_13412724805436051564_4df8c99c467bba1a.wav"
    for split in ("train", "dev", "test"):
        selected, languages = [], set()
        with (tsv_root / f"{split}.tsv").open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            fields = reader.fieldnames
            for row in reader:
                lang = row["lang_id_name"]
                if (
                    lang in {"en_us", "ar_eg", "fa_ir"} - languages
                    and 1 < float(row["duration"]) < 30
                ) or persian_name in row["path"]:
                    selected.append(row)
                    languages.add(lang)
        assert len(languages) == 3
        with (tsv / f"{split}.tsv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(selected)
    cs = tmp_path / "cs"
    for subset in ("xtts/train", "read/test", "xtts/test1", "xtts/test2", "mms/test"):
        dest = cs / subset
        dest.mkdir(parents=True)
        (dest / "audio").symlink_to(
            cs_root / subset / "audio", target_is_directory=True
        )
        rows, partitions = [], set()
        for row in fleurs_prep.iter_json_objects(cs_root / subset / "metadata.jsonl"):
            labels = fleurs_prep.parse_label_sequence(
                row["language"], fleurs_prep.read_label_map(None), strict=True
            )
            if set(labels) != {"ara", "eng"} or not 1 < float(row["duration"]) < 30:
                continue
            utt = fleurs_prep.sanitize_id(f"cs_{subset.replace('/', '_')}_{row['id']}")
            part = fleurs_prep.stable_fraction(utt) < 0.02
            if subset != "xtts/train" or part not in partitions:
                rows.append(row)
                partitions.add(part)
            if subset != "xtts/train" or len(partitions) == 2:
                break
        (dest / "metadata.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    out = tmp_path / "manifests"
    command = [
        "bash",
        "local/data.sh",
        "--skip_fleurs_download",
        "true",
        "--fleurs_tsv_root",
        str(tsv),
        "--cs_root",
        str(cs),
        "--outdir",
        str(out),
        "--nlsyms_txt",
        str(out / "nlsyms.txt"),
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    prepared = subprocess.run(
        command,
        cwd=RECIPE / "lid1",
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    yodas_prep.verify_metadata(yodas_root / "metadata")
    records = yodas_prep.load_yodas_records(
        yodas_root / "metadata", yodas_root / "audio"
    )
    splits = yodas_prep.split_yodas(records, 0.8, 0.1, "cs-yodas-v1")
    report = {
        "command": command,
        "audio_copies": 0,
        "splits": {},
        "yodas_pins": {
            "verified": True,
            "revision": yodas_prep.PAPER_REVISION,
            "metadata_sha256": yodas_prep.METADATA_SHA256,
        },
        "existing_source_roots": {key: os.environ[key] for key in required},
    }
    for split in ("train", "valid"):
        base = yodas_prep.read_existing_data_dir(out / f"{split}_lidseq")
        added, _ = yodas_prep.duration_cap(splits[split], 70.0)
        selected = base + added[:2]
        name = f"{split}_lidseq_csall_yodas"
        yodas_prep.write_data_dir(name, selected, out)
        source, final = out / name, tmp_path / "raw_copy" / name
        audit = duration_view.prepare_view(
            source, source, mode="raw_copy", output_dir=final
        )
        assert audit["num_dropped"] == 0
        assert duration_view.read_kv(source / "wav.scp") == duration_view.read_kv(
            final / "wav.scp"
        )
        wanted, matched = duration_view.read_kv(source / "utt2langs"), {}
        with (live / "lid1/data" / f"dur70_{name}" / "utt2langs").open() as f:
            for line in f:
                key, labels = line.rstrip().split(maxsplit=1)
                if key in wanted:
                    matched[key] = labels
        assert matched == wanted
        assert all(Path(item.wav).is_file() for item in selected)
        report["splits"][split] = {
            "count": len(selected),
            "id_sha256": audit["kept_id_sha256"],
        }
        if split == "train":
            persian = next(ex for ex in selected if persian_name in ex.wav)
            info = sf.info(persian.wav)
            assert (info.frames, info.samplerate) == (15345, 16000)
            assert (
                duration_view.read_kv(final / "utt2num_samples")[persian.uttid]
                == "16368"
            )
            report["persian"] = {
                "uttid": persian.uttid,
                "actual_samples": info.frames,
                "raw_copy_samples": 16368,
            }
    # A NEW/classwise-style wrong-side ID must fail preflight, not be repaired.
    valid_cs = next(
        line
        for line in (out / "valid_lidseq/wav.scp").read_text().splitlines()
        if line.startswith("cs_xtts_train_")
    )
    train_wavs = out / "train_lidseq/wav.scp"
    original = train_wavs.read_text()
    try:
        train_wavs.write_text(original + valid_cs + "\n")
        invalid = subprocess.run(
            ["bash", "local/verify_lid_data.sh", "--data_dir", str(out)],
            cwd=RECIPE / "lid1",
            env=env,
            capture_output=True,
            text=True,
        )
        assert (
            invalid.returncode != 0
            and "violates OLD global_hash 0.02" in invalid.stderr
        )
    finally:
        train_wavs.write_text(original)
    report["wrong_split_rejected"] = True
    report["generated_bytes"] = sum(
        p.stat().st_size
        for p in tmp_path.rglob("*")
        if p.is_file() and not p.is_symlink()
    )
    assert not list(out.rglob("*.wav"))
    (tmp_path / "real_offline_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
