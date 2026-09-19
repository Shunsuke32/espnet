"""Tests for VoxLingua source validation and recipe-local manifests."""

import numpy as np
import soundfile as sf

from egs3.voxlingua107.lid.dataset import Dataset
from egs3.voxlingua107.lid.dataset import builder as builder_module
from egs3.voxlingua107.lid.dataset.builder import VoxLingua107Builder


def _touch_wav(root, language):
    language_dir = root / language
    language_dir.mkdir(parents=True, exist_ok=True)
    (language_dir / "sample.wav").touch()


def test_source_requires_every_training_language(tmp_path, monkeypatch):
    """Require the full source language inventory before building."""
    monkeypatch.setattr(
        builder_module,
        "_ISO3_CODES",
        {"aa": "aaa", "bb": "bbb"},
    )
    _touch_wav(tmp_path / "dev", "aa")
    _touch_wav(tmp_path, "aa")
    _touch_wav(tmp_path, "unknown")
    builder = VoxLingua107Builder()

    assert not builder.is_source_prepared(source_dir=tmp_path)

    _touch_wav(tmp_path, "bb")

    assert builder.is_source_prepared(source_dir=tmp_path)


def test_built_requires_complete_training_metadata(tmp_path, monkeypatch):
    """Reject metadata missing a training language."""
    monkeypatch.setattr(
        builder_module,
        "_ISO3_CODES",
        {"aa": "aaa", "bb": "bbb"},
    )
    metadata_root = tmp_path / "data" / "voxlingua107"
    required = ("manifest.tsv", "utt2lang", "lang2utt", "category2utt")
    for split in ("train", "dev"):
        split_dir = metadata_root / split
        split_dir.mkdir(parents=True)
        for name in required:
            (split_dir / name).touch()
    category2utt = metadata_root / "train" / "category2utt"
    category2utt.write_text("aaa 0\n", encoding="utf-8")
    builder = VoxLingua107Builder()

    assert not builder.is_built(recipe_dir=tmp_path)

    category2utt.write_text("aaa 0\nbbb 1\n", encoding="utf-8")

    assert builder.is_built(recipe_dir=tmp_path)


def test_build_keeps_metadata_outside_source(tmp_path, monkeypatch):
    """Follow the ASR manifest layout without copying or changing source audio."""
    monkeypatch.setattr(builder_module, "_ISO3_CODES", {"aa": "aaa", "bb": "bbb"})
    source_dir = tmp_path / "source"
    recipe_dir = tmp_path / "recipe"
    for language in ("aa", "bb"):
        for split, filename in (("", "train.wav"), ("dev", "dev.wav")):
            path = source_dir / split / language / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, np.zeros(160), 16000)
    original_files = {path: path.read_bytes() for path in source_dir.rglob("*.wav")}
    builder = VoxLingua107Builder()
    builder.build(source_dir=source_dir, recipe_dir=recipe_dir)

    metadata_root = recipe_dir / "data" / "voxlingua107"
    assert builder.is_built(recipe_dir=recipe_dir)
    assert not (source_dir / "espnet3").exists()
    assert set(source_dir.rglob("*.wav")) == set(original_files)
    assert all(path.read_bytes() == content for path, content in original_files.items())
    dataset = Dataset("train", source_dir=source_dir, recipe_dir=recipe_dir)
    assert len(dataset) == 2
    assert dataset[0]["speech"].shape == (160,)
    assert dataset[1]["lid_labels"] == "bbb"
    assert (metadata_root / "train" / "category2utt").read_text() == "aaa 0\nbbb 1\n"

    other_data = tmp_path / "other_data"
    builder.build(source_dir=source_dir, data_dir=other_data)
    assert builder.is_built(data_dir=other_data)
    assert len(Dataset("dev", source_dir=source_dir, data_dir=other_data)) == 2
