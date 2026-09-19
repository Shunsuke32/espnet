"""Tests for the ESPnet2 evaluation-data adapter."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs2.TEMPLATE.lid1.local.prepare_ood_test import main as prepare_ood_test
from egs3.voxlingua107.lid.src.external_dataset import Dataset
from espnet2.fileio.read_text import read_2columns_text
from espnet3.components.data.dataset_module import instantiate_dataset_reference
from espnet3.utils.config_utils import load_and_merge_config


def _prepare_fixture(root):
    train_dir = root / "train"
    eval_dir = root / "eval"
    train_dir.mkdir()
    eval_dir.mkdir()
    (train_dir / "lang2utt").write_text("eng 0\njpn 1\n", encoding="utf-8")
    (eval_dir / "lang2utt").write_text("eng u2\njpn u1\nxyz u3\n", encoding="utf-8")
    (eval_dir / "utt2lang").write_text("u2 eng\nu1 jpn\nu3 xyz\n", encoding="utf-8")
    wave = root / "audio.wav"
    sf.write(wave, np.zeros((160, 2)), 16000)
    (eval_dir / "wav.scp").write_text(
        "".join(f"u{i} {wave}\n" for i in (2, 1, 3)), encoding="utf-8"
    )
    return train_dir / "lang2utt", eval_dir


def test_filter_matches_espnet2_and_preserves_integer_index_contract(tmp_path):
    """Match ESPnet2's OOD intersection while returning only model input fields."""
    inventory, data_dir = _prepare_fixture(tmp_path)
    prepare_ood_test(
        argparse.Namespace(dump_dir=str(tmp_path), train_set="train", test_sets="eval")
    )
    expected = read_2columns_text(tmp_path / "eval_cross_train" / "utt2lang")
    dataset = instantiate_dataset_reference(
        {
            "data_src": "egs3.voxlingua107.lid.src.external_dataset",
            "data_src_args": {
                "data_dir": str(data_dir),
                "train_lang2utt": str(inventory),
            },
        }
    )

    assert dataset.utterance_ids == sorted(expected)
    assert len(dataset) == 2
    for idx, utt_id in enumerate(dataset.utterance_ids):
        sample = dataset[idx]
        assert set(sample) == {"speech", "lid_labels"}
        assert sample["lid_labels"] == expected[utt_id]
        assert sample["speech"].shape == (160,)
        assert sample["speech"].dtype == np.float32


@pytest.mark.parametrize(
    "invalid", ["mismatched_ids", "segments", "pipe", "no_overlap"]
)
def test_invalid_metadata_fails_clearly(tmp_path, invalid):
    """Reject incomplete metadata and unformatted audio inputs."""
    inventory, data_dir = _prepare_fixture(tmp_path)
    if invalid == "mismatched_ids":
        (data_dir / "utt2lang").write_text("u1 jpn\n", encoding="utf-8")
    elif invalid == "segments":
        (data_dir / "segments").touch()
    elif invalid == "pipe":
        (data_dir / "wav.scp").write_text(
            "u1 cat audio.wav |\nu2 audio.wav\nu3 audio.wav\n", encoding="utf-8"
        )
    else:
        inventory.write_text("fra 0\n", encoding="utf-8")

    with pytest.raises(ValueError):
        Dataset(data_dir, inventory)


def test_wrong_sample_rate_is_not_silently_accepted(tmp_path):
    """Keep resampling explicit instead of feeding the wrong rate to MMS."""
    inventory, data_dir = _prepare_fixture(tmp_path)
    sf.write(tmp_path / "audio.wav", np.zeros(80), 8000)
    dataset = Dataset(data_dir, inventory)
    with pytest.raises(ValueError, match="Expected 16000 Hz"):
        dataset[0]


def test_relative_audio_paths_are_relative_to_data_directory(tmp_path):
    """Resolve prepared relative paths independently of the inference directory."""
    inventory, data_dir = _prepare_fixture(tmp_path)
    (data_dir / "wav.scp").write_text(
        "".join(f"u{i} ../audio.wav\n" for i in (1, 2, 3)), encoding="utf-8"
    )
    assert Dataset(data_dir, inventory)[0]["speech"].shape == (160,)


def test_language_filter_reports_evaluation_denominator(tmp_path, caplog):
    """Expose retained counts and excluded languages in the stage log."""
    inventory, data_dir = _prepare_fixture(tmp_path)
    with caplog.at_level(logging.INFO):
        Dataset(data_dir, inventory)
    assert "retained 2/3 utterances" in caplog.text
    assert "languages=['eng', 'jpn']" in caplog.text
    assert "excluded_languages=['xyz']" in caplog.text


def test_external_config_uses_model_inventory_for_all_ood_sets():
    """Use the model's inventory for all five external evaluation splits."""
    config = load_and_merge_config(
        Path("egs3/voxlingua107/lid/conf/inference_external.yaml"),
        config_name="inference.yaml",
        default_package="egs3.TEMPLATE.lid",
        resolve=False,
    )
    assert [entry.name for entry in config.dataset.test] == [
        "babel",
        "fleurs",
        "ml_superb2",
        "ml_superb2_dialect",
        "voxpopuli",
    ]
    for entry in config.dataset.test:
        assert entry.data_src.endswith("external_dataset")
        assert entry.data_src_args.train_lang2utt == config.model.lang2utt
