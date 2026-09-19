"""Tests for metadata preparation without downloading external corpora."""

import numpy as np
import pytest
import soundfile as sf

from egs3.voxlingua107.lid.src.external_data import prepare_external_data
from egs3.voxlingua107.lid.src.external_dataset import Dataset
from espnet2.fileio.read_text import read_2columns_text


def _prepare_fixture(root):
    source = root / "source"
    source.mkdir()
    for idx in (1, 2):
        sf.write(root / f"{idx}.wav", np.zeros(160), 16000)
    (source / "wav.scp").write_text("u2 2.wav\nu1 1.wav\n", encoding="utf-8")
    return source


def test_asr_tags_explicit_mapping_and_audio_paths(tmp_path):
    """Convert language-tagged ASR text without modifying source audio or metadata."""
    source = _prepare_fixture(tmp_path)
    (source / "text").write_text(
        "u1 [en_us] hello\nu2 [org_jpn] example\n", encoding="utf-8"
    )
    mapping = tmp_path / "language_map"
    mapping.write_text("en_us eng\norg_jpn jpn\n", encoding="utf-8")
    target = prepare_external_data(
        source,
        tmp_path / "prepared",
        label_file="text",
        language_map=mapping,
        audio_root=tmp_path,
    )
    assert read_2columns_text(target / "utt2lang") == {"u1": "eng", "u2": "jpn"}
    assert read_2columns_text(target / "wav.scp") == {
        "u1": str(tmp_path / "1.wav"),
        "u2": str(tmp_path / "2.wav"),
    }
    assert set(target.iterdir()) == {
        target / "wav.scp",
        target / "utt2lang",
        target / "lang2utt",
    }
    assert len(Dataset(target, target / "lang2utt")) == 2
    assert (source / "wav.scp").read_text() == "u2 2.wav\nu1 1.wav\n"


def test_monolingual_input_and_existing_output_protection(tmp_path):
    """Support explicit monolingual labels and never replace existing outputs."""
    source = _prepare_fixture(tmp_path)
    output = tmp_path / "prepared"
    prepare_external_data(source, output, language="eng", audio_root=tmp_path)
    assert read_2columns_text(output / "utt2lang") == {"u1": "eng", "u2": "eng"}
    with pytest.raises(FileExistsError):
        prepare_external_data(source, output, language="jpn", audio_root=tmp_path)
    assert read_2columns_text(output / "utt2lang")["u1"] == "eng"


def test_formatted_audio_can_reuse_original_absolute_text_path(tmp_path):
    """Read labels left in the original recipe after ESPnet2 formats audio."""
    source = _prepare_fixture(tmp_path)
    label_dir = tmp_path / "original"
    label_dir.mkdir()
    text = label_dir / "text"
    text.write_text("u1 [eng] hello\nu2 [jpn] example\n", encoding="utf-8")
    output = prepare_external_data(
        source, tmp_path / "prepared", label_file=text, audio_root=tmp_path
    )
    assert read_2columns_text(output / "utt2lang") == {"u1": "eng", "u2": "jpn"}


def test_dialect_labels_are_not_implicitly_collapsed(tmp_path):
    """Retain dialect labels in metadata and filter only against the model inventory."""
    source = _prepare_fixture(tmp_path)
    (source / "utt2lang").write_text("u1 cmn\nu2 yue\n", encoding="utf-8")
    output = prepare_external_data(source, tmp_path / "prepared", audio_root=tmp_path)
    inventory = tmp_path / "train_lang2utt"
    inventory.write_text("cmn 0\n", encoding="utf-8")
    dataset = Dataset(output, inventory)
    assert dataset.utterance_ids == ["u1"]
    assert read_2columns_text(output / "utt2lang")["u2"] == "yue"


@pytest.mark.parametrize("label", ["en_us", "english", "eng jpn", ""])
def test_non_iso3_labels_require_an_explicit_mapping(tmp_path, label):
    """Reject guessed or malformed ISO3 labels."""
    source = _prepare_fixture(tmp_path)
    with pytest.raises(ValueError, match="ISO3"):
        prepare_external_data(
            source, tmp_path / "prepared", language=label, audio_root=tmp_path
        )
    assert not (tmp_path / "prepared").exists()


def test_unlabeled_asr_text_is_rejected(tmp_path):
    """Do not mistake the first transcription word for a language code."""
    source = _prepare_fixture(tmp_path)
    (source / "text").write_text("u1 hello\nu2 example\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing initial"):
        prepare_external_data(source, tmp_path / "prepared", label_file="text")


def test_segments_require_espnet2_formatter(tmp_path):
    """Require the shared ESPnet2 audio preparation for recording-level inputs."""
    source = _prepare_fixture(tmp_path)
    (source / "segments").write_text("u1 rec1 0 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Materialize segments"):
        prepare_external_data(source, tmp_path / "prepared", language="eng")
