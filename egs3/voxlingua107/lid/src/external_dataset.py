"""Read prepared ESPnet2 language-identification evaluation data."""

import logging
from pathlib import Path

from torch.utils.data import Dataset as TorchDataset

from espnet2.fileio.read_text import read_2columns_text
from espnet2.fileio.sound_scp import SoundScpReader

logger = logging.getLogger(__name__)


class Dataset(TorchDataset):
    """Expose a Kaldi-style evaluation directory through the recipe Dataset API.

    Args:
        data_dir: Prepared directory with utterance-level ``wav.scp`` and
            ISO-639-3 ``utt2lang``. Audio must be at ``sample_rate``; pipe commands
            and ``segments`` must first be materialized with ESPnet2's
            ``pyscripts/audio/format_wav_scp.py``. Relative audio paths are
            interpreted relative to ``data_dir``.
        train_lang2utt: The training model's language inventory. Only exact
            language matches are retained, as in ESPnet2 ``prepare_ood_test``.
            Dialect labels are not silently mapped to a parent language.
        sample_rate: Expected audio sampling rate, in Hz.

    Raises:
        ValueError: Metadata keys disagree, unsupported audio input is present,
            or no language overlaps with the training inventory.
    """

    def __init__(self, data_dir, train_lang2utt, sample_rate=16000):
        """Read metadata and retain the model's supported languages."""
        data_root = Path(data_dir).expanduser().resolve()
        if (data_root / "segments").exists():
            raise ValueError("Materialize segments with ESPnet2 format_wav_scp first.")
        self.audio = SoundScpReader(data_root / "wav.scp", dtype="float32")
        self.labels = read_2columns_text(data_root / "utt2lang")
        if set(self.audio.keys()) != set(self.labels):
            raise ValueError("wav.scp and utt2lang must have identical utterance IDs.")
        if any(len(label.split()) != 1 for label in self.labels.values()):
            raise ValueError("utt2lang must contain exactly one label per utterance.")
        train_languages = read_2columns_text(train_lang2utt)
        self.utterance_ids = sorted(
            key for key, label in self.labels.items() if label in train_languages
        )
        logger.info(
            "External LID data %s: retained %d/%d utterances; "
            "languages=%s; excluded_languages=%s",
            data_root,
            len(self.utterance_ids),
            len(self.labels),
            sorted(set(self.labels.values()) & set(train_languages)),
            sorted(set(self.labels.values()) - set(train_languages)),
        )
        if not self.utterance_ids:
            raise ValueError("No evaluation languages match train_lang2utt.")
        for key in self.utterance_ids:
            path = self.audio.data[key]
            if path.rstrip().endswith("|"):
                raise ValueError("Materialize pipes with ESPnet2 format_wav_scp first.")
            if not Path(path).is_absolute():
                self.audio.data[key] = str(data_root / path)
        self.sample_rate = int(sample_rate)

    def __len__(self):
        """Return the number of utterances with model-supported language labels."""
        return len(self.utterance_ids)

    def __getitem__(self, idx):
        """Return speech and its language without adding model-unsupported fields."""
        utt_id = self.utterance_ids[int(idx)]
        sample_rate, speech = self.audio[utt_id]
        if sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz, got {sample_rate} Hz.")
        if speech.ndim == 2:
            speech = speech.mean(axis=1)
        return {"speech": speech, "lid_labels": self.labels[utt_id]}
