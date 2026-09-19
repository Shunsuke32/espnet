"""VoxLingua107 dataset backed by generated TSV manifests."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.voxlingua107.lid.dataset.builder import (
    VoxLingua107Builder,
    resolve_metadata_root,
    resolve_source_root,
)
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_KNOWN_SPLITS = {str(split) for split in _CONFIG["dataset"]["supported_splits"]}


@dataclass(frozen=True)
class VoxLingua107Example:
    """One manifest entry."""

    utt_id: str
    audio_path: Path
    language: str


def _read_manifest(path: Path) -> list[VoxLingua107Example]:
    examples = []
    with path.open("r", encoding="utf-8") as manifest:
        for raw_line in manifest:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            utt_id, audio_path, language = line.split("\t")
            examples.append(VoxLingua107Example(utt_id, Path(audio_path), language))
    if not examples:
        raise RuntimeError(f"Manifest is empty: {path}")
    return examples


class VoxLingua107Dataset(TorchDataset):
    """Dataset returning raw speech and a language label."""

    def __init__(
        self,
        split: str,
        source_dir: str | Path | None = None,
        sample_rate: int = 16000,
        recipe_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        """Read source audio through manifests in ``data_dir`` or recipe-local data."""
        self.split = str(split)
        if self.split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")

        self.source_root = resolve_source_root(source_dir)
        builder = VoxLingua107Builder()
        if not builder.is_source_prepared(source_dir=self.source_root):
            builder.prepare_source(source_dir=self.source_root)
        if not builder.is_built(recipe_dir=recipe_dir, data_dir=data_dir):
            builder.build(
                source_dir=self.source_root, recipe_dir=recipe_dir, data_dir=data_dir
            )

        manifest = (
            resolve_metadata_root(recipe_dir, data_dir) / self.split / "manifest.tsv"
        )
        self.examples = _read_manifest(manifest)
        self.sample_rate = int(sample_rate)

    def __len__(self) -> int:
        """Return the number of manifest entries."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Read one waveform and its language code without copying source audio."""
        example = self.examples[int(idx)]
        speech, sample_rate = sf.read(example.audio_path, dtype="float32")
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected {self.sample_rate} Hz, got {sample_rate} Hz: "
                f"{example.audio_path}"
            )
        if speech.ndim == 2:
            speech = speech.mean(axis=1)
        return {
            "speech": np.asarray(speech, dtype=np.float32),
            "lid_labels": example.language,
        }
