"""Prepare LID metadata from an already formatted ESPnet2 evaluation split."""

import argparse
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from espnet2.fileio.read_text import read_2columns_text


def prepare_external_data(
    source_dir,
    output_dir,
    label_file="utt2lang",
    language=None,
    language_map=None,
    audio_root=None,
):
    """Convert existing ASR/LID metadata; do not download or copy audio.

    Args:
        source_dir: ESPnet2 data directory with an utterance-level ``wav.scp``.
            Use the existing ESPnet2 formatter first for pipes or segments.
        output_dir: New directory for ``wav.scp``, ``utt2lang`` and ``lang2utt``.
            Existing directories are rejected to protect earlier experiments.
        label_file: File in ``source_dir`` containing either one language code
            per utterance, or ``text`` with an initial ``[language]`` tag.
        language: Optional label for a monolingual split, instead of label_file.
        language_map: Optional two-column file mapping source labels to the
            model's ISO-639-3 labels. No dialect mapping is inferred.
        audio_root: Working directory used by the original ESPnet2 recipe to
            resolve relative wav.scp paths. Defaults to ``source_dir``.

    Returns:
        Path: The new data directory, consumed by ``src.external_dataset``.

    Raises:
        FileExistsError: The output directory already exists.
        ValueError: Labels, audio paths or source metadata are invalid.

    Notes:
        This only prepares metadata. Closed-set language filtering is performed
        by the Dataset against the actual model's ``train_lang2utt`` inventory.
        Neither audio samples nor the original metadata are modified.
    """
    source_root = Path(source_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if (source_root / "segments").exists():
        raise ValueError("Materialize segments with ESPnet2 format_wav_scp first.")
    paths = read_2columns_text(source_root / "wav.scp")
    if not paths:
        raise ValueError("wav.scp is empty.")
    if language is None:
        labels = read_2columns_text(source_root / label_file)
        if Path(label_file).name == "text":
            for key, text in labels.items():
                match = re.match(r"^\[([^\[\]\s]+)\](?:\s|$)", text)
                if match is None:
                    raise ValueError(f"Missing initial [language] tag for {key}.")
                labels[key] = match.group(1)
    else:
        labels = dict.fromkeys(paths, language)
    if set(paths) != set(labels):
        raise ValueError("wav.scp and the label file must have identical IDs.")
    mapping = read_2columns_text(language_map) if language_map else {}
    labels = {key: mapping.get(label, label) for key, label in labels.items()}
    if any(re.fullmatch(r"[a-z]{3}", label) is None for label in labels.values()):
        raise ValueError("Use language_map to map all source labels to ISO3 codes.")
    audio_root = Path(audio_root or source_root).expanduser().resolve()
    for key, value in paths.items():
        if value.rstrip().endswith("|"):
            raise ValueError("Materialize pipes with ESPnet2 format_wav_scp first.")
        path = Path(value)
        path = path if path.is_absolute() else audio_root / path
        if not path.is_file():
            raise ValueError(f"Audio path does not exist for {key}: {path}")
        paths[key] = str(path.resolve())
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".lid-data-", dir=output_root.parent) as temporary:
        staging = Path(temporary) / "data"
        staging.mkdir()
        for name, values in (("wav.scp", paths), ("utt2lang", labels)):
            (staging / name).write_text(
                "".join(f"{key} {values[key]}\n" for key in sorted(values)),
                encoding="utf-8",
            )
        (staging / "lang2utt").write_text(
            "".join(
                f"{label} {' '.join(sorted(k for k in labels if labels[k] == label))}\n"
                for label in sorted(set(labels.values()))
            ),
            encoding="utf-8",
        )
        staging.rename(output_root)
    return output_root


def main():
    """Run metadata preparation for a caller-selected, already acquired corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-file", default="utt2lang")
    parser.add_argument("--language")
    parser.add_argument("--language-map")
    parser.add_argument("--audio-root")
    prepare_external_data(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
