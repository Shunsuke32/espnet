#!/usr/bin/env python3
"""Download/create FLEURS TSV manifests with ESPnet's upstream data source.

This follows egs2/fleurs/asr1/local/create_dataset.py where possible, but
datasets>=4 no longer supports the google/xtreme_s dataset script.  In that
environment we use the official google/fleurs parquet dataset directly.  By
default, audio bytes are materialized under the recipe download directory so
the TSVs point to user-readable files.  The TSVs are then consumed by
local/prepare_fleurs_cs_lid_data.py.

For this LID recipe, we add explicit lang_id_name, num_samples, sampling_rate,
and duration columns so labels and train/validation duration filters are
auditable.  We never save audio as <id>.wav, which avoids collisions for
repeated FLEURS ids.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence


FIELDS = [
    # CommonVoice-like fields expected by ESPnet FLEURS data_prep.pl.
    "client_id",
    "path",
    "sentence",
    "upvotes",
    "downvotes",
    "age",
    "gender",
    "accent",
    # Extra columns consumed by this LID recipe.
    "id",
    "lang_id",
    "lang_id_name",
    "language",
    "num_samples",
    "sampling_rate",
    "duration",
]

PAPER_FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"


def clean_sentence(text: object) -> str:
    s = "" if text is None else str(text).strip()
    for i in range(8192, 8208):
        s = s.replace(chr(i), " ")
    for i in range(8232, 8240):
        s = s.replace(chr(i), " ")
    return s.replace(chr(160), " ")


def get_audio_path(row: Dict[str, Any]) -> str:
    path = row.get("path") or row.get("file_name")
    if path:
        return str(path)
    audio = row.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return str(audio["path"])
    raise ValueError(f"FLEURS row has no audio path. row keys={list(row.keys())}")


def safe_path_component(value: object, default: str = "item") -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or default


def materialize_audio(
    row: Dict[str, Any],
    *,
    raw_lang: str,
    hf_split: str,
    audio_root: Path,
) -> str:
    """Write google/fleurs audio bytes to a user-readable stable WAV path.

    In some sandboxed data-prep runs, datasets exposes extracted paths under
    /root/.cache.  Those paths are not readable by the actual ESPnet training
    process, so the manifest must point to audio files owned by this recipe.
    """
    audio = row.get("audio")
    audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
    src = (
        row.get("path")
        or row.get("file_name")
        or (audio.get("path") if isinstance(audio, dict) else None)
        or row.get("id")
    )
    src_text = str(src)
    src_name = Path(src_text).name if src_text else ""
    ext = Path(src_name).suffix or ".wav"
    stem = safe_path_component(Path(src_name).stem or row.get("id"), "audio")
    rid = safe_path_component(row.get("id"), "row")
    digest_src = f"{raw_lang}\t{hf_split}\t{row.get('id')}\t{src_text}"
    digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:16]
    dest = (
        audio_root
        / safe_path_component(raw_lang, "lang")
        / hf_split
        / f"{rid}_{stem}_{digest}{ext}"
    )

    if dest.exists() and dest.stat().st_size > 0:
        return str(dest.resolve())

    if audio_bytes is None:
        src_path = Path(src_text)
        if src_path.is_file() and os.access(src_path, os.R_OK):
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
            with src_path.open("rb") as fin, tmp.open("wb") as fout:
                while True:
                    chunk = fin.read(1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
            os.replace(tmp, dest)
            return str(dest.resolve())
        raise ValueError(
            "google/fleurs row has no audio bytes and source path is not readable: "
            f"lang={raw_lang} split={hf_split} id={row.get('id')} path={src_text}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(audio_bytes)
    os.replace(tmp, dest)
    return str(dest.resolve())


def maybe_int(x: object) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(x)
    except Exception:
        return None


def maybe_float(x: object) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def row_num_samples(row: Dict[str, Any]) -> Optional[int]:
    ns = maybe_int(row.get("num_samples")) or maybe_int(row.get("n_samples"))
    audio = row.get("audio")
    if ns is None and isinstance(audio, dict):
        ns = maybe_int(audio.get("num_samples"))
    return ns


def load_official_configs() -> Sequence[str]:
    prep = Path(__file__).with_name("prepare_fleurs_cs_lid_data.py")
    spec = importlib.util.spec_from_file_location("prepare_fleurs_cs_lid_data", prep)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {prep}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return tuple(mod.OFFICIAL_FLEURS_CONFIGS)


def normalize_configs(lang: str) -> Sequence[str]:
    if lang == "all" or lang == "fleurs.all":
        return load_official_configs()
    configs = []
    for item in lang.split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("fleurs."):
            item = item.split(".", 1)[1]
        configs.append(item)
    if not configs:
        raise ValueError("--lang produced an empty config list")
    return tuple(configs)


def datasets_major_version() -> Optional[int]:
    try:
        version = importlib.metadata.version("datasets")
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        return int(version.split(".", 1)[0])
    except ValueError:
        return None


def choose_source(source: str) -> str:
    if source != "auto":
        return source
    major = datasets_major_version()
    if major is not None and major >= 4:
        return "google_fleurs"
    return "xtreme_s"


def write_split(ds, split: str, out_tsv: Path, lang_names: Iterable[str]) -> None:
    names = list(lang_names)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    tmp_tsv = out_tsv.with_name(f".{out_tsv.name}.tmp.{os.getpid()}")
    try:
        with tmp_tsv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, delimiter="\t", fieldnames=FIELDS, extrasaction="ignore"
            )
            writer.writeheader()
            for row in ds[split]:
                lang_id = int(row["lang_id"])
                raw_lang = names[lang_id]
                ns = row_num_samples(row)
                sr = (
                    maybe_int(row.get("sampling_rate"))
                    or maybe_int(row.get("sample_rate"))
                    or 16000
                )
                dur = maybe_float(row.get("duration"))
                if dur is None and ns is not None and sr:
                    dur = float(ns) / float(sr)
                sent = clean_sentence(
                    row.get("transcription") or row.get("sentence") or ""
                )
                # Always include raw FLEURS label as a bracketed prefix for audit,
                # even for single-language configs.
                sent_with_lang = (
                    sent if sent.startswith("[") else f"[{raw_lang}] {sent}"
                )
                writer.writerow(
                    {
                        "client_id": row.get("id"),
                        "path": get_audio_path(row),
                        "sentence": sent_with_lang,
                        "upvotes": "",
                        "downvotes": "",
                        "age": "",
                        "gender": row.get("gender") or "",
                        "accent": row.get("language") or "",
                        "id": row.get("id"),
                        "lang_id": lang_id,
                        "lang_id_name": raw_lang,
                        "language": row.get("language") or "",
                        "num_samples": "" if ns is None else ns,
                        "sampling_rate": sr,
                        "duration": "" if dur is None else f"{dur:.9f}",
                    }
                )
        os.replace(tmp_tsv, out_tsv)
    except Exception:
        try:
            tmp_tsv.unlink()
        except FileNotFoundError:
            pass
        raise


def gender_name(value: object) -> str:
    if str(value) == "0":
        return "male"
    if str(value) == "1":
        return "female"
    return "" if value is None else str(value)


def google_fleurs_row(
    row: Dict[str, Any],
    *,
    raw_lang: str,
    lang_id: int,
    hf_split: Optional[str] = None,
    audio_root: Optional[Path] = None,
) -> Dict[str, object]:
    ns = row_num_samples(row)
    sr = (
        maybe_int(row.get("sampling_rate"))
        or maybe_int(row.get("sample_rate"))
        or 16000
    )
    dur = maybe_float(row.get("duration"))
    if dur is None and ns is not None and sr:
        dur = float(ns) / float(sr)
    sent = clean_sentence(
        row.get("raw_transcription")
        or row.get("transcription")
        or row.get("sentence")
        or ""
    )
    sent_with_lang = sent if sent.startswith("[") else f"[{raw_lang}] {sent}"
    rid = row.get("id")
    if audio_root is not None and hf_split is not None:
        wav_path = materialize_audio(
            row, raw_lang=raw_lang, hf_split=hf_split, audio_root=audio_root
        )
    else:
        wav_path = get_audio_path(row)
    return {
        "client_id": rid,
        "path": wav_path,
        "sentence": sent_with_lang,
        "upvotes": "",
        "downvotes": "",
        "age": "",
        "gender": gender_name(row.get("gender")),
        "accent": row.get("language") or raw_lang,
        "id": rid,
        "lang_id": lang_id,
        "lang_id_name": raw_lang,
        "language": row.get("language") or raw_lang,
        "num_samples": "" if ns is None else ns,
        "sampling_rate": sr,
        "duration": "" if dur is None else f"{dur:.9f}",
    }


def limited_rows(
    rows: Iterable[Dict[str, Any]], limit: int
) -> Iterator[Dict[str, Any]]:
    for i, row in enumerate(rows):
        if limit > 0 and i >= limit:
            break
        yield row


def ordered_langs_in_tsv(path: Path) -> list[str]:
    langs: list[str] = []
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            lang = str(row.get("lang_id_name") or "")
            if lang and lang not in seen:
                seen.add(lang)
                langs.append(lang)
    return langs


def existing_partial_tsvs(
    out_root: Path, split_map: Dict[str, str]
) -> Optional[Dict[str, Path]]:
    found: Dict[str, Path] = {}
    for hf_split, out_name in split_map.items():
        matches = sorted(
            out_root.glob(f".{out_name}.tsv.tmp.*"), key=lambda p: p.stat().st_mtime
        )
        if not matches:
            return None
        found[hf_split] = matches[-1]
    return found


def write_google_fleurs(
    *,
    configs: Sequence[str],
    out_root: Path,
    cache_dir: str,
    subsample_per_lang: int,
    resume_partial: bool,
    materialize_audio_files: bool,
    audio_root: Optional[Path],
    revision: str,
) -> None:
    from datasets import Audio, load_dataset  # type: ignore

    out_root.mkdir(parents=True, exist_ok=True)
    if materialize_audio_files and audio_root is None:
        audio_root = out_root / "audio"
    split_map = {"train": "train", "validation": "dev", "test": "test"}
    writers = {}
    files = {}
    final_paths = {}
    tmp_paths = {}
    success = False
    start_idx = 0
    resume_paths = (
        existing_partial_tsvs(out_root, split_map) if resume_partial else None
    )
    try:
        for hf_split, out_name in split_map.items():
            p = out_root / f"{out_name}.tsv"
            if resume_paths is not None:
                tmp = resume_paths[hf_split]
                f = tmp.open("a", encoding="utf-8", newline="")
            else:
                tmp = out_root / f".{out_name}.tsv.tmp.{os.getpid()}"
                f = tmp.open("w", encoding="utf-8", newline="")
            files[hf_split] = f
            final_paths[hf_split] = p
            tmp_paths[hf_split] = tmp
            w = csv.DictWriter(
                f, delimiter="\t", fieldnames=FIELDS, extrasaction="ignore"
            )
            if resume_paths is None:
                w.writeheader()
            writers[hf_split] = w

        if resume_paths is not None:
            lang_lists = {
                split: ordered_langs_in_tsv(path)
                for split, path in resume_paths.items()
            }
            train_langs = lang_lists["train"]
            if any(langs != train_langs for langs in lang_lists.values()):
                raise RuntimeError(
                    f"partial TSV split language lists disagree: {lang_lists}"
                )
            if tuple(train_langs) != tuple(configs[: len(train_langs)]):
                raise RuntimeError(
                    "partial TSV language order does not match requested configs: "
                    f"partial={train_langs[:5]}... n={len(train_langs)}"
                )
            start_idx = len(train_langs)
            print(
                f"Resuming google/fleurs TSV creation from config {start_idx + 1}/{len(configs)}",
                flush=True,
            )

        for lang_id, cfg in enumerate(configs[start_idx:], start=start_idx):
            print(
                f"Loading google/fleurs config {lang_id + 1}/{len(configs)}: {cfg}",
                flush=True,
            )
            for hf_split in split_map:
                ds_split = load_dataset(
                    "google/fleurs",
                    cfg,
                    revision=revision,
                    split=hf_split,
                    streaming=True,
                    cache_dir=cache_dir,
                )
                if materialize_audio_files:
                    ds_split = ds_split.cast_column("audio", Audio(decode=False))
                for row in limited_rows(ds_split, subsample_per_lang):
                    writers[hf_split].writerow(
                        google_fleurs_row(
                            row,
                            raw_lang=cfg,
                            lang_id=lang_id,
                            hf_split=hf_split,
                            audio_root=audio_root if materialize_audio_files else None,
                        )
                    )
        success = True
    finally:
        for f in files.values():
            f.close()
        if success:
            for hf_split in split_map:
                os.replace(tmp_paths[hf_split], final_paths[hf_split])
        elif resume_paths is None:
            for tmp in tmp_paths.values():
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass

    print(f"Wrote FLEURS TSV manifests to {out_root} using google/fleurs")
    (out_root / "source.json").write_text(
        json.dumps(
            {
                "dataset": "google/fleurs",
                "revision": revision,
                "configs": list(configs),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--lang", default="all", help="FLEURS config, e.g. all or en_us"
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="output directory, e.g. ${FLEURS}/all",
    )
    parser.add_argument("--cache_dir", default="downloads/cache")
    parser.add_argument("--revision", default=PAPER_FLEURS_REVISION)
    parser.add_argument(
        "--nlsyms_txt",
        default="data/nlsyms.txt",
        help="deprecated compatibility option; canonical nlsyms are generated by prepare_fleurs_cs_lid_data.py",
    )
    parser.add_argument("--subsample_per_lang", type=int, default=0)
    parser.add_argument(
        "--resume_partial", type=str, default="true", choices=("true", "false")
    )
    parser.add_argument(
        "--materialize_audio",
        type=str,
        default="true",
        choices=("true", "false"),
        help="for google/fleurs, write audio bytes under --audio_root and put those readable paths in TSVs",
    )
    parser.add_argument(
        "--audio_root",
        type=Path,
        default=None,
        help="audio output dir; default: <out_root>/audio",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "xtreme_s", "google_fleurs"),
        default="google_fleurs",
        help="FLEURS source. The paper profile uses the pinned google/fleurs revision.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Install datasets first: python3 -m pip install datasets soundfile"
        ) from e

    source = choose_source(args.source)
    if source == "google_fleurs":
        write_google_fleurs(
            configs=normalize_configs(args.lang),
            out_root=args.out_root,
            cache_dir=args.cache_dir,
            subsample_per_lang=args.subsample_per_lang,
            resume_partial=args.resume_partial == "true",
            materialize_audio_files=args.materialize_audio == "true",
            audio_root=args.audio_root,
            revision=args.revision,
        )
        return

    config = args.lang if args.lang.startswith("fleurs.") else f"fleurs.{args.lang}"
    try:
        ds = load_dataset("google/xtreme_s", config, cache_dir=args.cache_dir)
    except Exception as e:
        if args.source == "xtreme_s":
            raise
        print(
            f"google/xtreme_s load failed ({type(e).__name__}: {e}); falling back to google/fleurs"
        )
        write_google_fleurs(
            configs=normalize_configs(args.lang),
            out_root=args.out_root,
            cache_dir=args.cache_dir,
            subsample_per_lang=args.subsample_per_lang,
            resume_partial=args.resume_partial == "true",
            materialize_audio_files=args.materialize_audio == "true",
            audio_root=args.audio_root,
            revision=args.revision,
        )
    else:
        if "lang_id" not in ds["train"].features:
            raise KeyError("google/xtreme_s FLEURS has no lang_id feature")
        names = ds["train"].features["lang_id"].names
        split_map = {"train": "train", "validation": "dev", "test": "test"}
        for hf_split, out_name in split_map.items():
            write_split(ds, hf_split, args.out_root / f"{out_name}.tsv", names)
        print(f"Wrote FLEURS TSV manifests to {args.out_root}")


if __name__ == "__main__":
    main()
