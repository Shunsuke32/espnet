#!/usr/bin/env python3
"""Prepare CS-YODAS views compatible with the FLEURS/CS-FLEURS LID data.

This script is intentionally manifest-only by default.  It does not overwrite
the existing train_lidseq/train_fleurs_lid data directories.  It creates new
ESPnet/Kaldi data directories for:

* ASR-style/LID2/LID3 two-label sequence training.
* LID1 pair-as-one-class training.

CS-YODAS train/valid examples can be duration-capped independently from the
already-prepared FLEURS/CS-FLEURS manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


LOGGER = logging.getLogger("prepare_cs_yodas_views")

LANG_CONFIGS = ("ara", "cmn", "fra", "hin", "jpn", "rus")
BASE_ENGLISH_NAMES = {
    "ara": "Arabic",
    "cmn": "Chinese",
    "fra": "French",
    "hin": "Hindi",
    "jpn": "Japanese",
    "rus": "Russian",
}
LANGUAGE_NAME_TO_CANONICAL = {
    "Arabic": "ara",
    "Chinese": "cmn",
    "French": "fra",
    "Hindi": "hin",
    "Japanese": "jpn",
    "Russian": "rus",
    "English": "eng",
}
KALDI_UTTID_RE = re.compile(
    r"(?P<video>.+)_(?P<lang>[a-z]{3})_(?P<start>\d{9})_(?P<end>\d{9})$"
)
ASR_SEGMENT_RE = re.compile(r".+_asr_(?P<start>\d{9})_(?P<end>\d{9})$")
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.=-]+")


@dataclass(frozen=True)
class Example:
    uttid: str
    wav: str
    labels: Tuple[str, ...]
    speaker: str
    source: str
    subset: str
    raw_label_value: str
    raw_label_parts: Tuple[str, ...]
    duration_sec: Optional[float]
    num_samples: Optional[int]
    metadata: Mapping[str, object] = field(default_factory=dict)


def sanitize_id(value: object) -> str:
    text = SAFE_ID_RE.sub("_", str(value)).strip("._-")
    return text or "empty"


def canonical_pair(labels: Sequence[str]) -> str:
    uniq: List[str] = []
    for lab in labels:
        if lab not in uniq:
            uniq.append(lab)
    if len(uniq) == 1:
        return uniq[0]
    if len(uniq) != 2:
        raise ValueError(f"pair-class view supports one or two labels, got {labels}")
    if "eng" in uniq:
        other = uniq[0] if uniq[1] == "eng" else uniq[1]
        return f"{other}-eng"
    return "-".join(sorted(uniq))


def token_text(labels: Sequence[str]) -> str:
    return " ".join(f"<{lab}>" for lab in labels)


def read_kv(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            data[key] = value
    return data


def read_labels_jsonl(path: Path) -> Dict[str, Mapping[str, object]]:
    data: Dict[str, Mapping[str, object]] = {}
    if not path.exists():
        return data
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[str(obj["uttid"])] = obj
    return data


def read_existing_data_dir(data_dir: Path) -> List[Example]:
    wavs = read_kv(data_dir / "wav.scp")
    utt2langs = read_kv(data_dir / "utt2langs")
    utt2spk = read_kv(data_dir / "utt2spk") if (data_dir / "utt2spk").exists() else {}
    utt2category = (
        read_kv(data_dir / "utt2category")
        if (data_dir / "utt2category").exists()
        else {}
    )
    meta = read_labels_jsonl(data_dir / "labels.jsonl")
    examples: List[Example] = []
    for uttid in sorted(wavs):
        labels = tuple(utt2langs[uttid].split())
        m = meta.get(uttid, {})
        examples.append(
            Example(
                uttid=uttid,
                wav=wavs[uttid],
                labels=labels,
                speaker=utt2spk.get(uttid, uttid),
                source=str(utt2category.get(uttid) or m.get("source") or "existing"),
                subset=str(m.get("subset") or data_dir.name),
                raw_label_value=str(m.get("raw_label_value") or " ".join(labels)),
                raw_label_parts=tuple(str(x) for x in m.get("raw_label_parts", labels)),
                duration_sec=float(m["duration_sec"])
                if m.get("duration_sec") is not None
                else None,
                num_samples=int(m["num_samples"])
                if m.get("num_samples") is not None
                else None,
                metadata=m,
            )
        )
    return examples


def parse_context_duration(kaldi_uttid: str) -> Tuple[str, float, int, int]:
    match = KALDI_UTTID_RE.match(kaldi_uttid)
    if not match:
        raise ValueError(f"cannot parse kaldi_uttid: {kaldi_uttid}")
    start = int(match.group("start"))
    end = int(match.group("end"))
    return match.group("video"), (end - start) / 1000.0, start, end


def parse_asr_offsets(record_id: str) -> Tuple[Optional[float], Optional[float]]:
    match = ASR_SEGMENT_RE.match(record_id)
    if not match:
        return None, None
    return int(match.group("start")) / 1000.0, int(match.group("end")) / 1000.0


def is_exact_base_english(config: str, obj: Mapping[str, object]) -> bool:
    languages = obj.get("languages")
    return (
        isinstance(languages, list)
        and len(languages) == 2
        and set(languages) == {BASE_ENGLISH_NAMES[config], "English"}
    )


def load_yodas_records(metadata_dir: Path, audio_root: Path) -> List[Example]:
    examples: List[Example] = []
    for config in LANG_CONFIGS:
        path = metadata_dir / f"{config}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing CS-YODAS metadata: {path}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not is_exact_base_english(config, obj):
                    continue
                # Keep the sequence target stable even if metadata happens to
                # list English first.  Raw ordering remains in the audit fields.
                labels = (config, "eng")
                video_id, duration, context_start_ms, context_end_ms = (
                    parse_context_duration(obj["kaldi_uttid"])
                )
                asr_start, asr_end = parse_asr_offsets(obj["id"])
                uttid = f"csyodas_{config}_{sanitize_id(obj['id'])}"
                wav = str(audio_root / obj["wav_path"])
                metadata = {
                    "cs_yodas_id": obj["id"],
                    "config": config,
                    "language": obj.get("language"),
                    "languages": obj.get("languages"),
                    "text": obj.get("text"),
                    "kaldi_uttid": obj.get("kaldi_uttid"),
                    "video_id": video_id,
                    "wav_path": obj.get("wav_path"),
                    "context_start_ms": context_start_ms,
                    "context_end_ms": context_end_ms,
                    "asr_start_sec": asr_start,
                    "asr_end_sec": asr_end,
                    "consistent": obj.get("consistent"),
                    "Q1": obj.get("Q1"),
                    "Q2": obj.get("Q2"),
                    "Q3": obj.get("Q3"),
                    "Q4": obj.get("Q4"),
                    "Q5": obj.get("Q5"),
                }
                examples.append(
                    Example(
                        uttid=uttid,
                        wav=wav,
                        labels=labels,
                        speaker=uttid,
                        source="cs_yodas",
                        subset=config,
                        raw_label_value="+".join(obj["languages"]),
                        raw_label_parts=tuple(obj["languages"]),
                        duration_sec=duration,
                        num_samples=None,
                        metadata=metadata,
                    )
                )
    return examples


def split_yodas(
    examples: Sequence[Example],
    train_ratio: float,
    valid_ratio: float,
    seed: str,
) -> Dict[str, List[Example]]:
    by_lang: Dict[str, Dict[str, List[Example]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ex in examples:
        lang = ex.labels[0]
        video_id = str(ex.metadata["video_id"])
        by_lang[lang][video_id].append(ex)

    split = {"train": [], "valid": [], "test": []}
    for lang, groups in sorted(by_lang.items()):
        items = list(groups.items())
        items.sort(
            key=lambda kv: hashlib.sha1(f"{seed}\t{lang}\t{kv[0]}".encode()).hexdigest()
        )
        total = sum(len(v) for _, v in items)
        train_target = total * train_ratio
        valid_target = total * valid_ratio
        train_count = valid_count = 0
        for _, group in items:
            if train_count < train_target:
                dst = "train"
                train_count += len(group)
            elif valid_count < valid_target:
                dst = "valid"
                valid_count += len(group)
            else:
                dst = "test"
            split[dst].extend(group)
        LOGGER.info(
            "CS-YODAS split %s: total=%d train=%d valid=%d test=%d",
            lang,
            total,
            sum(1 for ex in split["train"] if ex.labels[0] == lang),
            sum(1 for ex in split["valid"] if ex.labels[0] == lang),
            sum(1 for ex in split["test"] if ex.labels[0] == lang),
        )
    return {k: sorted(v, key=lambda e: e.uttid) for k, v in split.items()}


def duration_cap(
    examples: Sequence[Example], max_duration: float
) -> Tuple[List[Example], List[Example]]:
    kept: List[Example] = []
    dropped: List[Example] = []
    for ex in examples:
        if ex.duration_sec is not None and ex.duration_sec >= max_duration:
            dropped.append(ex)
        else:
            kept.append(ex)
    return kept, dropped


def write_spk2utt(utt2spk: Mapping[str, str], path: Path) -> None:
    spk2utts: Dict[str, List[str]] = defaultdict(list)
    for utt, spk in sorted(utt2spk.items()):
        spk2utts[spk].append(utt)
    with path.open("w", encoding="utf-8") as f:
        for spk, utts in sorted(spk2utts.items()):
            f.write(f"{spk} {' '.join(sorted(utts))}\n")


def write_lang2utt(examples: Sequence[Example], path: Path) -> None:
    lang2utts: Dict[str, List[str]] = defaultdict(list)
    for ex in examples:
        lang2utts[ex.labels[0]].append(ex.uttid)
    with path.open("w", encoding="utf-8") as f:
        for lang, utts in sorted(lang2utts.items()):
            f.write(f"{lang} {' '.join(sorted(set(utts)))}\n")


def write_data_dir(name: str, examples: Sequence[Example], outdir: Path) -> None:
    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)
    examples = sorted(examples, key=lambda e: e.uttid)
    utt2spk = {ex.uttid: ex.uttid for ex in examples}
    write_utt2dur = bool(examples) and all(
        ex.duration_sec is not None for ex in examples
    )
    with (
        (d / "wav.scp").open("w", encoding="utf-8") as wav_f,
        (d / "text").open("w", encoding="utf-8") as text_f,
        (d / "utt2spk").open("w", encoding="utf-8") as u2s_f,
        (d / "utt2lang").open("w", encoding="utf-8") as u2l_f,
        (d / "utt2langs").open("w", encoding="utf-8") as u2ls_f,
        (d / "utt2category").open("w", encoding="utf-8") as u2c_f,
        (d / "labels.jsonl").open("w", encoding="utf-8") as js_f,
    ):
        for ex in examples:
            wav_f.write(f"{ex.uttid} {ex.wav}\n")
            text_f.write(f"{ex.uttid} {token_text(ex.labels)}\n")
            u2s_f.write(f"{ex.uttid} {utt2spk[ex.uttid]}\n")
            u2l_f.write(f"{ex.uttid} {ex.labels[0]}\n")
            u2ls_f.write(f"{ex.uttid} {' '.join(ex.labels)}\n")
            u2c_f.write(f"{ex.uttid} {ex.source}\n")
            payload = {
                **dict(ex.metadata),
                "uttid": ex.uttid,
                "wav": ex.wav,
                "labels": list(ex.labels),
                "speaker": ex.speaker,
                "source": ex.source,
                "subset": ex.subset,
                "raw_label_value": ex.raw_label_value,
                "raw_label_parts": list(ex.raw_label_parts),
                "duration_sec": ex.duration_sec,
                "num_samples": ex.num_samples,
            }
            js_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    if write_utt2dur:
        with (d / "utt2dur").open("w", encoding="utf-8") as dur_f:
            for ex in examples:
                dur_f.write(f"{ex.uttid} {ex.duration_sec:.6f}\n")
    else:
        (d / "utt2dur").unlink(missing_ok=True)
    write_spk2utt(utt2spk, d / "spk2utt")
    write_lang2utt(examples, d / "lang2utt")
    LOGGER.info(
        "wrote %s (%d utts, %d labels)",
        d,
        len(examples),
        len({e.labels[0] for e in examples}),
    )


def to_pair_examples(
    examples: Sequence[Example], source_suffix: str = "pair"
) -> List[Example]:
    out: List[Example] = []
    for ex in examples:
        pair = canonical_pair(ex.labels)
        meta = dict(ex.metadata)
        meta["original_labels"] = list(ex.labels)
        out.append(
            Example(
                uttid=ex.uttid,
                wav=ex.wav,
                labels=(pair,),
                speaker=ex.speaker,
                source=f"{ex.source}_{source_suffix}",
                subset=ex.subset,
                raw_label_value=pair,
                raw_label_parts=ex.labels,
                duration_sec=ex.duration_sec,
                num_samples=ex.num_samples,
                metadata=meta,
            )
        )
    return out


def write_inventory(name: str, examples: Sequence[Example], outdir: Path) -> None:
    counts: Dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[" ".join(ex.labels)] += 1
    path = outdir / "local" / f"{name}.label_inventory.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("label_sequence\tnum_utts\n")
        for labels, count in sorted(counts.items()):
            f.write(f"{labels}\t{count}\n")
    LOGGER.info("wrote %s", path)


def write_dropped(path: Path, examples: Sequence[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("uttid\tduration_sec\tlabels\twav\n")
        for ex in sorted(examples, key=lambda e: e.uttid):
            f.write(f"{ex.uttid}\t{ex.duration_sec}\t{' '.join(ex.labels)}\t{ex.wav}\n")


def ensure_no_split_leak(split: Mapping[str, Sequence[Example]]) -> None:
    owner: Dict[str, str] = {}
    for name, examples in split.items():
        for ex in examples:
            video_id = ex.metadata.get("video_id")
            if video_id is None:
                continue
            prev = owner.get(str(video_id))
            if prev is not None and prev != name:
                raise RuntimeError(
                    f"video_id leak across splits: {video_id} in {prev} and {name}"
                )
            owner[str(video_id)] = name


def link_outputs(
    outdir: Path, link_roots: Sequence[Path], names: Sequence[str]
) -> None:
    for root in link_roots:
        root.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = outdir / name
            dst = root / name
            if dst.exists() or dst.is_symlink():
                continue
            try:
                target = src.resolve().relative_to(dst.parent.resolve())
                dst.symlink_to(target)
            except ValueError:
                dst.symlink_to(src.resolve())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source_data_dir", type=Path, default=Path("lid1/data"))
    p.add_argument("--outdir", type=Path, default=Path("lid1/data"))
    p.add_argument("--metadata_dir", type=Path, required=True)
    p.add_argument("--audio_root", type=Path, required=True)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--valid_ratio", type=float, default=0.1)
    p.add_argument("--seed", default="cs-yodas-v1")
    p.add_argument("--max_yodas_train_valid_duration", type=float, default=70.0)
    p.add_argument("--prefix", default="")
    p.add_argument("--link_data_roots", type=Path, nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    src = args.source_data_dir
    outdir = args.outdir

    yodas = load_yodas_records(args.metadata_dir, args.audio_root)
    split = split_yodas(yodas, args.train_ratio, args.valid_ratio, args.seed)
    ensure_no_split_leak(split)
    y_train, drop_train = duration_cap(
        split["train"], args.max_yodas_train_valid_duration
    )
    y_valid, drop_valid = duration_cap(
        split["valid"], args.max_yodas_train_valid_duration
    )
    y_test = split["test"]
    LOGGER.info(
        "CS-YODAS selected: raw=%d train=%d(drop %d) valid=%d(drop %d) test=%d(no duration cap)",
        len(yodas),
        len(y_train),
        len(drop_train),
        len(y_valid),
        len(drop_valid),
        len(y_test),
    )

    existing_train = read_existing_data_dir(src / "train_lidseq")
    existing_valid = read_existing_data_dir(src / "valid_lidseq")
    fleurs_train = read_existing_data_dir(src / "train_fleurs_lid")
    fleurs_valid = read_existing_data_dir(src / "valid_fleurs_lid")
    existing_tests = {
        "test_lid_pair_fleurs": read_existing_data_dir(src / "test_fleurs_lid"),
        "test_lid_pair_cs_read_test": read_existing_data_dir(src / "test_cs_read_test"),
        "test_lid_pair_cs_xtts_test1": read_existing_data_dir(
            src / "test_cs_xtts_test1"
        ),
        "test_lid_pair_cs_xtts_test2": read_existing_data_dir(
            src / "test_cs_xtts_test2"
        ),
        "test_lid_pair_cs_mms_test": read_existing_data_dir(src / "test_cs_mms_test"),
        "test_lid_pair_cs_all": read_existing_data_dir(src / "test_cs_all"),
    }

    names: List[str] = []
    specs = {
        f"{args.prefix}train_yodas_lidseq": y_train,
        f"{args.prefix}valid_yodas_lidseq": y_valid,
        f"{args.prefix}test_yodas_lidseq": y_test,
        f"{args.prefix}train_lidseq_csall_yodas": [*existing_train, *y_train],
        f"{args.prefix}valid_lidseq_csall_yodas": [*existing_valid, *y_valid],
        f"{args.prefix}train_lid_pair_cs": to_pair_examples(existing_train),
        f"{args.prefix}valid_lid_pair_cs": to_pair_examples(existing_valid),
        f"{args.prefix}train_lid_pair_csall_yodas": to_pair_examples(
            [*existing_train, *y_train]
        ),
        f"{args.prefix}valid_lid_pair_csall_yodas": to_pair_examples(
            [*existing_valid, *y_valid]
        ),
        f"{args.prefix}train_lid_pair_fleurs": to_pair_examples(fleurs_train),
        f"{args.prefix}valid_lid_pair_fleurs": to_pair_examples(fleurs_valid),
        f"{args.prefix}test_lid_pair_yodas": to_pair_examples(y_test),
    }
    for name, examples in existing_tests.items():
        specs[f"{args.prefix}{name}"] = to_pair_examples(examples)
    for name, examples in specs.items():
        write_data_dir(name, examples, outdir)
        write_inventory(name, examples, outdir)
        names.append(name)

    write_dropped(
        outdir / "local" / f"{args.prefix}cs_yodas_dropped_train_duration.tsv",
        drop_train,
    )
    write_dropped(
        outdir / "local" / f"{args.prefix}cs_yodas_dropped_valid_duration.tsv",
        drop_valid,
    )
    if args.link_data_roots:
        link_outputs(outdir, args.link_data_roots, names)
    LOGGER.info("done")


if __name__ == "__main__":
    main()
