#!/usr/bin/env python3
"""Create a manifest-only training view without changing source data or audio.

Run after Stage 3 and before statistics. ``raw`` requires the formatted audio's
utt2num_samples. Historical dur70 ``raw_copy`` instead uses int(float(utt2dur)
* sample_rate), including the original six-decimal serialization. It must NOT
substitute waveform lengths: the Persian recording
fa_ir/train/31_13412724805436051564_4df8c99c467bba1a.wav has 15345 actual samples,
but OLD raw_copy uses 16368 and retains it. OLD raw excludes it.

The mode is mandatory. There is no metadata/header fallback, resampling,
download, audio copy, source rewrite, or test-set filtering. Existing outputs
are never overwritten. --audit-only makes no filesystem changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path


UTT_FILES = (
    "wav.scp",
    "text",
    "utt2spk",
    "utt2lang",
    "utt2langs",
    "utt2category",
    "utt2dur",
    "utt2uniq",
)
GROUP_FILES = ("spk2utt", "lang2utt", "category2utt")


def read_kv(path: Path) -> dict[str, str]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            fields = line.rstrip("\n").split(maxsplit=1)
            if len(fields) != 2 or fields[0] in rows:
                raise ValueError(f"malformed or duplicate key: {path}:{lineno}")
            rows[fields[0]] = fields[1]
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_copy_samples(value: str, sample_rate: int) -> int:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"invalid historical duration: {value}")
    # Match the historical awk int($2 * fs), not Decimal or round().
    return int(seconds * sample_rate)


def prepare_view(
    input_dir: Path,
    source_data_dir: Path,
    *,
    mode: str,
    output_dir: Path | None = None,
    sample_rate: int = 16000,
    min_duration: float = 1.0,
    max_non_yodas_duration: float = 30.0,
    max_yodas_duration: float = 70.0,
    reuse_existing: bool = False,
) -> dict:
    if mode not in {"raw", "raw_copy"}:
        raise ValueError("mode must be explicitly raw or raw_copy")
    if sample_rate <= 0 or not 0 <= min_duration < min(
        max_non_yodas_duration, max_yodas_duration
    ):
        raise ValueError("invalid duration bounds or sample rate")
    if not math.isfinite(max_non_yodas_duration + max_yodas_duration):
        raise ValueError("duration bounds must be finite")
    if not any(
        source_data_dir.name.startswith(prefix)
        for prefix in ("train_", "valid_", "dur70_train_", "dur70_valid_")
    ):
        raise ValueError("only explicit train/valid views are supported; never test")
    if (input_dir / "segments").exists():
        raise ValueError("format segmented input before finalizing its duration view")
    if output_dir is not None:
        target = output_dir.resolve()
        for root in (input_dir.resolve(), source_data_dir.resolve()):
            if (
                target == root
                or target.is_relative_to(root)
                or root.is_relative_to(target)
            ):
                raise ValueError("output must be separate from both input trees")
        if output_dir.exists() and not reuse_existing:
            raise FileExistsError(
                f"refusing to overwrite existing output: {output_dir}"
            )

    wavs = read_kv(input_dir / "wav.scp")
    ids = set(wavs)
    if not ids:
        raise ValueError("empty input inventory")
    count_path = input_dir / ("utt2dur" if mode == "raw_copy" else "utt2num_samples")
    count_values = read_kv(count_path)
    if set(count_values) != ids:
        raise ValueError("sample/duration and wav.scp inventories differ")
    samples = {
        key: raw_copy_samples(value, sample_rate) if mode == "raw_copy" else int(value)
        for key, value in count_values.items()
    }
    if any(n < 0 for n in samples.values()):
        raise ValueError("negative sample count")
    if mode == "raw_copy" and (input_dir / "utt2num_samples").exists():
        previous = {
            k: int(v) for k, v in read_kv(input_dir / "utt2num_samples").items()
        }
        if previous != samples:
            raise ValueError(
                "raw_copy utt2num_samples disagrees with historical utt2dur"
            )

    category_path = source_data_dir / "utt2category"
    categories = read_kv(category_path)
    if not ids <= categories.keys():
        raise ValueError("source categories missing formatted utterances")
    minimum = int(min_duration * sample_rate)
    maxima = {
        "fleurs": int(max_non_yodas_duration * sample_rate),
        "cs_fleurs": int(max_non_yodas_duration * sample_rate),
        "cs_yodas": int(max_yodas_duration * sample_rate),
    }
    kept = set()
    dropped = {}
    for utt in sorted(ids):
        category = categories[utt]
        source = category.removesuffix("_pair")
        if source not in maxima:
            raise ValueError(f"unknown source category for {utt}: {category}")
        n = samples[utt]
        if minimum < n < maxima[source]:
            kept.add(utt)
        else:
            dropped[utt] = {
                "num_samples": n,
                "source": category,
                "reason": "too_short" if n <= minimum else "too_long",
            }
    if not kept:
        raise ValueError("duration policy removes every training utterance")

    paths = {}
    for name in (*UTT_FILES, *GROUP_FILES, "labels.jsonl"):
        path = input_dir / name
        if not path.exists():
            path = source_data_dir / name
        if path.exists():
            paths[name] = path
    for name in ("text", "utt2spk", "utt2lang", "utt2langs", "spk2utt", "lang2utt"):
        if name not in paths:
            raise FileNotFoundError(f"missing {name}")
    classes = []
    for name in UTT_FILES:
        if name in paths:
            rows = read_kv(paths[name])
            if not ids <= rows.keys():
                raise ValueError(f"missing input utterances in {paths[name]}")
            if name == "utt2lang":
                classes = sorted({rows[utt] for utt in kept})
    audit = {
        "mode": mode,
        "sample_rate": sample_rate,
        "sample_count_source": count_path.name,
        "sample_count_policy": (
            "OLD int(binary64(serialized utt2dur) * sample_rate); not audio length"
            if mode == "raw_copy"
            else "formatted audio utt2num_samples"
        ),
        "min_samples_exclusive": minimum,
        "max_samples_exclusive": maxima,
        "num_input": len(ids),
        "num_kept": len(kept),
        "num_dropped": len(dropped),
        "kept_id_sha256": hashlib.sha256(
            ("\n".join(sorted(kept)) + "\n").encode()
        ).hexdigest(),
        "classes": classes,
        "dropped": dropped,
        "input_sha256": {
            str(path): sha256_file(path)
            for path in sorted({count_path, category_path, *paths.values()})
        },
    }
    if output_dir is None:
        return audit

    if output_dir.exists():
        existing = json.loads((output_dir / "duration_view.json").read_text())
        output_hashes = existing.pop("output_sha256")
        if (
            existing != audit
            or not output_hashes
            or any(
                not (output_dir / name).is_file()
                or sha256_file(output_dir / name) != digest
                for name, digest in output_hashes.items()
            )
        ):
            raise ValueError(f"stale or modified duration view: {output_dir}")
        return audit

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".duration-view-", dir=output_dir.parent
    ) as tmp:
        dest = Path(tmp) / "view"
        dest.mkdir()
        for name in UTT_FILES:
            if name not in paths:
                continue
            with (
                paths[name].open(encoding="utf-8") as src,
                (dest / name).open("w", encoding="utf-8") as out,
            ):
                for line in src:
                    if line.split(maxsplit=1)[0] in kept:
                        out.write(line)
        for name in GROUP_FILES:
            if name not in paths:
                continue
            with (
                paths[name].open(encoding="utf-8") as src,
                (dest / name).open("w", encoding="utf-8") as out,
            ):
                for line in src:
                    group, *utts = line.split()
                    selected = [utt for utt in utts if utt in kept]
                    if selected:
                        out.write(f"{group} {' '.join(selected)}\n")
        if "labels.jsonl" in paths:
            written = set()
            with (
                paths["labels.jsonl"].open(encoding="utf-8") as src,
                (dest / "labels.jsonl").open("w", encoding="utf-8") as out,
            ):
                for line in src:
                    obj = json.loads(line)
                    utt = obj["uttid"]
                    if utt in kept:
                        if utt in written:
                            raise ValueError(f"duplicate metadata for {utt}")
                        written.add(utt)
                        if obj.get("wav") != wavs[utt]:
                            obj.setdefault("source_wav", obj.get("wav"))
                            obj["wav"] = wavs[utt]
                        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if written != kept:
                raise ValueError("metadata missing retained utterances")
        with (dest / "utt2num_samples").open("w", encoding="utf-8") as f:
            for utt in sorted(kept):
                f.write(f"{utt} {samples[utt]}\n")
        for name in ("feats_type", "audio_format"):
            if (input_dir / name).exists():
                shutil.copyfile(input_dir / name, dest / name)
        audit["output_sha256"] = {
            path.name: sha256_file(path) for path in sorted(dest.iterdir())
        }
        (dest / "duration_view.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        dest.rename(output_dir)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("raw", "raw_copy"), required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--require-no-drops", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()
    if args.audit_only == (args.output_dir is not None):
        parser.error("choose exactly one of --audit-only and --output-dir")
    audit = prepare_view(
        args.input_dir,
        args.source_data_dir or args.input_dir,
        mode=args.mode,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.require_no_drops and audit["num_dropped"]:
        raise SystemExit(
            "raw_copy audit failed: duration exclusions would change membership"
        )


if __name__ == "__main__":
    main()
