#!/usr/bin/env python3
"""Validate an auto-generated fixed-batch budget against current data/options."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v!r}")


def count_utts(train_dir: Path, count_file: str) -> int:
    candidates = [
        train_dir / count_file,
        train_dir / "wav.scp",
        train_dir / "utt2spk",
        train_dir / "text",
        train_dir / "utt2lang",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        seen = set()
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                utt = line.split(maxsplit=1)[0]
                if utt in seen:
                    raise ValueError(f"duplicate utterance id in {path}: {utt}")
                seen.add(utt)
        if seen:
            return len(seen)
    raise FileNotFoundError(f"No countable file found under {train_dir}")


def same_float(a: Any, b: float) -> bool:
    try:
        return abs(float(a) - float(b)) < 1.0e-9
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--budget_json", required=True, type=Path)
    parser.add_argument("--train_dir", required=True, type=Path)
    parser.add_argument("--count_file", default="wav.scp")
    parser.add_argument("--task", choices=["asr", "lid"], required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--accum_grad", type=int, required=True)
    parser.add_argument("--effective_batch_size", type=int, required=True)
    parser.add_argument("--ngpu", type=int, required=True)
    parser.add_argument("--max_epoch", type=int, required=True)
    parser.add_argument("--target_passes", type=float, required=True)
    parser.add_argument("--warmup_ratio", type=float, required=True)
    parser.add_argument("--round_updates_per_epoch_to", type=int, required=True)
    parser.add_argument("--batch_type", required=True)
    parser.add_argument("--drop_last_iter", type=str2bool, required=True)
    args = parser.parse_args()

    with args.budget_json.open("r", encoding="utf-8") as f:
        budget = json.load(f)

    expected: dict[str, Any] = {
        "task": args.task,
        "num_train_utts": count_utts(args.train_dir, args.count_file),
        "batch_size": args.batch_size,
        "accum_grad": args.accum_grad,
        "effective_batch_size": args.effective_batch_size,
        "ngpu": args.ngpu,
        "max_epoch": args.max_epoch,
        "round_updates_per_epoch_to": args.round_updates_per_epoch_to,
        "batch_type": args.batch_type,
        "drop_last_iter": bool(args.drop_last_iter),
    }
    errors = []
    for key, value in expected.items():
        if budget.get(key) != value:
            errors.append(f"{key}: budget={budget.get(key)!r} current={value!r}")
    for key, value in {
        "target_passes": args.target_passes,
        "warmup_ratio": args.warmup_ratio,
    }.items():
        if not same_float(budget.get(key), value):
            errors.append(f"{key}: budget={budget.get(key)!r} current={value!r}")

    if errors:
        joined = "\n  ".join(errors)
        raise SystemExit(
            f"Generated budget does not match current run: {args.budget_json}\n  {joined}"
        )
    print(f"validate_fixed_batch_budget: OK {args.budget_json}")


if __name__ == "__main__":
    main()
