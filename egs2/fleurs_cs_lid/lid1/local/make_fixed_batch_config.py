#!/usr/bin/env python3
"""Generate a fixed-count, data-dependent ESPnet training config.

Policy used in this recipe:
  * Use utterance-count mini-batches, not batch_bins/numel/length batching.
  * ESPnet2 batch_size is global and is not multiplied by ngpu.
  * effective_batch_size = batch_size * accum_grad.
  * max_epoch is explicit (the paper uses 30, except LID2 uses 15).
  * num_iters_per_epoch is computed from the actual prepared train-set size so
    that total exposure is about target_passes over that train set.
  * warmup is computed as a fixed ratio of total optimizer updates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise SystemExit(f"PyYAML is required: {e}")


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v!r}")


def ceil_to_multiple(x: float, m: int) -> int:
    if m <= 1:
        return max(1, int(math.ceil(x)))
    return max(m, int(math.ceil(x / m)) * m)


def count_utts(train_dir: Path, count_file: str) -> int:
    candidates = [
        train_dir / count_file,
        train_dir / "wav.scp",
        train_dir / "utt2spk",
        train_dir / "text",
        train_dir / "utt2lang",
    ]
    for p in candidates:
        if p.is_file():
            seen = set()
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    utt = line.split(maxsplit=1)[0]
                    if utt in seen:
                        raise ValueError(f"duplicate utterance id in {p}: {utt}")
                    seen.add(utt)
            if seen:
                return len(seen)
    raise FileNotFoundError(f"No countable file found under {train_dir}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return obj


def compute_budget(
    *,
    num_utts: int,
    batch_size: int,
    accum_grad: int,
    effective_batch_size: int,
    ngpu: int,
    max_epoch: int,
    target_passes: float,
    warmup_ratio: float,
    round_updates_per_epoch_to: int,
) -> dict[str, Any]:
    if num_utts <= 0:
        raise ValueError("num_utts must be positive")
    if batch_size <= 0 or accum_grad <= 0 or effective_batch_size <= 0:
        raise ValueError(
            "batch_size, accum_grad, effective_batch_size must be positive"
        )
    if batch_size * accum_grad != effective_batch_size:
        raise ValueError(
            f"batch_size * accum_grad must equal effective_batch_size: {batch_size} * {accum_grad} != {effective_batch_size}"
        )
    if ngpu > 1 and batch_size % ngpu != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by ngpu={ngpu} for equal per-GPU utterance counts"
        )
    if max_epoch <= 0:
        raise ValueError("max_epoch must be positive")
    if target_passes <= 0:
        raise ValueError("target_passes must be positive")
    if not (0.0 <= warmup_ratio < 1.0):
        raise ValueError("warmup_ratio must be in [0, 1)")

    exact_total_updates = target_passes * num_utts / effective_batch_size
    exact_updates_per_epoch = exact_total_updates / max_epoch
    updates_per_epoch = ceil_to_multiple(
        exact_updates_per_epoch, round_updates_per_epoch_to
    )
    total_updates = updates_per_epoch * max_epoch
    num_iters_per_epoch = updates_per_epoch * accum_grad
    total_seen_examples = total_updates * effective_batch_size
    actual_passes = total_seen_examples / num_utts
    full_examples_per_pass = (num_utts // batch_size) * batch_size
    warmup_steps = int(round(total_updates * warmup_ratio))
    if warmup_ratio > 0 and warmup_steps < 1:
        warmup_steps = 1
    return {
        "num_train_utts": num_utts,
        "batch_size": batch_size,
        "accum_grad": accum_grad,
        "effective_batch_size": effective_batch_size,
        "ngpu": ngpu,
        "per_gpu_microbatch": batch_size // max(1, ngpu),
        "full_microbatches_per_pass": num_utts // batch_size,
        "full_examples_per_pass": full_examples_per_pass,
        "remainder_examples_per_pass": num_utts % batch_size,
        "max_epoch": max_epoch,
        "target_passes": target_passes,
        "exact_total_updates": exact_total_updates,
        "exact_updates_per_epoch": exact_updates_per_epoch,
        "round_updates_per_epoch_to": round_updates_per_epoch_to,
        "updates_per_epoch": updates_per_epoch,
        "num_iters_per_epoch": num_iters_per_epoch,
        "total_optimizer_updates": total_updates,
        "total_seen_examples": total_seen_examples,
        "actual_passes": actual_passes,
        "actual_passes_over_full_examples": (
            total_seen_examples / full_examples_per_pass
            if full_examples_per_pass > 0
            else None
        ),
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
    }


def patch_config(
    cfg: dict[str, Any],
    budget: dict[str, Any],
    *,
    task: str,
    batch_type: str,
    drop_last_iter: bool,
) -> dict[str, Any]:
    cfg = dict(cfg)
    # ESPnet train entrypoints reject unknown top-level config keys.
    cfg.pop("comment", None)
    for key in [
        "batch_bins",
        "valid_batch_bins",
        "max_batch_size",
        "min_batch_size",
        "valid_batch_type",
        "upsampling_factor",
        "dataset_scaling_factor",
        "category_upsampling_factor",
        "dataset_upsampling_factor",
        "fold_length",
    ]:
        cfg.pop(key, None)
    cfg["max_epoch"] = int(budget["max_epoch"])
    cfg["num_iters_per_epoch"] = int(budget["num_iters_per_epoch"])
    cfg["batch_type"] = batch_type
    cfg["batch_size"] = int(budget["batch_size"])
    cfg["valid_batch_size"] = int(budget["batch_size"])
    cfg["accum_grad"] = int(budget["accum_grad"])
    cfg["drop_last_iter"] = bool(drop_last_iter)
    if task == "lid" and batch_type == "catbel":
        cfg["iterator_type"] = "category"
        cfg["valid_iterator_type"] = "category"
    else:
        cfg.pop("iterator_type", None)
        cfg.pop("valid_iterator_type", None)
    if batch_type == "sorted":
        cfg["sort_in_batch"] = "descending"
        cfg["sort_batch"] = "descending"
    sched = str(cfg.get("scheduler", "")).lower()
    sch = dict(cfg.get("scheduler_conf") or {})
    if sched == "warmuplr":
        sch["warmup_steps"] = int(budget["warmup_steps"])
    elif sched == "tristagelr":
        sch["max_steps"] = int(budget["total_optimizer_updates"])
        sch["warmup_ratio"] = float(budget["warmup_ratio"])
        hold = float(sch.get("hold_ratio", 0.2))
        if hold + float(budget["warmup_ratio"]) >= 1.0:
            hold = max(0.0, 1.0 - float(budget["warmup_ratio"]))
        sch["hold_ratio"] = hold
        sch["decay_ratio"] = max(0.0, 1.0 - float(budget["warmup_ratio"]) - hold)
    cfg["scheduler_conf"] = sch
    return cfg


def write_tsv(path: Path, budget: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("key\tvalue\n")
        for k, v in budget.items():
            f.write(f"{k}\t{v}\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task", choices=["asr", "lid"], required=True)
    p.add_argument("--base_config", "--template", required=True, type=Path)
    p.add_argument(
        "--output_config", "--out_config", "--output", required=True, type=Path
    )
    p.add_argument("--train_dir", "--train_data_dir", required=True, type=Path)
    p.add_argument("--count_file", default="wav.scp")
    p.add_argument("--batch_size", type=int, choices=[2, 4, 8, 16, 32], required=True)
    p.add_argument("--accum_grad", type=int, choices=[1, 2, 4, 8, 16], required=True)
    p.add_argument("--effective_batch_size", type=int, default=32)
    p.add_argument("--ngpu", "--num_gpus", type=int, default=2)
    p.add_argument("--max_epoch", type=int, default=30)
    p.add_argument("--target_passes", "--num_passes", type=float, default=3.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--round_updates_per_epoch_to", type=int, default=10)
    p.add_argument(
        "--batch_type", choices=["sorted", "unsorted", "catbel"], default=None
    )
    p.add_argument("--drop_last_iter", type=str2bool, default=True)
    p.add_argument("--budget_json", "--summary", type=Path, default=None)
    p.add_argument("--budget_tsv", "--report", type=Path, default=None)
    args = p.parse_args(argv)

    batch_type = args.batch_type or ("sorted" if args.task == "asr" else "catbel")
    if args.task == "asr" and batch_type == "catbel":
        raise ValueError("catbel is for LID/classification, not ASR sequence training")
    n = count_utts(args.train_dir, args.count_file)
    budget = compute_budget(
        num_utts=n,
        batch_size=args.batch_size,
        accum_grad=args.accum_grad,
        effective_batch_size=args.effective_batch_size,
        ngpu=args.ngpu,
        max_epoch=args.max_epoch,
        target_passes=args.target_passes,
        warmup_ratio=args.warmup_ratio,
        round_updates_per_epoch_to=args.round_updates_per_epoch_to,
    )
    budget.update(
        {
            "task": args.task,
            "base_config": str(args.base_config),
            "output_config": str(args.output_config),
            "train_dir": str(args.train_dir),
            "count_file": args.count_file,
            "batch_type": batch_type,
            "drop_last_iter": bool(args.drop_last_iter),
            "drop_last_effective_train_examples_per_pass": (
                budget["full_examples_per_pass"]
                if args.drop_last_iter
                else budget["num_train_utts"]
            ),
            "drop_last_remainder_examples_per_pass": (
                budget["remainder_examples_per_pass"] if args.drop_last_iter else 0
            ),
        }
    )
    cfg = patch_config(
        load_yaml(args.base_config),
        budget,
        task=args.task,
        batch_type=batch_type,
        drop_last_iter=args.drop_last_iter,
    )
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as f:
        f.write(
            "# Auto-generated by local/make_fixed_batch_config.py. Do not edit by hand.\n"
        )
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    if args.budget_json:
        args.budget_json.parent.mkdir(parents=True, exist_ok=True)
        args.budget_json.write_text(
            json.dumps(budget, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if args.budget_tsv:
        write_tsv(args.budget_tsv, budget)
    print(json.dumps(budget, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
