#!/usr/bin/env python3
"""Paired significance tests for utterance-level exact-accuracy decisions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.stats import binomtest, ttest_rel


TRUE_VALUES = {"1", "true", "yes"}
FALSE_VALUES = {"0", "false", "no"}


def parse_bool(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return 1
    if normalized in FALSE_VALUES:
        return 0
    raise ValueError(f"expected a binary correctness value, got {value!r}")


def read_details(
    path: Path, column: str | None
) -> Tuple[Dict[str, int], Dict[str, str]]:
    scores: Dict[str, int] = {}
    refs: Dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"empty details file: {path}")
        utt_column = "utt" if "utt" in reader.fieldnames else "utt_id"
        if utt_column not in reader.fieldnames:
            raise ValueError(f"no utt/utt_id column in {path}: {reader.fieldnames}")
        if column is None:
            for candidate in ("exact", "set_exact", "correct"):
                if candidate in reader.fieldnames:
                    column = candidate
                    break
        if column is None or column not in reader.fieldnames:
            raise ValueError(
                f"no correctness column in {path}; pass --*_column from {reader.fieldnames}"
            )
        for row in reader:
            utt = row[utt_column]
            if utt in scores:
                raise ValueError(f"duplicate utterance id in {path}: {utt}")
            scores[utt] = parse_bool(row[column])
            if "ref" in row:
                refs[utt] = row["ref"]
    return scores, refs


def bootstrap_ci(
    differences: np.ndarray, samples: int, seed: int
) -> Tuple[float, float]:
    if samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, len(differences), size=len(differences))
        values[index] = differences[draw].mean()
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--system_a", type=Path, required=True)
    parser.add_argument("--system_b", type=Path, required=True)
    parser.add_argument("--system_a_name", default="system_a")
    parser.add_argument("--system_b_name", default="system_b")
    parser.add_argument("--system_a_column", default=None)
    parser.add_argument("--system_b_column", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3702)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    a_map, a_refs = read_details(args.system_a, args.system_a_column)
    b_map, b_refs = read_details(args.system_b, args.system_b_column)
    if set(a_map) != set(b_map):
        missing_a = sorted(set(b_map) - set(a_map))
        missing_b = sorted(set(a_map) - set(b_map))
        raise ValueError(
            "systems must cover identical utterances; "
            f"missing in A={missing_a[:20]}, missing in B={missing_b[:20]}"
        )
    for utt in set(a_refs) & set(b_refs):
        if a_refs[utt] != b_refs[utt]:
            raise ValueError(
                f"reference mismatch for {utt}: {a_refs[utt]!r} != {b_refs[utt]!r}"
            )

    utterances = sorted(a_map)
    if not utterances:
        raise ValueError("no paired utterances")
    a = np.asarray([a_map[utt] for utt in utterances], dtype=np.float64)
    b = np.asarray([b_map[utt] for utt in utterances], dtype=np.float64)
    differences = b - a
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    discordant = a_only + b_only

    if np.all(differences == differences[0]):
        t_statistic = 0.0 if differences[0] == 0 else float("inf")
        t_pvalue = 1.0 if differences[0] == 0 else 0.0
    else:
        t_result = ttest_rel(b, a)
        t_statistic = float(t_result.statistic)
        t_pvalue = float(t_result.pvalue)
    mcnemar_pvalue = (
        float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    ci_low, ci_high = bootstrap_ci(differences, args.bootstrap_samples, args.seed)

    result = {
        "num_paired_utterances": len(utterances),
        "system_a": args.system_a_name,
        "system_b": args.system_b_name,
        "system_a_accuracy": float(a.mean()),
        "system_b_accuracy": float(b.mean()),
        "accuracy_difference_b_minus_a": float(differences.mean()),
        "paired_t_test": {
            "statistic": t_statistic,
            "two_sided_pvalue": t_pvalue,
        },
        "exact_mcnemar": {
            "a_correct_b_wrong": a_only,
            "a_wrong_b_correct": b_only,
            "two_sided_pvalue": mcnemar_pvalue,
        },
        "paired_bootstrap_95_percent_ci": {
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "lower": ci_low,
            "upper": ci_high,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
