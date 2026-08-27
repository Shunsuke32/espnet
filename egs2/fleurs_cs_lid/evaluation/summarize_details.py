#!/usr/bin/env python3
"""Summarize exact accuracy by seen/unseen status and language set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


def canonical_set(tokens: Iterable[str]) -> Tuple[str, ...]:
    labels = {
        token.strip().lower().strip("<>[](){}'\"") for token in tokens if token.strip()
    }
    if not labels:
        raise ValueError("empty language set")
    return tuple(sorted(labels))


def read_label_sets(path: Path) -> Dict[str, Tuple[str, ...]]:
    values: Dict[str, Tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            utt = parts[0]
            if utt in values:
                raise ValueError(f"duplicate utterance id in {path}:{lineno}: {utt}")
            values[utt] = canonical_set(parts[1:])
    return values


def read_correctness(path: Path, column: str | None) -> Dict[str, int]:
    values: Dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"empty details file: {path}")
        utt_column = "utt" if "utt" in reader.fieldnames else "utt_id"
        if column is None:
            for candidate in ("exact", "set_exact", "correct"):
                if candidate in reader.fieldnames:
                    column = candidate
                    break
        if utt_column not in reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(
                f"details columns must include utt and correctness: {reader.fieldnames}"
            )
        for row in reader:
            utt = row[utt_column]
            if utt in values:
                raise ValueError(f"duplicate utterance id in details: {utt}")
            value = row[column].strip().lower()
            if value not in {"0", "1", "false", "true"}:
                raise ValueError(f"non-binary correctness for {utt}: {value!r}")
            values[utt] = int(value in {"1", "true"})
    return values


def add_stat(stats: dict, group_type: str, group: str, correct: int) -> None:
    item = stats[(group_type, group)]
    item["n"] += 1
    item["correct"] += correct


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--ref_utt2langs", type=Path, required=True)
    parser.add_argument("--train_utt2langs", type=Path, required=True)
    parser.add_argument("--correctness_column", default=None)
    parser.add_argument("--output_prefix", type=Path, required=True)
    args = parser.parse_args()

    train = read_label_sets(args.train_utt2langs)
    refs = read_label_sets(args.ref_utt2langs)
    correctness = read_correctness(args.details, args.correctness_column)
    if set(refs) != set(correctness):
        missing = sorted(set(refs) - set(correctness))
        extra = sorted(set(correctness) - set(refs))
        raise ValueError(
            f"reference/details mismatch: missing={missing[:20]}, extra={extra[:20]}"
        )

    seen_sets = set(train.values())
    stats = defaultdict(lambda: {"n": 0, "correct": 0})
    for utt, labels in refs.items():
        correct = correctness[utt]
        seen = labels in seen_sets
        add_stat(stats, "overall", "all", correct)
        add_stat(stats, "seen_status", "seen" if seen else "unseen", correct)
        add_stat(stats, "language_set", "-".join(labels), correct)

    rows = []
    for (group_type, group), values in sorted(stats.items()):
        n = values["n"]
        rows.append(
            {
                "group_type": group_type,
                "group": group,
                "n": n,
                "correct": values["correct"],
                "accuracy": values["correct"] / n,
            }
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_prefix.with_suffix(".tsv")
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["group_type", "group", "n", "correct", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_prefix.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "details": str(args.details),
                "ref_utt2langs": str(args.ref_utt2langs),
                "train_utt2langs": str(args.train_utt2langs),
                "seen_definition": "unordered canonical language set occurs in training utt2langs",
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"tsv": str(tsv_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
