#!/usr/bin/env python3
"""Plot post-sigmoid threshold sweeps from saved LID3 score JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class Series:
    key: str
    label: str
    color: str
    linestyle: str = "-"


DEFAULT_SERIES = [
    Series("test_fleurs_lid", "FLEURS", "#2c7fb8"),
    Series("test_cs_all", "CS-FLEURS all", "#4d4d4d"),
    Series("test_cs_read_test", "CS read", "#e66101"),
    Series("test_cs_xtts_test1", "CS XTTS1", "#2ca02c"),
    Series("test_cs_xtts_test2", "CS XTTS2", "#d62728", "--"),
    Series("test_cs_mms_test", "CS MMS", "#6f46a6"),
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def topk_reference(data: dict) -> Tuple[int, float]:
    rows = data.get("topk", [])
    if not rows:
        return 0, 0.0
    best = max(rows, key=lambda row: int(row.get("topk", 0)))
    return int(best["topk"]), 100.0 * float(best["accuracy"])


def curve(data: dict) -> List[Tuple[float, float]]:
    rows = data.get("sigmoid_threshold_sweep", [])
    return [(float(row["threshold"]), 100.0 * float(row["accuracy"])) for row in rows]


def best_threshold(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not points:
        return 0.0, 0.0
    return max(points, key=lambda item: item[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=Path, required=True)
    parser.add_argument("--output_prefix", type=Path, required=True)
    parser.add_argument("--include", nargs="*", default=None)
    args = parser.parse_args()

    include: Optional[set[str]] = set(args.include) if args.include else None
    series = [s for s in DEFAULT_SERIES if include is None or s.key in include]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    loaded: Dict[str, dict] = {}
    for s in series:
        path = args.score_dir / f"{s.key}.sigmoid_threshold_sweep.json"
        if path.exists():
            loaded[s.key] = load_json(path)

    if not loaded:
        raise SystemExit(f"no score JSON files found under {args.score_dir}")

    topk_rows: List[dict] = []
    best_rows: List[dict] = []
    curve_rows: List[dict] = []
    curves: Dict[str, List[Tuple[float, float]]] = {}
    topk_refs: Dict[str, Tuple[int, float]] = {}

    for s in series:
        if s.key not in loaded:
            continue
        data = loaded[s.key]
        points = curve(data)
        curves[s.key] = points
        topk, acc = topk_reference(data)
        topk_refs[s.key] = (topk, acc)
        best_thr, best_acc = best_threshold(points)
        n = int(data.get("num_ref", 0))
        topk_rows.append(
            {
                "set": s.key,
                "label": s.label,
                "n": n,
                "topk": topk,
                "accuracy_percent": f"{acc:.6f}",
            }
        )
        best_rows.append(
            {
                "set": s.key,
                "label": s.label,
                "n": n,
                "best_threshold": f"{best_thr:.6f}",
                "best_accuracy_percent": f"{best_acc:.6f}",
                "topk_reference_percent": f"{acc:.6f}",
            }
        )
        for thr, point_acc in points:
            curve_rows.append(
                {
                    "set": s.key,
                    "label": s.label,
                    "threshold": f"{thr:.6f}",
                    "accuracy_percent": f"{point_acc:.6f}",
                }
            )

    for suffix, rows, header in [
        (
            ".topk_reference.tsv",
            topk_rows,
            ["set", "label", "n", "topk", "accuracy_percent"],
        ),
        (
            ".best_threshold.tsv",
            best_rows,
            [
                "set",
                "label",
                "n",
                "best_threshold",
                "best_accuracy_percent",
                "topk_reference_percent",
            ],
        ),
        (
            ".threshold_curves.tsv",
            curve_rows,
            ["set", "label", "threshold", "accuracy_percent"],
        ),
    ]:
        with args.output_prefix.with_suffix(suffix).open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(15.6, 6.2))
    handles: List[Line2D] = []
    legend_labels: List[str] = []
    for s in series:
        if s.key not in loaded:
            continue
        xs = [x for x, _ in curves[s.key]]
        ys = [y for _, y in curves[s.key]]
        topk, acc = topk_refs[s.key]
        ax.plot(xs, ys, color=s.color, linestyle=s.linestyle, linewidth=2.5, alpha=0.96)
        ax.axhline(acc, color=s.color, linestyle=":", linewidth=2.0, alpha=0.55)
        handles.append(
            Line2D([0], [0], color=s.color, linestyle=s.linestyle, linewidth=2.5)
        )
        legend_labels.append(s.label)
        handles.append(
            Line2D([0], [0], color=s.color, linestyle=":", linewidth=2.0, alpha=0.7)
        )
        legend_labels.append(f"top{topk}={acc:.1f}%")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-2, 105)
    ax.set_xlabel("Sigmoid threshold", fontsize=18)
    ax.set_ylabel("Exact set accuracy (%)", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, which="major", alpha=0.25)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=False,
        fontsize=11.5,
        handlelength=2.4,
        columnspacing=0.8,
        handletextpad=0.45,
        labelspacing=0.25,
    )
    fig.subplots_adjust(top=0.80, left=0.075, right=0.995, bottom=0.16)
    png = args.output_prefix.with_suffix(".png")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(png)


if __name__ == "__main__":
    main()
