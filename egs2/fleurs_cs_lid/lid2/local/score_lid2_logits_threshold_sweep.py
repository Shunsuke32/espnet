#!/usr/bin/env python3
"""Score saved LID2 logits with top-k and threshold sweeps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def canon(label: str) -> str:
    return label.strip().lower().strip("<>[](){}'\"")


def read_refs(path: Path) -> Dict[str, Tuple[str, ...]]:
    refs: Dict[str, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            labels: List[str] = []
            for token in parts[1:]:
                label = canon(token)
                if not label:
                    continue
                if label in labels:
                    raise ValueError(
                        f"duplicate reference label for {parts[0]}: {label}"
                    )
                labels.append(label)
            utt = parts[0]
            if utt in refs:
                raise ValueError(f"duplicate utterance id in reference: {utt}")
            if not labels:
                raise ValueError(f"empty reference label set: {utt}")
            refs[utt] = tuple(labels)
    return refs


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(scores[idx])[::-1]]


def threshold_indices(scores: np.ndarray, threshold: float) -> np.ndarray:
    idx = np.flatnonzero(scores >= threshold)
    if idx.size == 0:
        return idx
    return idx[np.argsort(scores[idx])[::-1]]


def as_set(labels: List[str], idx: Iterable[int]) -> set[str]:
    return {labels[int(i)] for i in idx}


def pct(num: int, den: int) -> float:
    return num / den if den else 0.0


def score_thresholds(
    *,
    labels: List[str],
    utt_ids: List[str],
    scores: np.ndarray,
    refs: Dict[str, Tuple[str, ...]],
    thresholds: List[float],
) -> List[dict]:
    pred = {utt: scores[i] for i, utt in enumerate(utt_ids)}
    rows: List[dict] = []
    total = len(refs)
    for threshold in thresholds:
        exact = 0
        card_counts: Dict[int, int] = {}
        for utt, ref in refs.items():
            score = pred.get(utt)
            if score is None:
                card_counts[0] = card_counts.get(0, 0) + 1
                continue
            idx = threshold_indices(score, threshold)
            card_counts[int(idx.size)] = card_counts.get(int(idx.size), 0) + 1
            if as_set(labels, idx) == set(ref):
                exact += 1
        rows.append(
            {
                "threshold": threshold,
                "exact": exact,
                "accuracy": pct(exact, total),
                "prediction_cardinality": {
                    str(k): v for k, v in sorted(card_counts.items())
                },
            }
        )
    return rows


def score_topk(
    *,
    labels: List[str],
    utt_ids: List[str],
    probs: np.ndarray,
    refs: Dict[str, Tuple[str, ...]],
    topks: List[int],
) -> List[dict]:
    pred = {utt: probs[i] for i, utt in enumerate(utt_ids)}
    total = len(refs)
    rows: List[dict] = []
    for k in topks:
        exact = 0
        per_card: Dict[int, Dict[str, int]] = {}
        for utt, ref in refs.items():
            ref_card = len(ref)
            per_card.setdefault(ref_card, {"total": 0, "exact": 0})
            per_card[ref_card]["total"] += 1
            score = pred.get(utt)
            if score is None:
                continue
            idx = topk_indices(score, k)
            if as_set(labels, idx) == set(ref):
                exact += 1
                per_card[ref_card]["exact"] += 1
        rows.append(
            {
                "topk": k,
                "exact": exact,
                "accuracy": pct(exact, total),
                "per_ref_cardinality": {
                    str(card): {
                        "total": stats["total"],
                        "exact": stats["exact"],
                        "accuracy": pct(stats["exact"], stats["total"]),
                    }
                    for card, stats in sorted(per_card.items())
                },
            }
        )
    return rows


def write_cardinality_matched_details(
    path: Path,
    *,
    labels: List[str],
    utt_ids: List[str],
    probs: np.ndarray,
    refs: Dict[str, Tuple[str, ...]],
) -> None:
    """Write the paper metric: top-1 for FL and top-2 for CS utterances."""
    pred = {utt: probs[i] for i, utt in enumerate(utt_ids)}
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["utt", "ref", "pred", "reference_cardinality", "exact"])
        for utt, ref in sorted(refs.items()):
            idx = topk_indices(pred[utt], len(ref))
            hypothesis = tuple(labels[int(i)] for i in idx)
            writer.writerow(
                [
                    utt,
                    " ".join(ref),
                    " ".join(hypothesis),
                    len(ref),
                    int(set(hypothesis) == set(ref)),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits_npz", type=Path, required=True)
    parser.add_argument("--ref_utt2langs", type=Path, required=True)
    parser.add_argument("--output_prefix", type=Path, required=True)
    parser.add_argument("--topks", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--softmax_thresholds",
        type=float,
        nargs="+",
        default=[i / 100.0 for i in range(0, 101)],
    )
    parser.add_argument(
        "--allow_missing_predictions",
        action="store_true",
        help="score missing predictions as incorrect instead of failing",
    )
    parser.add_argument(
        "--logit_thresholds",
        type=float,
        nargs="+",
        default=[i / 2.0 for i in range(0, 61)],
    )
    args = parser.parse_args()

    data = np.load(args.logits_npz, allow_pickle=False)
    utt_ids = [str(x) for x in data["utt_ids"]]
    labels = [str(x) for x in data["labels"]]
    logits = data["logits"].astype(np.float64)
    if len(utt_ids) != len(set(utt_ids)):
        raise ValueError("duplicate utterance ids in logits archive")
    if logits.shape != (len(utt_ids), len(labels)):
        raise ValueError(
            f"logits shape {logits.shape} does not match "
            f"{len(utt_ids)} utterances x {len(labels)} labels"
        )
    probs = softmax(logits)
    refs = read_refs(args.ref_utt2langs)
    missing = sorted(set(refs) - set(utt_ids))
    extra = sorted(set(utt_ids) - set(refs))
    if missing and not args.allow_missing_predictions:
        raise ValueError(
            f"missing predictions for {len(missing)} references: {missing[:20]}"
        )
    if extra:
        raise ValueError(
            f"predictions contain {len(extra)} utterances absent from references: "
            f"{extra[:20]}"
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_prefix.with_suffix(".logits_and_softmax_probs.npz"),
        utt_ids=np.asarray(utt_ids),
        labels=np.asarray(labels),
        logits=logits.astype(np.float32),
        softmax_probs=probs.astype(np.float32),
    )
    np.savez_compressed(
        args.output_prefix.with_suffix(".softmax_probs.npz"),
        utt_ids=np.asarray(utt_ids),
        labels=np.asarray(labels),
        softmax_probs=probs.astype(np.float32),
    )

    details_path = args.output_prefix.with_suffix(
        ".cardinality_matched_topk.details.tsv"
    )
    if not missing:
        write_cardinality_matched_details(
            details_path,
            labels=labels,
            utt_ids=utt_ids,
            probs=probs,
            refs=refs,
        )

    topk = score_topk(
        labels=labels, utt_ids=utt_ids, probs=probs, refs=refs, topks=args.topks
    )
    softmax_sweep = score_thresholds(
        labels=labels,
        utt_ids=utt_ids,
        scores=probs,
        refs=refs,
        thresholds=args.softmax_thresholds,
    )
    logit_sweep = score_thresholds(
        labels=labels,
        utt_ids=utt_ids,
        scores=logits,
        refs=refs,
        thresholds=args.logit_thresholds,
    )
    result = {
        "logits_npz": str(args.logits_npz),
        "ref_utt2langs": str(args.ref_utt2langs),
        "num_ref": len(refs),
        "num_pred": len(utt_ids),
        "num_missing_pred": len(missing),
        "num_extra_pred": len(extra),
        "missing_pred_examples": missing[:20],
        "extra_pred_examples": extra[:20],
        "cardinality_matched_details_tsv": str(details_path) if not missing else None,
        "score_note": "logits are the saved model scores; if extracted with --apply_loss_scale they include AAMSoftmax scale",
        "topk": topk,
        "softmax_threshold_sweep": softmax_sweep,
        "logit_threshold_sweep": logit_sweep,
    }
    json_path = args.output_prefix.with_suffix(".threshold_sweep.json")
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    tsv_path = args.output_prefix.with_suffix(".threshold_sweep.tsv")
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["score_type", "threshold_or_topk", "exact", "n", "accuracy_percent"]
        )
        for row in topk:
            writer.writerow(
                [
                    "topk",
                    row["topk"],
                    row["exact"],
                    len(refs),
                    f"{100 * row['accuracy']:.6f}",
                ]
            )
        for row in softmax_sweep:
            writer.writerow(
                [
                    "softmax_threshold",
                    row["threshold"],
                    row["exact"],
                    len(refs),
                    f"{100 * row['accuracy']:.6f}",
                ]
            )
        for row in logit_sweep:
            writer.writerow(
                [
                    "logit_threshold",
                    row["threshold"],
                    row["exact"],
                    len(refs),
                    f"{100 * row['accuracy']:.6f}",
                ]
            )
    print(json.dumps({"json": str(json_path), "tsv": str(tsv_path)}, indent=2))


if __name__ == "__main__":
    main()
