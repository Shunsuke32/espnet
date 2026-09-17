"""Model-independent scoring of frozen LID logits (no fitted transforms)."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


LANGUAGE = re.compile(r"^[a-z]{3}$")
SOFTMAX_THRESHOLDS = [i / 100 for i in range(101)]
SIGMOID_THRESHOLDS = [0.0, 0.001, 0.002, 0.005] + [i / 100 for i in range(1, 101)]
LOGIT_THRESHOLDS = [i / 2 for i in range(61)]


def language_set(tokens):
    labels = tuple(token.strip().lower().strip("<>") for token in tokens)
    if not labels or len(labels) > 2 or any(not LANGUAGE.fullmatch(x) for x in labels):
        raise ValueError(f"expected one or two canonical language labels: {tokens}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate reference labels: {tokens}")
    return tuple(sorted(labels))


def class_languages(label):
    return language_set(label.split("-"))


def read_manifest(path, excluded=()):
    """Keep IDs explicit; never let a dictionary silently discard duplicates."""
    rows = {}
    encountered = set()
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=1)
            utt = parts[0]
            encountered.add(utt)
            if utt in excluded:
                continue
            if len(parts) != 2 or utt in rows:
                raise ValueError(
                    f"empty value or duplicate ID in {path}:{lineno}: {utt}"
                )
            rows[utt] = parts[1]
    return rows, encountered


def read_refs(path, excluded=()):
    rows, encountered = read_manifest(path, excluded)
    return {
        utt: language_set(value.split()) for utt, value in rows.items()
    }, encountered


def validate_labels(labels):
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("empty or duplicate class inventory")
    expanded = [class_languages(label) for label in labels]
    if len(set(expanded)) != len(expanded):
        raise ValueError("class inventory contains equivalent atomic pairs")
    return expanded


def validate_scores(utt_ids, labels, logits):
    validate_labels(labels)
    if not utt_ids or len(set(utt_ids)) != len(utt_ids):
        raise ValueError("empty or duplicate utterance IDs in scores")
    if logits.shape != (len(utt_ids), len(labels)) or not np.isfinite(logits).all():
        raise ValueError("score shape mismatch or non-finite logits")


def probabilities(logits, activation):
    logits = np.asarray(logits, dtype=np.float64)
    if activation == "softmax":
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
    if activation != "sigmoid":
        raise ValueError(f"unknown activation: {activation}")
    out = np.empty_like(logits)
    positive = logits >= 0
    out[positive] = 1 / (1 + np.exp(-logits[positive]))
    exp = np.exp(logits[~positive])
    out[~positive] = exp / (1 + exp)
    return out


def check_alignment(expected, actual, context):
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"{context}: missing={missing[:10]}, extra={extra[:10]}")


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_tsv(path, columns, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize(correct, refs, seen_pairs):
    groups = defaultdict(list)
    for utt, ref in refs.items():
        groups[("overall", "all")].append(correct[utt])
        groups[("reference_cardinality", str(len(ref)))].append(correct[utt])
        groups[("language_set", "-".join(ref))].append(correct[utt])
        if len(ref) == 2:
            status = "seen" if ref in seen_pairs else "unseen"
            groups[("pair_seen_status", status)].append(correct[utt])
    return [
        dict(
            group_type=kind,
            group=name,
            n=len(values),
            correct=sum(values),
            accuracy=sum(values) / len(values),
        )
        for (kind, name), values in sorted(groups.items())
    ]


def evaluate_set(
    output_dir,
    name,
    labels,
    utt_ids,
    logits,
    probs,
    refs,
    seen_pairs,
    head,
    thresholds=None,
    logit_thresholds=None,
):
    """Top-1 FL / top-2 CS, or expanded atomic top-1, on exactly the given IDs."""
    check_alignment(refs, utt_ids, name)
    validate_scores(utt_ids, labels, logits)
    if probs.shape != logits.shape or not np.isfinite(probs).all():
        raise ValueError("probability shape mismatch or non-finite probabilities")
    expanded = validate_labels(labels)
    atomic = head["prediction_mode"] == "atomic_top1"
    if not atomic and any(len(x) != 1 for x in expanded):
        raise ValueError("individual-language head contains pair classes")
    index = {utt: i for i, utt in enumerate(utt_ids)}
    details = []
    correct = {}
    for utt, ref in sorted(refs.items()):
        k = 1 if atomic else len(ref)
        if k > len(labels):
            raise ValueError(f"cannot select top{k} from {len(labels)} classes")
        # Rank logits, not rounded/saturated probabilities. Ties use class order.
        ranks = np.argsort(-logits[index[utt]], kind="stable")[:k]
        predicted = (
            expanded[int(ranks[0])] if atomic else tuple(labels[i] for i in ranks)
        )
        exact = int(set(predicted) == set(ref))
        correct[utt] = exact
        details.append(
            dict(
                utt=utt,
                ref=" ".join(ref),
                pred=" ".join(predicted),
                reference_cardinality=len(ref),
                exact=exact,
            )
        )
    summary = summarize(correct, refs, seen_pairs)
    result = dict(
        set=name,
        num_ref=len(refs),
        num_pred=len(utt_ids),
        num_missing_pred=0,
        head=head,
        exact=sum(correct.values()),
        accuracy=sum(correct.values()) / len(refs),
        metric="atomic_top1_expanded" if atomic else "top1_fleurs_top2_cs",
        seen_definition="unordered pair occurs in frozen training utt2langs",
        groups=summary,
    )
    prefix = Path(output_dir) / name
    write_tsv(str(prefix) + ".details.tsv", list(details[0]), details)
    write_tsv(str(prefix) + ".summary.tsv", list(summary[0]), summary)

    sweep_rows = []
    if head["sweep"]:
        if thresholds is None:
            thresholds = (
                SIGMOID_THRESHOLDS
                if head["activation"] == "sigmoid"
                else SOFTMAX_THRESHOLDS
            )
        thresholds = sorted(set(thresholds))
        if any(not np.isfinite(x) or not 0 <= x <= 1 for x in thresholds):
            raise ValueError("probability thresholds must be finite and in [0, 1]")
        domains = [(head["activation"], probs, thresholds)]
        if head["activation"] == "softmax":
            grid = (
                LOGIT_THRESHOLDS
                if logit_thresholds is None
                else sorted(set(logit_thresholds))
            )
            if any(not np.isfinite(x) for x in grid):
                raise ValueError("logit thresholds must be finite")
            domains.append(("logit", logits, grid))
        for domain, scores, grid in domains:
            for threshold in grid:
                decisions = {}
                cardinalities = defaultdict(int)
                for utt, ref in refs.items():
                    selected = np.flatnonzero(scores[index[utt]] >= threshold)
                    cardinalities[len(selected)] += 1
                    decisions[utt] = int({labels[i] for i in selected} == set(ref))
                for group in summarize(decisions, refs, seen_pairs):
                    sweep_rows.append(
                        dict(score_type=domain, threshold=threshold, **group)
                    )
                result.setdefault("threshold_prediction_cardinality", {}).setdefault(
                    domain, {}
                )[str(threshold)] = dict(cardinalities)
        write_tsv(str(prefix) + ".threshold_sweep.tsv", list(sweep_rows[0]), sweep_rows)
        result["threshold_sweep"] = sweep_rows
    write_json(str(prefix) + ".json", result)
    return result


def plot_results(output_dir, results):
    if not any(result.get("threshold_sweep") for result in results):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for domain in ("softmax", "sigmoid", "logit"):
        fig, ax = plt.subplots(figsize=(10, 5))
        count = 0
        for result in results:
            rows = [
                row
                for row in result.get("threshold_sweep", [])
                if row["score_type"] == domain and row["group_type"] == "overall"
            ]
            if not rows:
                continue
            (line,) = ax.plot(
                [r["threshold"] for r in rows],
                [100 * r["accuracy"] for r in rows],
                label=result["set"],
            )
            ax.axhline(
                100 * result["accuracy"],
                color=line.get_color(),
                linestyle=":",
                label=f"{result['set']} {result['metric']}",
            )
            count += 1
        if count:
            ax.set(
                xlabel=f"{domain.capitalize()} threshold",
                ylabel="Exact set accuracy (%)",
                ylim=(-1, 101),
            )
            ax.grid(alpha=0.25)
            ax.legend(fontsize="small")
            fig.tight_layout()
            fig.savefig(Path(output_dir) / f"{domain}_threshold_sweep.png", dpi=160)
        plt.close(fig)
