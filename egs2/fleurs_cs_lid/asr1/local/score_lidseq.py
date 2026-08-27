#!/usr/bin/env python3
"""Score variable-cardinality LID sequences.

References are usually data/<set>/utt2langs:
    uttid eng
    uttid hin eng

Hypotheses are ESPnet ASR text files:
    uttid <lid:hin> <lid:eng>

The script reports both order-sensitive sequence accuracy and unordered set
metrics, because code-switch detection often cares about the set of languages
present while an encoder-decoder may also learn an order.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

_LID_TOKEN_RE = re.compile(r"^<\s*(?:lid:)?([A-Za-z0-9_.-]+)\s*>$")

ALIAS = {
    "msa": "zlm",
    "zsm": "zlm",
    "may": "zlm",
    "fil": "tgl",
    "tl": "tgl",
    "swa": "swh",
    "ori": "ory",
    "zho": "cmn",
    "chi": "cmn",
    "cze": "ces",
    "dut": "nld",
    "fre": "fra",
    "ger": "deu",
    "gre": "ell",
    "per": "fas",
    "rum": "ron",
    "slo": "slk",
    "wel": "cym",
}


def read_label_map(path: Optional[Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if path is None or not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b, *_ = line.split()
            mapping[a.lower()] = b.lower()
    return mapping


def normalize_token(tok: str, mapping: Mapping[str, str]) -> Optional[str]:
    t = tok.strip().lower()
    if not t:
        return None
    # Drop common ASR artifacts.
    if t in {"<blank>", "<unk>", "<sos/eos>", "<s>", "</s>", "▁"}:
        return None
    m = _LID_TOKEN_RE.match(t)
    if m:
        t = m.group(1).lower()
    t = t.strip("<>[](){}'\".,;:")
    if t.startswith("lid:"):
        t = t[4:]
    candidates = []
    for cand in (t, t.replace("-", "_"), t.replace("_", "-")):
        if cand and cand not in candidates:
            candidates.append(cand)
    for cand in candidates:
        if cand in mapping:
            t = mapping[cand]
            break
    else:
        t = candidates[0] if candidates else t
    return ALIAS.get(t, t)


def parse_labels(
    parts: Sequence[str], mapping: Mapping[str, str], unique: bool = True
) -> Tuple[str, ...]:
    out: List[str] = []
    for p in parts:
        x = normalize_token(p, mapping)
        if x is None:
            continue
        if unique:
            if x not in out:
                out.append(x)
        else:
            out.append(x)
    return tuple(out)


def read_sequences(
    path: Path, mapping: Mapping[str, str], unique: bool = False
) -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            utt, toks = parts[0], parts[1:]
            if utt in out:
                raise ValueError(f"duplicate utterance id in {path}:{lineno}: {utt}")
            out[utt] = parse_labels(toks, mapping, unique=unique)
    return out


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def f1(p: float, r: float) -> float:
    return safe_div(2 * p * r, p + r)


def score(
    ref: Mapping[str, Tuple[str, ...]], hyp: Mapping[str, Tuple[str, ...]]
) -> Dict[str, object]:
    labels = sorted(
        {x for seq in ref.values() for x in seq}
        | {x for seq in hyp.values() for x in seq}
    )
    n = len(ref)
    missing = 0
    extra_hyp = sorted(set(hyp) - set(ref))
    seq_exact = 0
    set_exact = 0
    len_exact = 0
    tp = Counter()
    fp = Counter()
    fn = Counter()
    details = []
    by_ref_len = defaultdict(lambda: Counter(n=0, set_exact=0, seq_exact=0))

    for utt, rseq in ref.items():
        hseq = hyp.get(utt, tuple())
        if utt not in hyp:
            missing += 1
        rset, hset = set(rseq), set(hseq)
        seq_ok = int(hseq == rseq)
        set_ok = int(hset == rset)
        len_ok = int(len(hseq) == len(rseq))
        seq_exact += seq_ok
        set_exact += set_ok
        len_exact += len_ok
        key = len(rset)
        by_ref_len[key]["n"] += 1
        by_ref_len[key]["set_exact"] += set_ok
        by_ref_len[key]["seq_exact"] += seq_ok
        for lab in labels:
            if lab in rset and lab in hset:
                tp[lab] += 1
            elif lab not in rset and lab in hset:
                fp[lab] += 1
            elif lab in rset and lab not in hset:
                fn[lab] += 1
        details.append((utt, " ".join(rseq), " ".join(hseq), seq_ok, set_ok, len_ok))

    micro_tp = sum(tp.values())
    micro_fp = sum(fp.values())
    micro_fn = sum(fn.values())
    micro_p = safe_div(micro_tp, micro_tp + micro_fp)
    micro_r = safe_div(micro_tp, micro_tp + micro_fn)
    macro_ps = []
    macro_rs = []
    macro_f1s = []
    per_label = {}
    for lab in labels:
        p = safe_div(tp[lab], tp[lab] + fp[lab])
        r = safe_div(tp[lab], tp[lab] + fn[lab])
        lab_f1 = f1(p, r)
        macro_ps.append(p)
        macro_rs.append(r)
        macro_f1s.append(lab_f1)
        per_label[lab] = {
            "precision": p,
            "recall": r,
            "f1": lab_f1,
            "tp": tp[lab],
            "fp": fp[lab],
            "fn": fn[lab],
        }

    out: Dict[str, object] = {
        "num_ref": n,
        "num_hyp": len(hyp),
        "num_missing_hyp": missing,
        "num_extra_hyp": len(extra_hyp),
        "extra_hyp_examples": extra_hyp[:20],
        "sequence_exact_accuracy": safe_div(seq_exact, n),
        "unordered_set_exact_accuracy": safe_div(set_exact, n),
        "unordered_set_accuracy": safe_div(set_exact, n),
        "length_accuracy": safe_div(len_exact, n),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": f1(micro_p, micro_r),
        "macro_precision": safe_div(sum(macro_ps), len(macro_ps)),
        "macro_recall": safe_div(sum(macro_rs), len(macro_rs)),
        "macro_f1": safe_div(sum(macro_f1s), len(macro_f1s)),
        "by_ref_label_count": {
            str(k): {
                "n": c["n"],
                "set_exact_accuracy": safe_div(c["set_exact"], c["n"]),
                "sequence_exact_accuracy": safe_div(c["seq_exact"], c["n"]),
            }
            for k, c in sorted(by_ref_len.items())
        },
        "per_label": per_label,
    }
    out["_details"] = details
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ref", type=Path, required=True, help="utt2langs or text reference"
    )
    parser.add_argument("--hyp", type=Path, required=True, help="ESPnet decoded text")
    parser.add_argument("--label_map", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--details_out", type=Path, default=None)
    parser.add_argument(
        "--allow_missing_hyp",
        action="store_true",
        help="do not fail when references lack hypotheses",
    )
    parser.add_argument(
        "--allow_extra_hyp",
        action="store_true",
        help="do not fail when hypotheses contain IDs absent from references",
    )
    args = parser.parse_args()

    mapping = read_label_map(args.label_map)
    ref = read_sequences(args.ref, mapping, unique=False)
    hyp = read_sequences(args.hyp, mapping, unique=False)
    for utt, labels in ref.items():
        if len(labels) != len(set(labels)):
            raise ValueError(f"duplicate reference labels for {utt}: {labels}")
    metrics = score(ref, hyp)
    details = metrics.pop("_details")

    if args.details_out:
        args.details_out.parent.mkdir(parents=True, exist_ok=True)
        with args.details_out.open("w", encoding="utf-8") as f:
            f.write("utt\tref\thyp\tseq_exact\tset_exact\tlen_exact\n")
            for row in details:
                f.write("\t".join(map(str, row)) + "\n")

    text = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not args.allow_missing_hyp and metrics["num_missing_hyp"] != 0:
        raise SystemExit(
            f"missing hypotheses: {metrics['num_missing_hyp']} / {metrics['num_ref']}"
        )
    if not args.allow_extra_hyp and metrics["num_extra_hyp"] != 0:
        raise SystemExit(
            f"extra hypotheses: {metrics['num_extra_hyp']} / {metrics['num_hyp']}"
        )


if __name__ == "__main__":
    main()
