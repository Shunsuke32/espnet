#!/usr/bin/env python3
"""Top-k/threshold evaluation with saved logits for LID2 softmax models.

LID2 uses the MMS+ECAPA LID architecture with a soft-target AAMSoftmax-style
loss.  This script extracts the inference-time class logits before softmax,
saves them to ``.npz``, and scores top-1/top-2 and threshold-based label sets.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


LOGGER = logging.getLogger("lid_softmax_logits")


def canon(label: str) -> str:
    return label.strip().lower().strip("<>[](){}'\"")


def read_lang2utt(path: Path) -> List[str]:
    labels: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if parts:
                labels.append(canon(parts[0]))
    if not labels:
        raise ValueError(f"empty lang2utt: {path}")
    if len(labels) != len(set(labels)):
        seen = set()
        dup = []
        for lab in labels:
            if lab in seen and lab not in dup:
                dup.append(lab)
            seen.add(lab)
        raise ValueError(f"duplicate labels in lang2utt: {dup[:20]}")
    return labels


def read_refs(path: Path) -> Dict[str, Tuple[str, ...]]:
    refs: Dict[str, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            labs: List[str] = []
            for tok in parts[1:]:
                lab = canon(tok)
                if not lab:
                    continue
                if lab in labs:
                    raise ValueError(
                        f"duplicate reference label for {parts[0]}: {lab}"
                    )
                labs.append(lab)
            utt = parts[0]
            if utt in refs:
                raise ValueError(f"duplicate utterance id in reference: {utt}")
            if not labs:
                raise ValueError(f"empty reference label set: {utt}")
            refs[utt] = tuple(labs)
    return refs


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def build_iterator(args: argparse.Namespace, lid_train_args: argparse.Namespace):
    from espnet2.tasks.lid import LIDTask

    merged = vars(lid_train_args).copy()
    merged.update(
        {
            "valid_data_path_and_name_and_type": [
                (str(args.wav_scp), "speech", "sound")
            ],
            "valid_shape_file": [],
            "dtype": args.dtype,
            "valid_batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "use_preprocessor": False,
            "preprocessor_conf": {
                "fix_duration": False,
                "target_duration": None,
                "noise_apply_prob": 0.0,
                "rir_apply_prob": 0.0,
            },
        }
    )
    merged_args = argparse.Namespace(**merged)
    return LIDTask.build_streaming_iterator(
        merged_args.valid_data_path_and_name_and_type,
        dtype=merged_args.dtype,
        batch_size=merged_args.valid_batch_size,
        num_workers=merged_args.num_workers,
        preprocess_fn=LIDTask.build_preprocess_fn(merged_args, train=False),
        collate_fn=LIDTask.build_collate_fn(merged_args, False),
        inference=True,
    )


def _find_loss_weight(loss):
    for attr in ("weight", "W"):
        weight = getattr(loss, attr, None)
        if weight is not None and getattr(weight, "ndim", None) == 2:
            return weight
    for attr in ("fc", "linear", "classifier"):
        module = getattr(loss, attr, None)
        weight = getattr(module, "weight", None)
        if weight is not None and getattr(weight, "ndim", None) == 2:
            return weight
    raise RuntimeError(f"unsupported LID loss for logit extraction: {type(loss)}")


def logits_from_model(
    model, speech, speech_lengths, num_labels: int, apply_loss_scale: bool
):
    import torch.nn.functional as F

    feats, feat_lengths = model.extract_feats(speech, speech_lengths)
    frame = model.encode_frame(feats)
    utt_level = model.pooling(frame, feat_lengths=feat_lengths)
    emb = model.project_lang_embd(utt_level)
    weight = _find_loss_weight(model.loss)
    logits = F.linear(F.normalize(emb), F.normalize(weight))

    if logits.shape[1] != num_labels:
        if logits.shape[1] % num_labels != 0:
            raise RuntimeError(
                f"logit dim {logits.shape[1]} is incompatible with {num_labels} labels"
            )
        k = logits.shape[1] // num_labels
        logits = logits.reshape(logits.shape[0], num_labels, k).max(dim=2).values

    if apply_loss_scale:
        scale = getattr(model.loss, "scale", getattr(model.loss, "s", None))
        if scale is not None:
            logits = logits * float(scale)
    return logits


def infer(args: argparse.Namespace):
    import torch
    from espnet2.tasks.lid import LIDTask
    from espnet2.torch_utils.device_funcs import to_device

    device = "cuda" if args.ngpu > 0 else "cpu"
    model, train_args = LIDTask.build_model_from_file(
        str(args.lid_train_config), str(args.lid_model_file), device
    )
    model.eval()

    labels = read_lang2utt(args.lang2utt)
    if args.topk > len(labels):
        raise ValueError(f"--topk {args.topk} exceeds number of labels {len(labels)}")
    nclasses = getattr(
        model.loss, "out_features", getattr(model.loss, "nclasses", None)
    )
    if nclasses is not None and int(nclasses) != len(labels):
        raise ValueError(f"model has {nclasses} classes but lang2utt has {len(labels)}")

    iterator = build_iterator(args, train_args)
    probs_by_utt: Dict[str, np.ndarray] = {}
    logits_chunks: List[np.ndarray] = []
    utt_order: List[str] = []
    started = time.time()
    seen = 0

    with torch.no_grad():
        for utt_ids, batch in iterator:
            batch = to_device(batch, device)
            logits = logits_from_model(
                model,
                batch["speech"],
                batch["speech_lengths"],
                len(labels),
                args.apply_loss_scale,
            )
            probs = torch.softmax(logits, dim=-1)
            probs_cpu = probs.detach().float().cpu().numpy()
            logits_cpu = logits.detach().float().cpu().numpy()
            logits_chunks.append(logits_cpu)
            utt_order.extend(list(utt_ids))
            for utt, prob in zip(utt_ids, probs_cpu):
                if utt in probs_by_utt:
                    raise ValueError(f"duplicate utterance id from iterator: {utt}")
                probs_by_utt[utt] = prob
            seen += len(utt_ids)
            if args.log_interval > 0 and seen % args.log_interval == 0:
                elapsed = max(time.time() - started, 1e-6)
                LOGGER.info(
                    "processed %d utterances (%.2f utt/s)", seen, seen / elapsed
                )

    if args.logits_output is not None:
        args.logits_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.logits_output,
            utt_ids=np.asarray(utt_order),
            labels=np.asarray(labels),
            logits=(
                np.concatenate(logits_chunks, axis=0)
                if logits_chunks
                else np.empty((0, len(labels)), dtype=np.float32)
            ),
        )
        LOGGER.info("wrote logits: %s", args.logits_output)

    return labels, probs_by_utt


def labels_for_threshold(
    labels: List[str], prob: np.ndarray, threshold: float
) -> List[Tuple[str, float]]:
    inds = np.flatnonzero(prob >= threshold)
    if len(inds) == 0:
        return []
    inds = inds[np.argsort(prob[inds])[::-1]]
    return [(labels[int(i)], float(prob[int(i)])) for i in inds]


def labels_for_topk(
    labels: List[str], prob: np.ndarray, topk: int
) -> List[Tuple[str, float]]:
    topk = min(topk, len(labels))
    inds = np.argpartition(-prob, kth=topk - 1)[:topk]
    inds = inds[np.argsort(prob[inds])[::-1]]
    return [(labels[int(i)], float(prob[int(i)])) for i in inds]


def score_predictions(
    labels: List[str],
    probs_by_utt: Dict[str, np.ndarray],
    refs: Dict[str, Tuple[str, ...]],
    thresholds: List[float],
) -> dict:
    total = len(refs)
    missing: List[str] = []
    top1_exact = top2_exact = 0
    per_card: Dict[str, Dict[str, int]] = {}
    threshold_out = {str(thr): {"exact": 0, "cardinality": {}} for thr in thresholds}

    for utt, ref in refs.items():
        prob = probs_by_utt.get(utt)
        card = str(len(ref))
        per_card.setdefault(
            card,
            {"total": 0, "top1_exact": 0, "top2_exact": 0},
        )
        per_card[card]["total"] += 1
        if prob is None:
            missing.append(utt)
            continue
        ref_set = set(ref)
        top1 = labels_for_topk(labels, prob, 1)
        top2 = labels_for_topk(labels, prob, 2)
        if set(x for x, _ in top1) == ref_set:
            top1_exact += 1
            per_card[card]["top1_exact"] += 1
        if set(x for x, _ in top2) == ref_set:
            top2_exact += 1
            per_card[card]["top2_exact"] += 1
        for thr in thresholds:
            key = str(thr)
            pred = labels_for_threshold(labels, prob, thr)
            pred_set = set(x for x, _ in pred)
            if pred_set == ref_set:
                threshold_out[key]["exact"] += 1
            c = str(len(pred))
            threshold_out[key]["cardinality"][c] = (
                threshold_out[key]["cardinality"].get(c, 0) + 1
            )

    per_card_out = {}
    for card, stats in sorted(per_card.items(), key=lambda x: int(x[0])):
        denom = stats["total"]
        per_card_out[card] = {
            "total": denom,
            "top1_exact": stats["top1_exact"],
            "top1_accuracy": safe_div(stats["top1_exact"], denom),
            "top2_exact": stats["top2_exact"],
            "top2_accuracy": safe_div(stats["top2_exact"], denom),
        }
    threshold_results = {}
    for key, stats in threshold_out.items():
        threshold_results[key] = {
            "exact": stats["exact"],
            "set_accuracy": safe_div(stats["exact"], total),
            "prediction_cardinality": dict(
                sorted(stats["cardinality"].items(), key=lambda kv: int(kv[0]))
            ),
        }

    return {
        "num_ref": total,
        "num_pred": len(probs_by_utt),
        "num_missing_pred": len(missing),
        "missing_pred_examples": missing[:20],
        "top1_exact": top1_exact,
        "top1_set_accuracy": safe_div(top1_exact, total),
        "top2_exact": top2_exact,
        "top2_set_accuracy": safe_div(top2_exact, total),
        "per_ref_cardinality": per_card_out,
        "threshold_results": threshold_results,
    }


def write_tsv(
    path: Path,
    labels: List[str],
    probs_by_utt: Dict[str, np.ndarray],
    topk: int,
    thresholds: List[float],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        header = ["utt"]
        header += [f"top{i + 1}" for i in range(topk)]
        header += [f"thr{thr:g}" for thr in thresholds]
        f.write("\t".join(header) + "\n")
        for utt in sorted(probs_by_utt):
            prob = probs_by_utt[utt]
            row = [utt]
            row += [f"{lab}:{p:.8f}" for lab, p in labels_for_topk(labels, prob, topk)]
            for thr in thresholds:
                pred = labels_for_threshold(labels, prob, thr)
                row.append(",".join(f"{lab}:{p:.8f}" for lab, p in pred))
            f.write("\t".join(row) + "\n")
    LOGGER.info("wrote predictions: %s", path)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lid_train_config", type=Path, required=True)
    p.add_argument("--lid_model_file", type=Path, required=True)
    p.add_argument("--lang2utt", type=Path, required=True)
    p.add_argument("--wav_scp", type=Path, required=True)
    p.add_argument("--ref_utt2langs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--logits_output", type=Path, required=True)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.3])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ngpu", type=int, default=0)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--apply_loss_scale", action="store_true")
    p.add_argument("--allow_missing_pred", action="store_true")
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    thresholds = sorted({float(x) for x in args.thresholds}, reverse=True)
    labels, probs_by_utt = infer(args)
    write_tsv(args.output, labels, probs_by_utt, args.topk, thresholds)
    refs = read_refs(args.ref_utt2langs)
    missing = sorted(set(refs) - set(probs_by_utt))
    extra = sorted(set(probs_by_utt) - set(refs))
    if extra:
        raise ValueError(
            f"predictions contain {len(extra)} utterances absent from references: "
            f"{extra[:20]}"
        )
    if missing and not args.allow_missing_pred:
        raise ValueError(
            f"missing predictions for {len(missing)} references: {missing[:20]}"
        )
    result = score_predictions(labels, probs_by_utt, refs, thresholds)
    result.update(
        {
            "score_type": "lid2_softmax_logits",
            "lid_train_config": str(args.lid_train_config),
            "lid_model_file": str(args.lid_model_file),
            "lang2utt": str(args.lang2utt),
            "wav_scp": str(args.wav_scp),
            "ref_utt2langs": str(args.ref_utt2langs),
            "logits_output": str(args.logits_output),
            "thresholds": thresholds,
            "apply_loss_scale": bool(args.apply_loss_scale),
        }
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
