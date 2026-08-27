#!/usr/bin/env python3
"""Top-k evaluation for LID3 sigmoid/BCE multi-label models.

The standard ESPnet LID inference path is single-label oriented.  LID3 reads
raw logits from the loss head, applies sigmoid, and scores forced top-k sets.
Both the linear BCE head and cosine/sub-center BCE head are supported.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


LOGGER = logging.getLogger("lid_sigmoid_topk")


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
        dup = sorted({x for x in labels if labels.count(x) > 1})[:20]
        raise ValueError(f"duplicate labels in lang2utt: {dup}")
    return labels


def read_refs(path: Path) -> Dict[str, Tuple[str, ...]]:
    refs: Dict[str, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            labels: List[str] = []
            for tok in parts[1:]:
                lab = canon(tok)
                if not lab:
                    continue
                if lab in labels:
                    raise ValueError(
                        f"duplicate reference label for {parts[0]}: {lab}"
                    )
                labels.append(lab)
            utt = parts[0]
            if utt in refs:
                raise ValueError(f"duplicate utterance id in reference: {utt}")
            if not labels:
                raise ValueError(f"empty reference label set: {utt}")
            refs[utt] = tuple(labels)
    return refs


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def get_threshold_labels(pred: Dict[str, object], key: str) -> Tuple[str, ...]:
    prefix = "threshold_labels_by_value."
    if key.startswith(prefix):
        threshold_key = key[len(prefix) :]
        by_value = pred["threshold_labels_by_value"]  # type: ignore[index]
        return tuple(by_value[threshold_key])  # type: ignore[index]
    return tuple(pred[key])  # type: ignore[arg-type]


def score_predictions(
    preds: Dict[str, Dict[str, object]],
    refs: Dict[str, Tuple[str, ...]],
    forced_k: int,
    threshold_label_key: str = "threshold_labels",
    threshold_value: float = 0.5,
) -> dict:
    total = 0
    exact = 0
    missing: List[str] = []
    per_card: Dict[str, Dict[str, int]] = {}
    threshold_exact = 0
    threshold_card: Dict[str, int] = {}

    for utt, ref in refs.items():
        total += 1
        pred = preds.get(utt)
        if pred is None:
            missing.append(utt)
            continue
        forced = tuple(pred["forced_labels"][:forced_k])  # type: ignore[index]
        thresh = get_threshold_labels(pred, threshold_label_key)
        card_key = str(len(ref))
        per_card.setdefault(card_key, {"total": 0, "exact": 0})
        per_card[card_key]["total"] += 1
        if set(forced) == set(ref):
            exact += 1
            per_card[card_key]["exact"] += 1
        if set(thresh) == set(ref):
            threshold_exact += 1
        threshold_card[str(len(thresh))] = threshold_card.get(str(len(thresh)), 0) + 1

    per_card_out = {
        k: {
            "total": v["total"],
            "exact": v["exact"],
            "accuracy": safe_div(v["exact"], v["total"]),
        }
        for k, v in sorted(per_card.items(), key=lambda kv: int(kv[0]))
    }
    return {
        "num_ref": total,
        "num_pred": len(preds),
        "num_missing_pred": len(missing),
        "missing_pred_examples": missing[:20],
        "forced_k": forced_k,
        "forced_topk_set_accuracy": safe_div(exact, total),
        "forced_topk_exact": exact,
        "per_ref_cardinality": per_card_out,
        "threshold": threshold_value,
        "threshold_set_accuracy": safe_div(threshold_exact, total),
        "threshold_exact": threshold_exact,
        "threshold_prediction_cardinality": dict(
            sorted(threshold_card.items(), key=lambda kv: int(kv[0]))
        ),
    }


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


def logits_from_model(model, speech, speech_lengths):
    feats, feat_lengths = model.extract_feats(speech, speech_lengths)
    frame = model.encode_frame(feats)
    utt_level = model.pooling(frame, feat_lengths=feat_lengths)
    emb = model.project_lang_embd(utt_level)
    compute_logits = getattr(model.loss, "compute_logits", None)
    if compute_logits is not None:
        return compute_logits(emb)
    classifier = getattr(model.loss, "classifier", None)
    if classifier is None:
        raise RuntimeError(
            f"expected a BCE loss head with compute_logits/classifier, "
            f"got {type(model.loss)}"
        )
    return classifier(emb)


def infer(args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    import torch
    from espnet2.tasks.lid import LIDTask
    from espnet2.torch_utils.device_funcs import to_device

    device = "cuda" if args.ngpu > 0 else "cpu"
    model, train_args = LIDTask.build_model_from_file(
        str(args.lid_train_config), str(args.lid_model_file), device
    )
    model.eval()

    idx2lang = read_lang2utt(args.lang2utt)
    thresholds = sorted({float(args.threshold), *map(float, args.extra_thresholds)})
    nclasses = getattr(model.loss, "nclasses", None)
    if nclasses is not None and int(nclasses) != len(idx2lang):
        raise ValueError(
            f"model has {nclasses} classes but lang2utt has {len(idx2lang)} labels"
        )
    if args.topk > len(idx2lang):
        raise ValueError(f"--topk {args.topk} exceeds {len(idx2lang)} labels")

    iterator = build_iterator(args, train_args)
    preds: Dict[str, Dict[str, object]] = {}
    logit_chunks: List[np.ndarray] = []
    utt_order: List[str] = []
    started = time.time()
    seen = 0
    with torch.no_grad():
        for utt_ids, batch in iterator:
            batch = to_device(batch, device)
            logits = logits_from_model(model, batch["speech"], batch["speech_lengths"])
            logits_cpu = logits.detach().float().cpu()
            if args.logits_output is not None:
                logit_chunks.append(logits_cpu.numpy())
                utt_order.extend(list(utt_ids))
            probs = torch.sigmoid(logits)
            vals, inds = torch.topk(probs, k=args.topk, dim=-1)
            threshold_masks = {thr: (probs >= thr).cpu() for thr in thresholds}
            probs_cpu = probs.cpu()
            for i, utt in enumerate(utt_ids):
                if utt in preds:
                    raise ValueError(f"duplicate utterance id from iterator: {utt}")
                forced = [
                    (idx2lang[int(j)], float(v))
                    for j, v in zip(inds[i].cpu(), vals[i].cpu())
                ]
                threshold_by_value = {}
                threshold_labels_by_value = {}
                for thr in thresholds:
                    thresh = [
                        (idx2lang[j], float(probs_cpu[i, j]))
                        for j, on in enumerate(threshold_masks[thr][i].tolist())
                        if on
                    ]
                    thresh.sort(key=lambda x: x[1], reverse=True)
                    key = f"{thr:g}"
                    threshold_by_value[key] = thresh
                    threshold_labels_by_value[key] = [x[0] for x in thresh]
                main_key = f"{float(args.threshold):g}"
                preds[utt] = {
                    "forced": forced,
                    "forced_labels": [x[0] for x in forced],
                    "threshold": threshold_by_value[main_key],
                    "threshold_labels": threshold_labels_by_value[main_key],
                    "threshold_by_value": threshold_by_value,
                    "threshold_labels_by_value": threshold_labels_by_value,
                }
            seen += len(utt_ids)
            if args.progress_interval > 0 and seen % args.progress_interval == 0:
                elapsed = max(time.time() - started, 1.0)
                LOGGER.info(
                    "processed %d utterances (%.2f utt/s)", seen, seen / elapsed
                )
    if args.logits_output is not None:
        logits_all = (
            np.concatenate(logit_chunks, axis=0)
            if logit_chunks
            else np.empty((0, len(idx2lang)), dtype=np.float32)
        )
        args.logits_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.logits_output,
            utt_ids=np.asarray(utt_order),
            labels=np.asarray(idx2lang),
            logits=logits_all.astype(np.float32, copy=False),
        )
        LOGGER.info("wrote logits: %s shape=%s", args.logits_output, logits_all.shape)
    return preds


def write_predictions(path: Path, preds: Dict[str, Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for utt in sorted(preds):
            pred = preds[utt]
            forced = " ".join(
                f"{lab}:{prob:.8f}"
                for lab, prob in pred["forced"]  # type: ignore[index]
            )
            thresh = " ".join(
                f"{lab}:{prob:.8f}"
                for lab, prob in pred["threshold"]  # type: ignore[index]
            )
            extra = []
            for key, vals in sorted(
                pred.get("threshold_by_value", {}).items(),  # type: ignore[union-attr]
                key=lambda kv: float(kv[0]),
            ):
                if key == "0.5":
                    continue
                text = " ".join(f"{lab}:{prob:.8f}" for lab, prob in vals)
                extra.append(f"threshold_{key}={text}")
            suffix = "\t" + "\t".join(extra) if extra else ""
            f.write(f"{utt}\tforced={forced}\tthreshold={thresh}{suffix}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--lid_train_config", type=Path, required=True)
    parser.add_argument("--lid_model_file", type=Path, required=True)
    parser.add_argument("--lang2utt", type=Path, required=True)
    parser.add_argument("--wav_scp", type=Path, required=True)
    parser.add_argument("--ref_utt2langs", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--score_k", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--extra_thresholds",
        type=float,
        nargs="*",
        default=[0.3],
        help="additional sigmoid thresholds to score in the same forward pass",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--logits_output",
        type=Path,
        default=None,
        help="optional .npz file with utt_ids, labels, and raw logits",
    )
    parser.add_argument("--allow_missing_pred", action="store_true")
    parser.add_argument("--progress_interval", type=int, default=500)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    preds = infer(args)
    write_predictions(args.output, preds)
    refs = read_refs(args.ref_utt2langs)
    extra = sorted(set(preds) - set(refs))
    if extra:
        raise ValueError(
            f"predictions contain {len(extra)} utterances absent from references: "
            f"{extra[:20]}"
        )
    result = score_predictions(
        preds,
        refs,
        args.score_k,
        threshold_value=float(args.threshold),
    )
    extra_threshold_results = {}
    for thr in sorted(
        {float(x) for x in args.extra_thresholds} - {float(args.threshold)}
    ):
        key = f"{thr:g}"
        extra_threshold_results[key] = score_predictions(
            preds,
            refs,
            args.score_k,
            threshold_label_key=f"threshold_labels_by_value.{key}",
            threshold_value=thr,
        )
    result.update(
        {
            "score_type": "lid3_sigmoid_forced_topk",
            "lid_train_config": str(args.lid_train_config),
            "lid_model_file": str(args.lid_model_file),
            "lang2utt": str(args.lang2utt),
            "wav_scp": str(args.wav_scp),
            "ref_utt2langs": str(args.ref_utt2langs),
            "extra_threshold_results": extra_threshold_results,
        }
    )
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not args.allow_missing_pred and result["num_missing_pred"] != 0:
        raise SystemExit(f"missing predictions: {result['num_missing_pred']}")


if __name__ == "__main__":
    main()
