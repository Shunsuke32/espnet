#!/usr/bin/env python3
"""Compute top-k posterior baselines from a trained ESPnet `lid1` model.

This is used after training the FLEURS-only LID classifier:

* FLEURS: top-1 accuracy on single-label utterances.
* CS-FLEURS: unordered top-2 set accuracy against `utt2langs`.

The class index -> language mapping follows ESPnet `lid_inference.py`: the order
of labels in the training `lang2utt` file.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
from pathlib import Path
from typing import Dict, IO, List, Optional, Tuple


LOGGER = logging.getLogger("lid_topk_softmax")
PAIR_CLASS_RE = re.compile(r"^[a-z]{3}(?:-[a-z]{3})+$")


def read_label_map(path: Optional[Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if path is None:
        return mapping
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) >= 2:
                mapping[parts[0].lower()] = parts[1].lower()
    return mapping


def canon(label: str, mapping: Dict[str, str]) -> str:
    x = label.strip().lower().strip("<>[](){}'\"")
    candidates: List[str] = []
    for cand in (x, x.replace("-", "_"), x.replace("_", "-")):
        if cand and cand not in candidates:
            candidates.append(cand)
    for cand in candidates:
        if cand in mapping:
            return mapping[cand]
    return candidates[0] if candidates else x


def read_lang2utt(path: Path, mapping: Dict[str, str]) -> List[str]:
    labels: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                labels.append(canon(parts[0], mapping))
    if not labels:
        raise ValueError(f"empty lang2utt: {path}")
    dup = sorted({x for x in labels if labels.count(x) > 1})
    if dup:
        raise ValueError(f"canonicalized labels are not unique in {path}: {dup[:20]}")
    return labels


def read_refs(path: Path, mapping: Dict[str, str]) -> Dict[str, Tuple[str, ...]]:
    refs: Dict[str, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            labs: List[str] = []
            for tok in parts[1:]:
                lab = canon(tok, mapping)
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


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def open_text_writer(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", compresslevel=1)
    return path.open("w", encoding="utf-8")


def write_jsonl(handle: IO[str], obj: dict) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def expand_atomic_class(label: str) -> Tuple[str, ...]:
    if PAIR_CLASS_RE.fullmatch(label):
        return tuple(label.split("-"))
    return (label,)


def score_topk(
    preds: Dict[str, List[Tuple[str, float]]],
    refs: Dict[str, Tuple[str, ...]],
    topk: int,
    prediction_mode: str,
) -> dict:
    single_total = single_top1 = 0
    pair_total = pair_top2 = 0
    contains_total = contains_ok = 0
    missing = []
    for utt, ref in refs.items():
        hyp = [x for x, _ in preds.get(utt, [])]
        if not hyp:
            missing.append(utt)
        if prediction_mode == "atomic_top1":
            predicted_set = set(expand_atomic_class(hyp[0])) if hyp else set()
            contained_set = predicted_set
        else:
            predicted_set = set(hyp[: len(set(ref))])
            contained_set = set(hyp[:topk])
        if len(ref) == 1:
            single_total += 1
            if predicted_set == set(ref):
                single_top1 += 1
        if len(ref) == 2:
            pair_total += 1
            if predicted_set == set(ref):
                pair_top2 += 1
        contains_total += 1
        if set(ref).issubset(contained_set):
            contains_ok += 1
    return {
        "num_ref": len(refs),
        "num_missing_pred": len(missing),
        "missing_pred_examples": missing[:20],
        "prediction_mode": prediction_mode,
        "topk": topk,
        "single_label_top1_accuracy": safe_div(single_top1, single_total),
        "single_label_total": single_total,
        "two_label_unordered_top2_accuracy": safe_div(pair_top2, pair_total),
        "two_label_total": pair_total,
        "all_reference_labels_contained_in_topk": safe_div(contains_ok, contains_total),
    }


def write_details(
    path: Path,
    preds: Dict[str, List[Tuple[str, float]]],
    refs: Dict[str, Tuple[str, ...]],
    prediction_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("utt\tref\tpred\treference_cardinality\texact\n")
        for utt, ref in sorted(refs.items()):
            hyp = [label for label, _ in preds[utt]]
            if prediction_mode == "atomic_top1":
                predicted = expand_atomic_class(hyp[0])
            else:
                predicted = tuple(hyp[: len(set(ref))])
            exact = int(set(predicted) == set(ref))
            f.write(
                f"{utt}\t{' '.join(ref)}\t{' '.join(predicted)}\t{len(ref)}\t{exact}\n"
            )


def build_iterator(args: argparse.Namespace, lid_train_args: argparse.Namespace):
    """Build the same kind of inference iterator used by espnet2.bin.lid_inference."""
    from espnet2.tasks.lid import LIDTask

    override = {
        "valid_data_path_and_name_and_type": [(str(args.wav_scp), "speech", "sound")],
        "valid_shape_file": [],
        "dtype": args.dtype,
        "valid_batch_size": args.batch_size,
        "num_workers": args.num_workers,
        # Match ESPnet's official lid.sh stage-6 inference path: provide only
        # speech and disable the LID preprocessor.  LIDPreprocessor expects
        # lid_labels when enabled, which would break posterior-only inference.
        "use_preprocessor": False,
        "preprocessor_conf": {
            "fix_duration": False,
            "target_duration": None,
            "noise_apply_prob": 0.0,
            "rir_apply_prob": 0.0,
        },
    }
    merged = vars(lid_train_args).copy()
    merged.update(override)
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
    """Return the class-prototype matrix used by ESPnet LID losses.

    Most ESPnet AAMSoftmax-style losses expose it as ``loss.weight``.  Some
    classifier implementations wrap the matrix in a child module, so keep a
    conservative fallback rather than silently using the wrong tensor.
    """
    for attr in ("weight", "W"):
        weight = getattr(loss, attr, None)
        if weight is not None and getattr(weight, "ndim", None) == 2:
            return weight
    for attr in ("fc", "linear", "classifier"):
        module = getattr(loss, attr, None)
        weight = getattr(module, "weight", None)
        if weight is not None and getattr(weight, "ndim", None) == 2:
            return weight
    raise RuntimeError(
        f"Unsupported LID loss for top-k scoring: {type(loss)} has no 2-D class weight matrix"
    )


def logits_from_model(
    model,
    speech,
    speech_lengths,
    apply_loss_scale: bool,
    num_labels: int,
):
    import torch.nn.functional as F

    feats, feat_lengths = model.extract_feats(speech, speech_lengths)
    frame = model.encode_frame(feats)
    utt_level = model.pooling(frame, feat_lengths=feat_lengths)
    emb = model.project_lang_embd(utt_level)
    weight = _find_loss_weight(model.loss)
    logits = F.linear(F.normalize(emb), F.normalize(weight))

    # Softmax/AAMSoftmax have one prototype per class.  The ESPnet
    # aamsoftmax_sc_topk loss has K sub-centers per class and its own forward()
    # reshapes (B, K*nclasses) -> (B, nclasses, K) then max-pools over K.
    if logits.shape[1] != num_labels:
        if logits.shape[1] % num_labels != 0:
            raise RuntimeError(
                f"LID loss produced {logits.shape[1]} logits, incompatible with "
                f"{num_labels} labels from lang2utt"
            )
        k = logits.shape[1] // num_labels
        logits = logits.reshape(logits.shape[0], num_labels, k).max(dim=2).values

    # Scaling does not change top-k, but gives calibrated-looking posteriors.
    if apply_loss_scale:
        scale = getattr(model.loss, "scale", getattr(model.loss, "s", None))
        if scale is not None:
            logits = logits * float(scale)
    return logits


def infer(args: argparse.Namespace) -> Dict[str, List[Tuple[str, float]]]:
    import torch
    from espnet2.tasks.lid import LIDTask
    from espnet2.torch_utils.device_funcs import to_device

    device = "cuda" if args.ngpu > 0 else "cpu"
    model, train_args = LIDTask.build_model_from_file(
        str(args.lid_train_config), str(args.lid_model_file), device
    )
    model.eval()
    idx2lang = read_lang2utt(args.lang2utt, read_label_map(args.label_map))
    if args.topk > len(idx2lang):
        raise ValueError(
            f"--topk {args.topk} exceeds number of LID classes {len(idx2lang)}"
        )
    iterator = build_iterator(args, train_args)
    preds: Dict[str, List[Tuple[str, float]]] = {}
    seen = 0
    loss_scale = getattr(model.loss, "scale", getattr(model.loss, "s", None))
    loss_scale = float(loss_scale) if loss_scale is not None else None
    args.loss_scale = loss_scale
    full_handle = None
    try:
        if args.full_output:
            full_handle = open_text_writer(args.full_output)
            write_jsonl(
                full_handle,
                {
                    "type": "metadata",
                    "score_type": "lid_classifier_softmax",
                    "lid_train_config": str(args.lid_train_config),
                    "lid_model_file": str(args.lid_model_file),
                    "lang2utt": str(args.lang2utt),
                    "label_map": str(args.label_map) if args.label_map else None,
                    "apply_loss_scale": bool(args.apply_loss_scale),
                    "loss_scale": loss_scale,
                    "labels": idx2lang,
                },
            )
        with torch.no_grad():
            for utt_ids, batch in iterator:
                batch = to_device(batch, device)
                logits = logits_from_model(
                    model,
                    batch["speech"],
                    batch["speech_lengths"],
                    args.apply_loss_scale,
                    len(idx2lang),
                )
                probs = torch.softmax(logits, dim=-1)
                vals, inds = torch.topk(probs, k=args.topk, dim=-1)
                vals_cpu = vals.detach().cpu()
                inds_cpu = inds.detach().cpu()
                logits_cpu = logits.detach().cpu() if full_handle is not None else None
                probs_cpu = probs.detach().cpu() if full_handle is not None else None
                for i, utt in enumerate(utt_ids):
                    top = [
                        (idx2lang[int(j)], float(v))
                        for j, v in zip(inds_cpu[i], vals_cpu[i])
                    ]
                    if utt in preds:
                        raise ValueError(f"duplicate utterance id from iterator: {utt}")
                    preds[utt] = top
                    if full_handle is not None:
                        assert logits_cpu is not None
                        assert probs_cpu is not None
                        write_jsonl(
                            full_handle,
                            {
                                "type": "utt",
                                "utt": utt,
                                "topk": [
                                    {
                                        "label": idx2lang[int(j)],
                                        "prob": float(v),
                                        "logit": float(logits_cpu[i, int(j)]),
                                    }
                                    for j, v in zip(inds_cpu[i], vals_cpu[i])
                                ],
                                "logits": [float(x) for x in logits_cpu[i].tolist()],
                                "probs": [float(x) for x in probs_cpu[i].tolist()],
                            },
                        )
                    seen += 1
                    if args.limit is not None and seen >= args.limit:
                        return preds
    finally:
        if full_handle is not None:
            full_handle.close()
    return preds


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lid_train_config", type=Path, required=True)
    p.add_argument("--lid_model_file", type=Path, required=True)
    p.add_argument(
        "--lang2utt",
        type=Path,
        required=True,
        help="training lang2utt defining class order",
    )
    p.add_argument("--wav_scp", type=Path, required=True)
    p.add_argument("--ref_utt2langs", type=Path, default=None)
    p.add_argument("--label_map", type=Path, default=None)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument(
        "--prediction_mode",
        choices=("individual_topk", "atomic_top1"),
        default="individual_topk",
        help="interpret classes as individual languages or atomic pair labels",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ngpu", type=int, default=0)
    p.add_argument("--dtype", default="float32")
    p.add_argument(
        "--target_duration",
        type=float,
        default=3.0,
        help="kept for compatibility; top-k inference disables LIDPreprocessor",
    )
    p.add_argument(
        "--fix_duration",
        action="store_true",
        help="kept for compatibility; top-k inference disables LIDPreprocessor",
    )
    p.add_argument("--apply_loss_scale", action="store_true")
    p.add_argument(
        "--allow_missing_pred",
        action="store_true",
        help="do not fail when references lack predictions",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="debug: stop after N utterances"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--results", type=Path, default=None)
    p.add_argument("--details_out", type=Path, default=None)
    p.add_argument(
        "--full_output",
        type=Path,
        default=None,
        help="optional JSONL(.gz) dump of all-label logits and probabilities",
    )
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()
    if args.ref_utt2langs and args.topk < 2:
        raise SystemExit(
            "--topk must be >= 2 when scoring references because CS top-2 accuracy is reported"
        )
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    preds = infer(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for utt in sorted(preds):
            f.write(
                utt
                + "\t"
                + "\t".join(f"{lab}:{prob:.8f}" for lab, prob in preds[utt])
                + "\n"
            )
    LOGGER.info("wrote %s", args.output)

    if args.ref_utt2langs:
        refs = read_refs(args.ref_utt2langs, read_label_map(args.label_map))
        extra = sorted(set(preds) - set(refs))
        if extra:
            raise ValueError(
                f"predictions contain {len(extra)} utterances absent from "
                f"references: {extra[:20]}"
            )
        result = score_topk(preds, refs, args.topk, args.prediction_mode)
        if result["num_missing_pred"] == 0 and args.details_out is not None:
            write_details(args.details_out, preds, refs, args.prediction_mode)
        result.update(
            {
                "score_type": "lid_classifier_softmax",
                "lid_train_config": str(args.lid_train_config),
                "lid_model_file": str(args.lid_model_file),
                "lang2utt": str(args.lang2utt),
                "wav_scp": str(args.wav_scp),
                "ref_utt2langs": str(args.ref_utt2langs),
                "label_map": str(args.label_map) if args.label_map else None,
                "apply_loss_scale": bool(args.apply_loss_scale),
                "loss_scale": getattr(args, "loss_scale", None),
                "prediction_mode": args.prediction_mode,
                "full_output": str(args.full_output) if args.full_output else None,
                "details_out": str(args.details_out) if args.details_out else None,
            }
        )
        text = json.dumps(result, indent=2, ensure_ascii=False)
        print(text)
        if args.results:
            args.results.parent.mkdir(parents=True, exist_ok=True)
            args.results.write_text(text + "\n", encoding="utf-8")
        if not args.allow_missing_pred and result["num_missing_pred"] != 0:
            raise SystemExit(
                f"missing predictions: {result['num_missing_pred']} / {result['num_ref']}"
            )


if __name__ == "__main__":
    main()
