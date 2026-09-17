#!/usr/bin/env python3
"""Freeze LID scores once, then score all requested views without more inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
import sys

import numpy as np
import yaml


RECIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RECIPE / "evaluation"))
from lid_metrics import (  # noqa: E402
    check_alignment,
    class_languages,
    evaluate_set,
    language_set,
    plot_results,
    probabilities,
    read_manifest,
    read_refs,
    validate_labels,
    validate_scores,
    write_json,
)

CS_SUBSETS = (
    "test_cs_read_test",
    "test_cs_xtts_test1",
    "test_cs_xtts_test2",
    "test_cs_mms_test",
)
LOGGER = logging.getLogger(__name__)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def head_spec(config, labels):
    expanded = validate_labels(labels)
    loss = config.get("loss", "aamsoftmax")
    conf = config.get("loss_conf") or {}
    if loss == "arc_margin_subcenter_intertopk_multilabel_bce":
        activation, scale, sweep = "sigmoid", float(conf.get("scale", 32)), True
    elif loss in {
        "aamsoftmax",
        "aamsoftmax_sc_topk",
        "aamsoftmax_sc_topk_softtarget",
        "softmax",
    }:
        activation = "softmax"
        default = 15 if loss == "aamsoftmax" else 32
        scale = 1.0 if loss == "softmax" else float(conf.get("scale", default))
        sweep = loss == "aamsoftmax_sc_topk_softtarget"
    else:
        raise ValueError(f"unsupported LID loss: {loss}")
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError("loss scale must be positive and finite")
    atomic = any(len(x) == 2 for x in expanded)
    if atomic and sweep:
        raise ValueError("KL/BCE heads must use individual-language classes")
    if config.get("lang_num") != len(labels):
        raise ValueError("config lang_num differs from frozen training class inventory")
    return dict(
        loss=loss,
        activation=activation,
        scale=scale,
        sweep=sweep,
        prediction_mode="atomic_top1" if atomic else "individual_topk",
    )


def read_exclusions(path):
    if path is None:
        return {}
    excluded = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["utt", "reason"]:
            raise ValueError("exclusions must be a TSV with header: utt<TAB>reason")
        for row in reader:
            if not row["utt"] or not row["reason"].strip() or row["utt"] in excluded:
                raise ValueError("empty or duplicate exclusion ID/reason")
            excluded[row["utt"]] = row["reason"].strip()
    return excluded


def prepare_views(data_dir, names, excluded):
    names = [name for value in names for name in value.split()]
    if not names or len(set(names)) != len(names):
        raise ValueError("empty or duplicate test set names")
    if any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError("test sets must be directory names, not paths")
    if "test_cs_all" in names and all(
        (data_dir / name).is_dir() for name in CS_SUBSETS
    ):
        names = list(dict.fromkeys([*names, *CS_SUBSETS]))
    aggregate = all(name in names for name in CS_SUBSETS)
    names = [name for name in names if not (aggregate and name == "test_cs_all")]
    views, wavs, encountered, owners = {}, {}, set(), {}
    for name in names:
        directory = data_dir / name
        if (directory / "segments").exists():
            raise ValueError(f"format segmented audio before evaluation: {directory}")
        refs, ref_ids = read_refs(directory / "utt2langs", excluded)
        audio, audio_ids = read_manifest(directory / "wav.scp", excluded)
        encountered.update(ref_ids | audio_ids)
        check_alignment(refs, audio, f"{name} reference/wav.scp")
        if not refs:
            raise ValueError(f"empty evaluation set after exclusions: {name}")
        for utt, value in audio.items():
            if utt in owners:
                raise ValueError(
                    f"duplicate ID across non-aggregate sets: {utt} in {owners[utt]} and {name}"
                )
            if value.endswith("|"):
                raise ValueError(
                    f"format piped wav.scp before evaluation: {name}/{utt}"
                )
            path = Path(value)
            if not path.is_absolute():
                path = (
                    path.resolve() if path.is_file() else (directory / path).resolve()
                )
            wavs[utt] = str(path)
            owners[utt] = name
        views[name] = refs
    if aggregate:
        combined = {utt: ref for name in CS_SUBSETS for utt, ref in views[name].items()}
        directory = data_dir / "test_cs_all"
        if directory.exists():
            refs, ref_ids = read_refs(directory / "utt2langs", excluded)
            audio, audio_ids = read_manifest(directory / "wav.scp", excluded)
            encountered.update(ref_ids | audio_ids)
            check_alignment(combined, refs, "CS subset union/reference")
            check_alignment(combined, audio, "CS subset union/wav.scp")
            if refs != combined:
                raise ValueError(
                    "CS aggregate reference labels differ from subset labels"
                )
            # Formatted aggregate audio may live at different paths. The union
            # is a scored view of the subsets, never a second audio input.
        views["test_cs_all"] = combined
    unused = set(excluded) - encountered
    if unused:
        raise ValueError(
            f"exclusions not present in selected manifests: {sorted(unused)[:10]}"
        )
    return views, wavs


def training_inventory(args, config):
    supplied = args.train_data_dir / "lang2utt"
    frozen = args.config_file.parent / "lang2utt"
    selected = frozen if frozen.is_file() else supplied
    rows, _ = read_manifest(selected)
    labels = list(rows)
    validate_labels(labels)
    if supplied.is_file() and list(read_manifest(supplied)[0]) != labels:
        raise ValueError("experiment/train_data_dir lang2utt class order differs")
    configured = config.get("lang2utt")
    if configured:
        candidates = [
            Path(configured),
            args.config_file.parent / configured,
            RECIPE / "lid1" / configured,
        ]
        existing = next((path for path in candidates if path.is_file()), None)
        if existing is not None and list(read_manifest(existing)[0]) != labels:
            raise ValueError(
                "config lang2utt class order differs from frozen inventory"
            )
    if not frozen.is_file():
        LOGGER.warning(
            "No experiment lang2utt snapshot; using explicitly supplied training inventory %s and freezing it in the evaluation output",
            selected,
        )
    train_file = args.train_data_dir / "utt2langs"
    train_refs, _ = read_manifest(train_file)
    if not train_refs:
        raise ValueError("empty training utt2langs")
    seen_pairs = set()
    for value in train_refs.values():
        tokens = value.split()
        # Pair training views store one atomic class even in utt2langs. Only
        # training inventory accepts this representation; test refs stay strict.
        ref = class_languages(tokens[0]) if len(tokens) == 1 else language_set(tokens)
        if len(ref) == 2:
            seen_pairs.add(ref)
    return (
        labels,
        selected,
        train_file,
        seen_pairs,
    )


def extract_state_dict(checkpoint):
    """ESPnet trainer snapshots store weights under model, not at the root."""
    import torch

    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint is not a state dictionary")
    kind = "state_dict"
    if "model" in checkpoint:
        checkpoint, kind = checkpoint["model"], "trainer_checkpoint.model"
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("empty model state dictionary")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in checkpoint.items()
    ):
        raise ValueError(
            "unsupported checkpoint structure; expected tensors or ESPnet model dictionary"
        )
    return checkpoint, kind


def load_model(config_file, model_file, device, trusted_checkpoint=False):
    import torch
    from espnet2.tasks.lid import LIDTask

    unsafe_fallback = False
    try:
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        if not trusted_checkpoint:
            raise ValueError(
                "Safe checkpoint loading refused a pickled object. Only for a "
                "checkpoint you personally trust, pass --trusted_checkpoint to "
                "allow weights_only=False (pickle can execute arbitrary code)."
            ) from error
        LOGGER.warning(
            "Explicit --trusted_checkpoint: loading %s with weights_only=False; "
            "pickle can execute arbitrary code",
            model_file,
        )
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
        unsafe_fallback = True
    state, kind = extract_state_dict(checkpoint)
    if unsafe_fallback:
        kind += ":trusted_pickle"
    model, train_args = LIDTask.build_model_from_file(str(config_file), None, "cpu")
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    LOGGER.info(
        "Loaded explicit model %s (%s); no best/latest substitution", model_file, kind
    )
    return model, train_args, kind


def model_logits(model, speech, lengths, head, num_labels):
    if head["activation"] == "sigmoid":
        feats, feat_lengths = model.extract_feats(speech, lengths)
        pooled = model.pooling(model.encode_frame(feats), feat_lengths=feat_lengths)
        logits = model.loss.compute_logits(model.project_lang_embd(pooled))
    else:
        from lid_topk_softmax import logits_from_model

        weight = model.loss.weight
        centers = getattr(model.loss, "K", 1)
        if weight.shape[0] != num_labels * centers:
            raise ValueError(
                "model prototype count differs from frozen class inventory"
            )
        logits = logits_from_model(model, speech, lengths, True, num_labels)
    if logits.ndim != 2 or logits.shape[1] != num_labels:
        raise ValueError("model output differs from frozen class inventory")
    return logits


def infer_scores(args, labels, head, wav_scp):
    import torch
    from espnet2.torch_utils.device_funcs import to_device
    from lid_topk_softmax import build_iterator

    device = "cuda" if args.ngpu else "cpu"
    model, train_args, kind = load_model(
        args.config_file,
        args.model_file,
        device,
        trusted_checkpoint=args.trusted_checkpoint,
    )
    # Model construction does not consume lang2utt; inference uses the snapshot
    # and disables target preprocessing even if the saved config enabled it.
    train_args.lang2utt = str((args.output_dir / "inputs" / "lang2utt").resolve())
    iterator_args = argparse.Namespace(
        wav_scp=wav_scp,
        dtype="float32",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    utterances, chunks = [], []
    with torch.inference_mode():
        for utt_ids, batch in build_iterator(iterator_args, train_args):
            batch = to_device(batch, device)
            logits = model_logits(
                model, batch["speech"], batch["speech_lengths"], head, len(labels)
            )
            utterances.extend(utt_ids)
            chunks.append(logits.float().cpu().numpy())
    scores = np.concatenate(chunks) if chunks else np.empty((0, len(labels)))
    validate_scores(utterances, labels, scores)
    return utterances, scores, kind


def freeze_file(source, destination):
    content = source.read_bytes()
    if destination.exists() and destination.read_bytes() != content:
        raise ValueError(f"frozen input changed: {destination}; use a new output_dir")
    if not destination.exists():
        destination.write_bytes(content)


def run(args, inference=None):
    if args.ngpu not in (0, 1) or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError(
            "ngpu must be 0/1, batch_size positive, num_workers nonnegative"
        )
    if args.ngpu == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = yaml.safe_load(args.config_file.read_text(encoding="utf-8"))
    labels, lang2utt, train_refs, seen_pairs = training_inventory(args, config)
    head = head_spec(config, labels)
    excluded = read_exclusions(args.exclusions)
    views, wavs = prepare_views(args.data_dir, args.test_sets, excluded)
    provenance = dict(
        schema_version=1,
        model_sha256=digest_file(args.model_file),
        config_sha256=digest_file(args.config_file),
        lang2utt_sha256=digest_file(lang2utt),
        train_refs_sha256=digest_file(train_refs),
        wav_manifest_sha256=digest_value(wavs),
        head=head,
        labels=labels,
        batch_size=args.batch_size,
        dtype="float32",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = args.output_dir / "inputs"
    frozen.mkdir(exist_ok=True)
    for source, name in (
        (args.config_file, "config.yaml"),
        (lang2utt, "lang2utt"),
        (train_refs, "train.utt2langs"),
    ):
        freeze_file(source, frozen / name)
    scores_file = args.output_dir / "scores.npz"
    if scores_file.exists():
        with np.load(scores_file, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata["provenance"] != provenance:
                raise ValueError("saved score provenance changed; use a new output_dir")
            utt_ids = archive["utt_ids"].tolist()
            saved_labels = archive["labels"].tolist()
            logits, probs = archive["logits"], archive["probabilities"]
        if saved_labels != labels:
            raise ValueError("saved class order differs from frozen training inventory")
        validate_scores(utt_ids, labels, logits)
        if not np.array_equal(probs, probabilities(logits, head["activation"])):
            raise ValueError(
                "saved probabilities do not match declared logits/activation"
            )
        LOGGER.info("Reusing %s without loading a model", scores_file)
    else:
        if args.score_only:
            raise ValueError(f"no saved scores: {scores_file}")
        wav_scp = frozen / "inference.wav.scp"
        wav_scp.write_text(
            "".join(f"{utt} {value}\n" for utt, value in sorted(wavs.items())),
            encoding="utf-8",
        )
        utt_ids, logits, kind = (inference or infer_scores)(args, labels, head, wav_scp)
        validate_scores(utt_ids, labels, logits)
        check_alignment(wavs, utt_ids, "inference output")
        probs = probabilities(logits, head["activation"])
        metadata = dict(
            provenance=provenance,
            checkpoint_format=kind,
            model_file=str(args.model_file.resolve()),
            config_file=str(args.config_file.resolve()),
        )
        temporary = scores_file.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            utt_ids=np.asarray(utt_ids),
            labels=np.asarray(labels),
            logits=logits,
            probabilities=probs,
            metadata=np.asarray(json.dumps(metadata)),
        )
        temporary.replace(scores_file)
    check_alignment(wavs, utt_ids, "saved scores/input manifests")
    indices = {utt: i for i, utt in enumerate(utt_ids)}
    results = []
    for name, refs in views.items():
        ids = sorted(refs)
        rows = [indices[utt] for utt in ids]
        results.append(
            evaluate_set(
                args.output_dir,
                name,
                labels,
                ids,
                logits[rows],
                probs[rows],
                refs,
                seen_pairs,
                head,
                args.thresholds,
                args.logit_thresholds,
            )
        )
    if not args.no_plots:
        plot_results(args.output_dir, results)
    write_json(
        args.output_dir / "evaluation.json",
        dict(
            scores_file=str(scores_file),
            metadata=metadata,
            exclusions=excluded,
            reference_sha256={name: digest_value(refs) for name, refs in views.items()},
            sets={
                r["set"]: dict(
                    n=r["num_ref"],
                    exact=r["exact"],
                    accuracy=r["accuracy"],
                    metric=r["metric"],
                )
                for r in results
            },
            aggregate_subsets=list(CS_SUBSETS)
            if "test_cs_all" in views and all(x in views for x in CS_SUBSETS)
            else [],
        ),
    )
    return results


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_file",
        type=Path,
        required=True,
        help="explicit checkpoint; no best/latest inference",
    )
    parser.add_argument(
        "--config_file", type=Path, required=True, help="saved experiment config.yaml"
    )
    parser.add_argument(
        "--train_data_dir",
        type=Path,
        required=True,
        help="checkpoint-matching frozen lang2utt and utt2langs",
    )
    parser.add_argument("--test_sets", nargs="+", required=True)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="formatted set directories with wav.scp and utt2langs",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--trusted_checkpoint",
        action="store_true",
        help="allow unsafe pickle fallback only for a personally trusted checkpoint; can execute code",
    )
    parser.add_argument(
        "--exclusions", type=Path, help="global excluded IDs with TSV header utt/reason"
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        help="probability sweep grid; never fitted",
    )
    parser.add_argument(
        "--logit_thresholds", nargs="+", type=float, help="KL raw-logit sweep grid"
    )
    parser.add_argument(
        "--score_only",
        action="store_true",
        help="require existing scores.npz; no model import/forward",
    )
    parser.add_argument("--no_plots", action="store_true")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Resolve ESPnet from this checkout, not another editable installation.
    sys.path.insert(0, str(RECIPE.parents[1]))
    args = get_parser().parse_args()
    run(args)
    print(args.output_dir / "evaluation.json")


if __name__ == "__main__":
    main()
