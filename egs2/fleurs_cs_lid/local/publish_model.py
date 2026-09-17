#!/usr/bin/env python3
"""Prepare one model-only publication offline; push only to a confirmed HF repo."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import stat

import yaml


ARCHITECTURE = set(
    (
        "input_size frontend frontend_conf specaug specaug_conf normalize normalize_conf "
        "preencoder preencoder_conf encoder encoder_conf postencoder postencoder_conf "
        "decoder decoder_conf ctc_conf joint_net_conf pooling pooling_conf projector "
        "projector_conf loss loss_conf model model_conf lang_num freeze_param use_adapter "
        "adapter adapter_conf token_type cleaner g2p use_preprocessor preprocessor "
        "preprocessor_conf speech_volume_normalize use_lang_prompt use_nlp_prompt"
    ).split()
)
LABEL = re.compile(r"[a-z]{3}(?:-[a-z]{3})?")
ASR_LABEL = re.compile(r"<[a-z]{3}>")
ASR_SPECIALS = {"<blank>", "<unk>", "<sos/eos>"}
EPOCH = re.compile(r"([1-9][0-9]*)epoch\.pth")
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}")
MMS = "facebook/mms-1b"


def regular_file(path):
    path = Path(path).resolve(strict=True)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"not a regular file: {path}")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def inspect_weights(path):
    """Inspect the mmap, never instantiate a backbone or materialize a second copy."""
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as error:
        raise ValueError("CPU mmap load failed; no trusted-pickle fallback") from error
    if not isinstance(state, dict) or not state:
        raise ValueError("require a nonempty model-only state_dict")
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(
                "require a model-only state_dict, not a trainer checkpoint"
            )
        if tensor.device.type != "cpu":
            raise ValueError("non-CPU tensor in checkpoint")


def inventory(path):
    labels = []
    for line in regular_file(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not LABEL.fullmatch(fields[0]):
            raise ValueError("lang2utt must contain language/atomic-pair class rows")
        labels.append(fields[0])
    canonical = [tuple(sorted(label.split("-"))) for label in labels]
    if (
        not labels
        or len(set(canonical)) != len(labels)
        or any(len(set(label)) != len(label) for label in canonical)
    ):
        raise ValueError("empty or duplicate lang2utt classes")
    return labels


def public_language_sets(path, labels):
    """Accept an explicitly frozen labels-only list, never infer pairs from heads."""
    vocabulary = {lang for label in labels for lang in label.split("-")}
    sets = set()
    for line in regular_file(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not LABEL.fullmatch("-".join(fields)) or len(set(fields)) != len(fields):
            raise ValueError("train_language_sets requires 1 or 2 languages, no IDs")
        if not set(fields) <= vocabulary:
            raise ValueError("training language set outside head inventory")
        sets.add(tuple(sorted(fields)))
    if not sets:
        raise ValueError("empty train_language_sets")
    return [list(languages) for languages in sorted(sets)]


def check_portable(value, keys=()):
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("config keys must be strings")
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if normalized.endswith("token") or any(
                word in normalized
                for word in (
                    "password",
                    "passwd",
                    "secret",
                    "credential",
                    "apikey",
                    "privatekey",
                    "authorization",
                )
            ):
                raise ValueError(f"sensitive config key is not publishable: {key}")
            if child is not None and (key.endswith(("_file", "_dir", "_path", "_scp"))):
                raise ValueError(f"unsupported external config dependency: {key}")
            if key == "path_or_url" and child != MMS:
                raise ValueError("only facebook/mms-1b path_or_url is supported")
            check_portable(child, keys + (key,))
    elif isinstance(value, (list, tuple)):
        for child in value:
            check_portable(child, keys)
    elif isinstance(value, str) and (
        "/" in value or "\\" in value or value.startswith("~")
    ):
        if keys == ("frontend_conf", "frontend_conf", "path_or_url") and value == MMS:
            return
        raise ValueError(f"unsupported source path in config: {'.'.join(keys)}")


def inline_list(value, recipe_dir=None):
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_absolute():
            if recipe_dir is None:
                raise ValueError("configured relative assets require --recipe_dir")
            path = Path(recipe_dir) / path
        value = regular_file(path).read_text("utf-8").splitlines()
    if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
        raise ValueError("expected an inline list or a readable vocabulary file")
    if len(set(value)) != len(value):
        raise ValueError("duplicate vocabulary entries")
    return value


def portable_config(
    source, task, config_path, labels, non_linguistic_symbols_file=None, recipe_dir=None
):
    config = copy.deepcopy({k: v for k, v in source.items() if k in ARCHITECTURE})
    config["init"] = None
    frontend = config.get("frontend_conf", {})
    frontend.pop("download_dir", None)
    preprocessing = config.get("preprocessor_conf", {})
    for key in list(preprocessing):
        if key.startswith(("noise_", "rir_")):
            del preprocessing[key]
    model = config.setdefault("model_conf", {})
    for legacy, current in (
        ("lidseq_order_insensitive_loss", "pit_loss"),
        ("lidseq_order_insensitive_reduction", "pit_loss_reduction"),
    ):
        if legacy in model:
            if current in model and model[current] != model[legacy]:
                raise ValueError(f"conflicting {legacy} and {current}")
            model[current] = model.pop(legacy)
    check_portable(config)
    if task == "lid":
        if source.get("lang_num") != len(labels):
            raise ValueError("lang_num differs from actual ordered lang2utt inventory")
        config["lang2utt"] = "lang2utt"
    else:
        tokens = inline_list(source.get("token_list"), recipe_dir)
        if not tokens:
            raise ValueError("ASR requires a nonempty token_list")
        if any(not ASR_LABEL.fullmatch(t) and t not in ASR_SPECIALS for t in tokens):
            raise ValueError(
                "LID-seq token_list allows only <xxx> languages and ESPnet specials"
            )
        if source.get("bpemodel") or source.get("token_type") == "bpe":
            raise ValueError("BPE assets unsupported; use recipe word/char ASR")
        symbols = source.get("non_linguistic_symbols")
        if non_linguistic_symbols_file is not None:
            symbols = regular_file(non_linguistic_symbols_file)
        if symbols is not None:
            symbols = inline_list(symbols, recipe_dir)
            if any(
                not ASR_LABEL.fullmatch(s) or s in ASR_SPECIALS for s in symbols
            ) or not set(symbols) <= set(tokens):
                raise ValueError(
                    "non_linguistic_symbols must be canonical languages and exactly match token_list entries"
                )
        config.update(token_list=tokens, bpemodel=None, non_linguistic_symbols=symbols)
    return config


def validate_destination(args):
    if args.repo_id and not re.fullmatch(
        r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*", args.repo_id
    ):
        raise ValueError("repo_id must be an explicit owner/repository")
    if args.push and (not args.repo_id or args.confirm_repo_id != args.repo_id):
        raise ValueError("--push requires --repo_id and matching --confirm_repo_id")


def prepare(args):
    validate_destination(args)
    if args.recipe_dir is not None and not args.recipe_dir.is_dir():
        raise ValueError("--recipe_dir must be the original recipe directory")
    if not MODEL_ID.fullmatch(args.model_id):
        raise ValueError("model_id must be a simple identifier, without paths")
    if args.task != "lid" and (args.lang2utt or args.train_language_sets):
        raise ValueError("lang2utt/train_language_sets are LID-only")
    if args.task != "asr" and args.non_linguistic_symbols_file:
        raise ValueError("non_linguistic_symbols_file is ASR-only")
    model = regular_file(args.model_file)
    match = EPOCH.fullmatch(model.name)
    epoch = int(match[1]) if match else None
    if args.checkpoint_kind == "averaged":
        if (
            not re.fullmatch(r"valid\.[\w.-]*ave[\w.-]*\.pth", model.name)
            or not args.checkpoint_description
        ):
            raise ValueError(
                "averaged selection requires valid.*ave*.pth and --checkpoint_description"
            )
    elif epoch is None:
        raise ValueError(
            "select an exact numbered checkpoint such as 15epoch.pth; averages require --checkpoint_kind averaged; no best/last fallback or checkpoint.pth"
        )
    before = identity(model)
    inspect_weights(model)
    config_path = regular_file(args.config_file)
    raw_config = config_path.read_bytes()
    source = yaml.safe_load(raw_config)
    if not isinstance(source, dict):
        raise ValueError("config must be a YAML mapping")
    labels = None
    if args.task == "lid":
        if not args.lang2utt:
            raise ValueError("LID requires actual training --lang2utt")
        labels = inventory(args.lang2utt)
        frozen = config_path.parent / "lang2utt"
        references = [frozen] if frozen.is_file() else []
        if source.get("lang2utt"):
            configured = Path(source["lang2utt"])
            if configured.is_absolute() or args.recipe_dir is not None:
                configured = (
                    configured
                    if configured.is_absolute()
                    else args.recipe_dir / configured
                )
                if configured.is_file():
                    references.append(configured)
        if not references:
            raise ValueError(
                "cannot verify training inventory: need frozen experiment lang2utt or original configured path (--recipe_dir for relative paths)"
            )
        if any(inventory(path) != labels for path in references):
            raise ValueError("lang2utt order differs from original training inventory")
    config = portable_config(
        source,
        args.task,
        config_path,
        labels,
        args.non_linguistic_symbols_file,
        args.recipe_dir,
    )
    language_sets = (
        public_language_sets(args.train_language_sets, labels)
        if args.train_language_sets
        else None
    )
    digest = sha256(model)
    if identity(model) != before:
        raise ValueError("checkpoint changed during preparation")
    metadata = {
        "schema_version": 1,
        "model_id": args.model_id,
        "task": args.task,
        "checkpoint": {
            "filename": model.name,
            "epoch": epoch,
            "model_type": args.checkpoint_kind,
            "selection": f"explicit_{args.checkpoint_kind}",
            "description": args.checkpoint_description,
            "sha256": digest,
            "bytes": before[2],
        },
        "source_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "license": "cc-by-nc-4.0",
    }
    artifacts = {"config.yaml": yaml.safe_dump(config, sort_keys=False)}
    if labels:
        metadata["labels"] = labels
        metadata["source_lang2utt_sha256"] = sha256(regular_file(args.lang2utt))
        # A placeholder satisfies upstream line parsing, but is never a train ref.
        artifacts["lang2utt"] = "".join(
            f"{label} __inventory_only__\n" for label in labels
        )
    if language_sets is not None:
        metadata["train_language_sets"] = language_sets
        metadata["source_train_language_sets_sha256"] = sha256(
            regular_file(args.train_language_sets)
        )
        artifacts["utt2langs"] = "".join(
            f"__train_set_{i}__ {' '.join(languages)}\n"
            for i, languages in enumerate(language_sets)
        )
    artifacts["provenance.json"] = json.dumps(metadata, indent=2) + "\n"
    header = {
        "license": "cc-by-nc-4.0",
        "library_name": "espnet",
        "pipeline_tag": "automatic-speech-recognition"
        if args.task == "asr"
        else "audio-classification",
    }
    if (
        config.get("frontend_conf", {}).get("frontend_conf", {}).get("path_or_url")
        == MMS
    ):
        header["base_model"] = MMS
    artifacts["README.md"] = (
        "---\n" + yaml.safe_dump(header, sort_keys=False) + "---\n\n"
        f"# {args.model_id}\n\n"
        f"ESPnet FLEURS-CS {args.task.upper()}, explicitly selected {args.checkpoint_kind} "
        f"(`{model.name}`). This is not a claim of best-checkpoint performance.\n\n"
        "License: CC-BY-NC-4.0 (including the MMS-1B backbone where used). "
        "No evaluation metrics are asserted. See `provenance.json` for hashes.\n\n"
        "No optimizer, audio or real utterance IDs. Use matching ESPnet CS-LID code "
        "(canonical PIT keys); MMS inference still requires its backbone/cache.\n\n"
        "LID: resolve `lang2utt` relative to the bundle for upstream preprocessing. "
        "Its placeholders are not training references. Optional `utt2langs` contains "
        "explicitly supplied frozen training language sets with synthetic IDs: valid "
        "for seen/unseen membership, never for utterance counts. Without that file, "
        "supply actual frozen training references separately.\n"
    )
    architecture_keys = (
        "frontend frontend_conf encoder encoder_conf model_conf freeze_param"
    )
    architecture_keys += (
        " decoder decoder_conf"
        if args.task == "asr"
        else " pooling pooling_conf projector projector_conf loss loss_conf lang_num"
    )
    architecture = {
        key: config[key] for key in architecture_keys.split() if key in config
    }
    artifacts["README.md"] += (
        "\n## Architecture\n\nSelected settings from the validated portable config:\n\n"
        "```yaml\n" + yaml.safe_dump(architecture, sort_keys=False) + "```\n"
    )
    if args.checkpoint_description:
        artifacts["README.md"] += (
            "\n## Checkpoint Description\n\n" + args.checkpoint_description + "\n"
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in artifacts.items():
        (output / name).write_text(content, encoding="utf-8")
    plan = {
        "schema_version": 1,
        "model_id": args.model_id,
        "repo_id": args.repo_id,
        "private": not args.public,
        "model_file": str(model),
        "model_sha256": digest,
        "model_identity": list(before),
        "artifacts": {name: sha256(output / name) for name in artifacts},
    }
    (output / "upload-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    return output


def verify(output):
    output = Path(output)
    plan = json.loads((output / "upload-plan.json").read_text(encoding="utf-8"))
    model = regular_file(plan["model_file"])
    if (
        str(model) != plan["model_file"]
        or list(identity(model)) != plan["model_identity"]
        or sha256(model) != plan["model_sha256"]
    ):
        raise ValueError("selected checkpoint changed since preparation")
    inspect_weights(model)
    expected = {"config.yaml", "provenance.json", "README.md", "lang2utt", "utt2langs"}
    if (
        not {"config.yaml", "provenance.json", "README.md"} <= plan["artifacts"].keys()
        or not plan["artifacts"].keys() <= expected
    ):
        raise ValueError("unexpected publication artifact list")
    for name, digest in plan["artifacts"].items():
        if (output / name).is_symlink() or sha256(
            regular_file(output / name)
        ) != digest:
            raise ValueError(f"publication artifact changed: {name}")
    return plan


def push(output, args):
    validate_destination(args)
    if not args.push:
        raise ValueError("upload requires explicit --push")
    plan = verify(output)
    if plan["repo_id"] != args.repo_id or plan["private"] != (not args.public):
        raise ValueError(
            "destination/visibility differs from prepared plan; prepare a new output"
        )
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi(endpoint="https://huggingface.co")
    try:
        info = api.repo_info(args.repo_id, repo_type="model")
    except RepositoryNotFoundError:
        api.create_repo(
            args.repo_id, repo_type="model", private=plan["private"], exist_ok=False
        )
        info = api.repo_info(args.repo_id, repo_type="model")
    if info.private != plan["private"]:
        raise ValueError(
            "existing HF repository visibility differs; never changing visibility"
        )
    operations = [
        CommitOperationAdd(
            path_in_repo=Path(plan["model_file"]).name,
            path_or_fileobj=plan["model_file"],
        )
    ] + [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(Path(output) / name))
        for name in plan["artifacts"]
    ]
    if list(identity(plan["model_file"])) != plan["model_identity"]:
        raise ValueError("selected checkpoint changed before upload")
    return api.create_commit(
        args.repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"Publish model-only {plan['model_id']}",
        parent_commit=info.sha,
    )


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="offline metadata preparation (default)")
    p.add_argument("--task", choices=("asr", "lid"), required=True)
    for flag in ("config_file", "model_file", "output_dir"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--model_id", required=True)
    p.add_argument(
        "--recipe_dir",
        type=Path,
        help="original recipe directory for configured relative assets",
    )
    p.add_argument("--checkpoint_kind", choices=("epoch", "averaged"), default="epoch")
    p.add_argument("--checkpoint_description", help="required for averaged weights")
    p.add_argument(
        "--lang2utt", type=Path, help="actual ordered training inventory (LID required)"
    )
    p.add_argument(
        "--train_language_sets", type=Path, help="frozen languages-only rows, no IDs"
    )
    p.add_argument(
        "--non_linguistic_symbols_file",
        "--non_linguistic_symbols",
        type=Path,
        help="explicit ASR symbols file to inline",
    )
    p.add_argument("--repo_id", help="owner/repository; optional until confirmed")
    p.add_argument(
        "--public", action="store_true", help="public visibility; default private"
    )
    p.add_argument(
        "--push", action="store_true", help="enable network upload; default offline"
    )
    p.add_argument("--confirm_repo_id", help="must equal --repo_id for --push")
    v = sub.add_parser(
        "verify", help="offline CPU mmap/hash check, no model construction"
    )
    v.add_argument("--output_dir", type=Path, required=True)
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    if args.command == "verify":
        verify(args.output_dir)
        print(
            "Verified saved selection and artifacts offline (not an inference/metrics check)."
        )
    else:
        output = prepare(args)
        if args.push:
            result = push(output, args)
            print(result)
        else:
            print(
                f"Offline preparation complete: {output}/upload-plan.json (no weights copied, no upload)."
            )


if __name__ == "__main__":
    main()
