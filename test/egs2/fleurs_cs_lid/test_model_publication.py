"""Publication tests use synthetic CPU states, never the Hub or training data."""

import hashlib
import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import torch
import yaml


RECIPE = Path(__file__).resolve().parents[3] / "egs2" / "fleurs_cs_lid"


@pytest.fixture
def publisher(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba-cache"))

    def no_network(*args, **kwargs):
        raise AssertionError("publication tests must never use network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(torch.cuda, "init", no_network)
    spec = importlib.util.spec_from_file_location(
        "recipe_model_publication", RECIPE / "local" / "publish_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(tmp_path, publisher, task="lid", **overrides):
    source = tmp_path / "experiment"
    source.mkdir()
    config = {
        "frontend": "s3prl",
        "frontend_conf": {
            "frontend_conf": {
                "upstream": "hf_wav2vec2_custom",
                "path_or_url": "facebook/mms-1b",
            },
            "download_dir": "/private/cache",
            "multilayer_feature": True,
        },
        "encoder": "ecapa_tdnn",
        "encoder_conf": {"output_size": 16},
        "model_conf": {},
        "freeze_param": ["frontend.upstream"],
        "lang_num": 3,
        "lang2utt": str(source / "lang2utt"),
        "token_list": ["<blank>", "<unk>", "<ara>", "<eng>", "<sos/eos>"],
        "token_type": "word",
        "preprocessor_conf": {
            "noise_info": [[1, "/private/noise.scp", [1, 1], [0, 1]]],
            "sample_rate": 16000,
        },
        "train_data_path_and_name_and_type": [
            ["/private/train.wav.scp", "speech", "sound"]
        ],
        "valid_data_path_and_name_and_type": [
            ["/private/valid.wav.scp", "speech", "sound"]
        ],
        "init_param": ["/private/init.pth"],
        "init": "xavier_uniform",
        "output_dir": "/private/experiment",
        "optim": "adam",
        "optim_conf": {"lr": 0.001},
        "ngpu": 8,
        "distributed": True,
    }
    config.update(overrides)
    (source / "config.yaml").write_text(yaml.safe_dump(config))
    (source / "lang2utt").write_text(
        "jpn private-utt-3\nara private-utt-1 private-utt-4\neng private-utt-2 private-utt-4\n"
    )
    torch.save(torch.nn.Linear(4, 3).state_dict(), source / "15epoch.pth")
    argv = [
        "prepare",
        "--task",
        task,
        "--config_file",
        str(source / "config.yaml"),
        "--model_file",
        str(source / "15epoch.pth"),
        "--model_id",
        "test-model",
        "--output_dir",
        str(tmp_path / "publication"),
    ]
    if task == "lid":
        argv += ["--lang2utt", str(source / "lang2utt")]
    return publisher.get_parser().parse_args(argv)


def test_offline_small_publication_no_ids_or_weights_copy(tmp_path, publisher, capsys):
    args = inputs(tmp_path, publisher)
    original = args.config_file.read_bytes()
    before = args.model_file.stat()
    output = publisher.prepare(args)
    assert set(p.name for p in output.iterdir()) == {
        "config.yaml",
        "README.md",
        "provenance.json",
        "lang2utt",
        "upload-plan.json",
    }
    assert sum(p.stat().st_size for p in output.iterdir()) < 20_000
    assert args.config_file.read_bytes() == original
    assert args.model_file.stat() == before
    config = yaml.safe_load((output / "config.yaml").read_text())
    assert config["freeze_param"] == ["frontend.upstream"]
    assert config["encoder_conf"] == {"output_size": 16}
    assert config["init"] is None
    assert config["lang2utt"] == "lang2utt"
    assert "download_dir" not in config["frontend_conf"]
    assert config["preprocessor_conf"] == {"sample_rate": 16000}
    assert "train_data_path_and_name_and_type" not in config
    assert "optim_conf" not in config
    assert "ngpu" not in config
    plan = publisher.verify(output)
    assert plan["private"] is True and plan["repo_id"] is None
    assert "upload-plan.json" not in plan["artifacts"]
    public_text = "".join((output / name).read_text() for name in plan["artifacts"])
    assert "/private" not in public_text
    assert str(args.config_file.parent) not in public_text
    assert "private-utt-" not in public_text
    assert "base_model: facebook/mms-1b" in public_text
    assert "license: cc-by-nc-4.0" in public_text
    metadata = json.loads((output / "provenance.json").read_text())
    assert metadata["labels"] == ["jpn", "ara", "eng"]
    assert metadata["checkpoint"]["epoch"] == 15
    assert metadata["checkpoint"]["selection"] == "explicit_epoch"
    assert (
        metadata["checkpoint"]["sha256"]
        == hashlib.sha256(args.model_file.read_bytes()).hexdigest()
    )
    assert metadata["source_config_sha256"] == hashlib.sha256(original).hexdigest()
    assert "train_language_sets" not in metadata
    publisher.main(["verify", "--output_dir", str(output)])
    assert "not an inference/metrics check" in capsys.readouterr().out


@pytest.mark.parametrize("task", ["asr", "lid"])
def test_model_card_uses_only_validated_architecture_and_operator_description(
    tmp_path, publisher, task
):
    args = inputs(
        tmp_path,
        publisher,
        task,
        decoder="transformer",
        decoder_conf={"num_blocks": 4},
        pooling="chn_attn_stat",
        pooling_conf={},
        projector="rawnet3",
        projector_conf={"output_size": 192},
        loss="arc_margin_subcenter_intertopk_multilabel_bce",
        loss_conf={"scale": 30, "pos_weight": 50},
        model_conf={"lidseq_order_insensitive_loss": True} if task == "asr" else {},
        optim_conf={"password": "unpublished-optimizer-secret"},
    )
    args.checkpoint_description = (
        "Operator-selected checkpoint; no best-performance claim."
    )
    output = publisher.prepare(args)
    card = (output / "README.md").read_text()
    summary = yaml.safe_load(card.split("```yaml\n", 1)[1].split("```", 1)[0])
    config = yaml.safe_load((output / "config.yaml").read_text())
    assert all(value == config[key] for key, value in summary.items())
    assert summary["frontend_conf"]["frontend_conf"]["path_or_url"] == "facebook/mms-1b"
    assert summary["frontend_conf"]["multilayer_feature"] is True
    assert summary["encoder"] == "ecapa_tdnn"
    assert summary["encoder_conf"] == {"output_size": 16}
    assert summary["freeze_param"] == ["frontend.upstream"]
    if task == "asr":
        assert summary["decoder_conf"] == {"num_blocks": 4}
        assert summary["model_conf"] == {"pit_loss": True}
        assert "pooling" not in summary and "loss_conf" not in summary
    else:
        assert summary["pooling"] == "chn_attn_stat"
        assert summary["pooling_conf"] == {}
        assert summary["projector"] == "rawnet3"
        assert summary["projector_conf"] == {"output_size": 192}
        assert summary["loss"] == config["loss"]
        assert summary["loss_conf"] == {"scale": 30, "pos_weight": 50}
        assert "decoder" not in summary
    assert args.checkpoint_description in card
    for excluded in (
        "unpublished-optimizer-secret",
        "/private",
        "private-utt-",
        "download_dir",
        "init_param",
        "lidseq_order_insensitive_loss",
    ):
        assert excluded not in card
    assert str(args.config_file.parent) not in card


def test_safe_cpu_mmap_and_no_unsafe_retry(tmp_path, publisher, monkeypatch):
    args = inputs(tmp_path, publisher)
    original = torch.load
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    publisher.prepare(args)
    assert calls == [{"map_location": "cpu", "weights_only": True, "mmap": True}]
    calls.clear()

    def refuse(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("unsafe pickle")

    monkeypatch.setattr(torch, "load", refuse)
    with pytest.raises(ValueError, match="no trusted-pickle fallback"):
        publisher.inspect_weights(args.model_file)
    assert calls == [{"map_location": "cpu", "weights_only": True, "mmap": True}]


@pytest.mark.parametrize(
    "kind", ["trainer", "checkpoint", "directory", "unversioned", "legacy_format"]
)
def test_reject_non_model_numbered_checkpoint(tmp_path, publisher, kind):
    args = inputs(tmp_path, publisher)
    if kind == "trainer":
        torch.save(
            {
                "model": torch.nn.Linear(4, 3).state_dict(),
                "optimizers": [],
                "reporter": {},
            },
            args.model_file,
        )
    elif kind == "checkpoint":
        args.model_file = args.model_file.with_name("checkpoint.pth")
        torch.save(torch.nn.Linear(4, 3).state_dict(), args.model_file)
    elif kind == "directory":
        args.model_file = args.model_file.with_name("16epoch.pth")
        args.model_file.mkdir()
    elif kind == "legacy_format":
        torch.save(
            torch.nn.Linear(4, 3).state_dict(),
            args.model_file,
            _use_new_zipfile_serialization=False,
        )
    else:
        new = args.model_file.with_name("valid.best.pth")
        args.model_file.rename(new)
        args.model_file = new
    with pytest.raises(ValueError):
        publisher.prepare(args)
    assert not args.output_dir.exists()


def test_symlink_selection_is_frozen_and_changes_detected(tmp_path, publisher):
    args = inputs(tmp_path, publisher)
    original = args.model_file
    alias = original.with_name("valid.best.pth")
    alias.symlink_to(original.name)
    args.model_file = alias
    output = publisher.prepare(args)
    other = original.with_name("16epoch.pth")
    torch.save(torch.nn.Linear(4, 3).state_dict(), other)
    alias.unlink()
    alias.symlink_to(other.name)
    assert publisher.verify(output)["model_file"] == str(original)
    torch.save(torch.nn.Linear(4, 3).state_dict(), original)
    with pytest.raises(ValueError, match="checkpoint changed"):
        publisher.verify(output)


@pytest.mark.parametrize(
    "kind", ["missing", "count", "duplicate", "snapshot", "configured"]
)
def test_lid_inventory_required_and_ordered(tmp_path, publisher, kind):
    args = inputs(tmp_path, publisher)
    if kind == "missing":
        args.lang2utt = None
    elif kind == "count":
        args.lang2utt.write_text("ara secret\neng secret\n")
    elif kind == "duplicate":
        args.lang2utt.write_text("ara-eng secret\neng-ara secret\neng secret\n")
    else:
        args.lang2utt = tmp_path / "other_inventory"
        args.lang2utt.write_text("ara secret\neng secret\njpn secret\n")
        if kind == "configured":
            config = yaml.safe_load(args.config_file.read_text())
            frozen = args.config_file.parent / "lang2utt"
            moved = frozen.with_name("historical-lang2utt")
            frozen.rename(moved)
            config["lang2utt"] = str(moved)
            args.config_file.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        publisher.prepare(args)


@pytest.mark.parametrize("reordered", [False, True])
def test_recipe_relative_training_inventory_requires_verified_order(
    tmp_path, publisher, reordered
):
    args = inputs(tmp_path, publisher, lang2utt="data/train/lang2utt")
    frozen = args.config_file.parent / "lang2utt"
    original = frozen.read_text()
    frozen.unlink()
    args.lang2utt = tmp_path / "supplied-lang2utt"
    args.lang2utt.write_text(
        "ara private1\neng private2\njpn private3\n" if reordered else original
    )
    # A matching decoy under the experiment directory must never be trusted.
    decoy = args.config_file.parent / "data" / "train" / "lang2utt"
    decoy.parent.mkdir(parents=True)
    decoy.write_text(args.lang2utt.read_text())
    recipe_dir = tmp_path / "original-recipe"
    configured = recipe_dir / "data" / "train" / "lang2utt"
    configured.parent.mkdir(parents=True)
    configured.write_text(original)
    with pytest.raises(ValueError, match="cannot verify training inventory"):
        publisher.prepare(args)
    assert not args.output_dir.exists()
    args.recipe_dir = recipe_dir
    if reordered:
        with pytest.raises(ValueError, match="order differs"):
            publisher.prepare(args)
        assert not args.output_dir.exists()
    else:
        output = publisher.prepare(args)
        assert publisher.inventory(output / "lang2utt") == ["jpn", "ara", "eng"]


@pytest.mark.parametrize("configured", [None, "/missing/training/lang2utt"])
def test_unverifiable_inventory_cannot_use_supplied_file_alone(
    tmp_path, publisher, configured
):
    args = inputs(tmp_path, publisher, lang2utt=configured)
    moved = tmp_path / "operator-supplied"
    args.lang2utt.rename(moved)
    args.lang2utt = moved
    with pytest.raises(ValueError, match="cannot verify training inventory"):
        publisher.prepare(args)
    assert not args.output_dir.exists()


def test_frozen_snapshot_suffices_when_relative_source_unavailable(tmp_path, publisher):
    args = inputs(tmp_path, publisher, lang2utt="data/historical/lang2utt")
    output = publisher.prepare(args)
    assert publisher.inventory(output / "lang2utt") == ["jpn", "ara", "eng"]


def test_actual_sets_not_inferred_from_class_inventory(tmp_path, publisher):
    args = inputs(tmp_path, publisher)
    args.train_language_sets = tmp_path / "train-language-sets.txt"
    args.train_language_sets.write_text("eng ara\nara eng\njpn\n")
    output = publisher.prepare(args)
    metadata = json.loads((output / "provenance.json").read_text())
    assert metadata["train_language_sets"] == [["ara", "eng"], ["jpn"]]
    assert (
        output / "utt2langs"
    ).read_text() == "__train_set_0__ ara eng\n__train_set_1__ jpn\n"
    assert "utt2langs" in publisher.verify(output)["artifacts"]
    args.train_language_sets.write_text("secret_utt ara eng\n")
    with pytest.raises(ValueError, match="no IDs"):
        publisher.public_language_sets(args.train_language_sets, metadata["labels"])


def test_explicit_averaged_state_never_invents_epochs(tmp_path, publisher):
    args = inputs(tmp_path, publisher)
    averaged = args.model_file.with_name("valid.accuracy.ave.pth")
    args.model_file.rename(averaged)
    args.model_file = averaged
    with pytest.raises(ValueError, match="averaged"):
        publisher.prepare(args)
    args.checkpoint_kind = "averaged"
    with pytest.raises(ValueError, match="checkpoint_description"):
        publisher.prepare(args)
    args.checkpoint_description = (
        "Existing evaluated average; component epochs not established."
    )
    output = publisher.prepare(args)
    selection = json.loads((output / "provenance.json").read_text())["checkpoint"]
    assert selection["model_type"] == "averaged" and selection["epoch"] is None
    assert selection["selection"] == "explicit_averaged"
    assert selection["filename"] == averaged.name
    assert "component_epochs" not in selection
    assert publisher.verify(output)["model_file"] == str(averaged)


def test_literal_null_bpe_and_inline_symbols_supported(tmp_path, publisher):
    args = inputs(
        tmp_path,
        publisher,
        "asr",
        bpemodel=None,
        non_linguistic_symbols=["<ara>", "<eng>"],
    )
    output = publisher.prepare(args)
    config = yaml.safe_load((output / "config.yaml").read_text())
    assert config["bpemodel"] is None
    assert config["non_linguistic_symbols"] == ["<ara>", "<eng>"]


def test_explicit_symbols_preserved_and_bracket_raw_mismatch_fatal(tmp_path, publisher):
    args = inputs(
        tmp_path,
        publisher,
        "asr",
        non_linguistic_symbols="data/nlsyms.txt",
        token_list=["<blank>", "<unk>", "<ara>", "<eng>", "<sos/eos>"],
    )
    args.non_linguistic_symbols_file = tmp_path / "actual-nlsyms.txt"
    args.non_linguistic_symbols_file.write_text("<eng>\n<ara>\n")
    output = publisher.prepare(args)
    config = yaml.safe_load((output / "config.yaml").read_text())
    assert config["non_linguistic_symbols"] == ["<eng>", "<ara>"]
    assert "data/nlsyms.txt" not in (output / "config.yaml").read_text()
    args.non_linguistic_symbols_file.write_text("eng\nara\n")
    args.output_dir = tmp_path / "mismatched"
    with pytest.raises(ValueError, match="exactly match token_list"):
        publisher.prepare(args)


@pytest.mark.parametrize("tokens_from_file", [False, True])
def test_asr_tokens_freeze_and_pit_aliases(tmp_path, publisher, tokens_from_file):
    tokens = ["<blank>", "<unk>", "<ara>", "<eng>", "<sos/eos>"]
    args = inputs(
        tmp_path,
        publisher,
        "asr",
        model_conf={
            "ctc_weight": 0,
            "lidseq_order_insensitive_loss": True,
            "lidseq_order_insensitive_reduction": "min",
        },
        freeze_param=[],
    )
    if tokens_from_file:
        (args.config_file.parent / "tokens.txt").write_text("\n".join(tokens) + "\n")
        config = yaml.safe_load(args.config_file.read_text())
        config["token_list"] = "tokens.txt"
        args.config_file.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="require --recipe_dir"):
            publisher.prepare(args)
        args.recipe_dir = args.config_file.parent
    output = publisher.prepare(args)
    config = yaml.safe_load((output / "config.yaml").read_text())
    assert config["token_list"] == tokens
    assert config["freeze_param"] == []
    assert config["model_conf"] == {
        "ctc_weight": 0,
        "pit_loss": True,
        "pit_loss_reduction": "min",
    }
    assert "lang2utt" not in config


@pytest.mark.parametrize("field", ["token_list", "non_linguistic_symbols"])
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize(
    "entry", ["hf_sensitive_value", "private-utterance-123", "ara", "<ara-eng>"]
)
def test_noncanonical_asr_assets_never_published(
    tmp_path, publisher, field, external, entry
):
    args = inputs(tmp_path, publisher, "asr")
    config = yaml.safe_load(args.config_file.read_text())
    if external:
        path = tmp_path / "external.txt"
        path.write_text(entry + "\n")
        config[field] = str(path)
    else:
        config[field] = [entry]
    args.config_file.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="token_list|canonical languages"):
        publisher.prepare(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("special", ["<blank>", "<unk>", "<sos/eos>"])
def test_asr_specials_cannot_be_non_linguistic_language_symbols(
    tmp_path, publisher, special
):
    args = inputs(tmp_path, publisher, "asr", non_linguistic_symbols=[special])
    with pytest.raises(ValueError, match="canonical languages"):
        publisher.prepare(args)


@pytest.mark.parametrize(
    "key",
    [
        "token",
        "hf_token",
        "accessToken",
        "password",
        "client_secret",
        "credentials",
        "api-key",
        "private_key",
        "Authorization",
    ],
)
def test_sensitive_nested_config_keys_rejected(tmp_path, publisher, key):
    args = inputs(
        tmp_path,
        publisher,
        "asr",
        encoder_conf={"nested": [{key: "must-not-be-published"}]},
    )
    with pytest.raises(ValueError, match="sensitive config key"):
        publisher.prepare(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    "config",
    [
        {"model_conf": {"pit_loss": False, "lidseq_order_insensitive_loss": True}},
        {
            "model_conf": {
                "pit_loss_reduction": "min",
                "lidseq_order_insensitive_reduction": "mean",
            }
        },
        {"normalize_conf": {"stats_file": "/private/stats.npz"}},
        {"bpemodel": "/private/bpe.model"},
        {"frontend_conf": {"frontend_conf": {"path_or_url": "local.pth"}}},
    ],
)
def test_refuse_conflicting_aliases_and_external_dependencies(
    tmp_path, publisher, config
):
    args = inputs(tmp_path, publisher, "asr", **config)
    with pytest.raises(ValueError):
        publisher.prepare(args)


def fake_hub(monkeypatch, *, private=True, missing=False):
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import RepositoryNotFoundError

    calls = []

    class API:
        def __init__(self, *, endpoint):
            assert endpoint == "https://huggingface.co"

        def repo_info(self, repo_id, **kwargs):
            calls.append(("info", repo_id, kwargs))
            if missing and not any(call[0] == "create" for call in calls):
                raise RepositoryNotFoundError(
                    "missing test repo",
                    response=httpx.Response(
                        404, request=httpx.Request("GET", "https://example.invalid")
                    ),
                )
            return SimpleNamespace(private=private, sha="test-head")

        def create_repo(self, repo_id, **kwargs):
            calls.append(("create", repo_id, kwargs))

        def create_commit(self, repo_id, **kwargs):
            calls.append(("commit", repo_id, kwargs))
            return "mock-commit"

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    return calls


@pytest.mark.parametrize("missing", [False, True])
def test_explicit_push_uses_original_file_no_plan_upload(
    tmp_path, publisher, monkeypatch, missing
):
    args = inputs(tmp_path, publisher)
    args.repo_id = args.confirm_repo_id = "test-owner/test-repo"
    args.push = True
    calls = fake_hub(monkeypatch, missing=missing)
    output = publisher.prepare(args)
    assert calls == []
    assert publisher.push(output, args) == "mock-commit"
    commit = calls[-1][2]
    assert commit["parent_commit"] == "test-head"
    ops = {op.path_in_repo: op for op in commit["operations"]}
    assert set(ops) == {
        "15epoch.pth",
        "config.yaml",
        "lang2utt",
        "README.md",
        "provenance.json",
    }
    assert ops["15epoch.pth"].path_or_fileobj == str(args.model_file)
    if missing:
        assert next(call[2] for call in calls if call[0] == "create") == {
            "repo_type": "model",
            "private": True,
            "exist_ok": False,
        }


@pytest.mark.parametrize(
    "kind",
    [
        "no_push",
        "no_repo",
        "no_confirmation",
        "wrong_confirmation",
        "visibility",
        "plan_visibility",
        "tamper",
    ],
)
def test_upload_guards(tmp_path, publisher, monkeypatch, kind):
    args = inputs(tmp_path, publisher)
    args.repo_id = args.confirm_repo_id = "test-owner/test-repo"
    calls = fake_hub(monkeypatch, private=kind != "visibility")
    output = publisher.prepare(args)
    args.push = kind != "no_push"
    if kind == "no_repo":
        args.repo_id = None
    elif kind == "no_confirmation":
        args.confirm_repo_id = None
    elif kind == "wrong_confirmation":
        args.confirm_repo_id = "other/repo"
    elif kind == "plan_visibility":
        args.public = True
    elif kind == "tamper":
        (output / "config.yaml").write_text("changed: true\n")
    with pytest.raises(ValueError):
        publisher.push(output, args)
    assert not any(call[0] in {"commit", "create"} for call in calls)
    if kind != "visibility":
        assert calls == []


def tiny_config(task):
    common = dict(
        frontend="default",
        frontend_conf=dict(fs=16000, n_fft=128, hop_length=32, n_mels=8),
        specaug=None,
        normalize=None,
        init=None,
        model_conf={},
        freeze_param=["frontend"],
        use_preprocessor=False,
    )
    if task == "lid":
        return dict(
            common,
            encoder="identity",
            encoder_conf={},
            pooling="mean",
            pooling_conf={},
            projector="rawnet3",
            projector_conf=dict(output_size=4),
            loss="arc_margin_subcenter_intertopk_multilabel_bce",
            loss_conf=dict(K=3, k_top=0),
            lang_num=3,
            lang2utt="/obsolete/train/lang2utt",
        )
    return dict(
        common,
        input_size=4,
        encoder="transformer",
        encoder_conf=dict(
            output_size=8,
            attention_heads=2,
            linear_units=16,
            num_blocks=1,
            input_layer="linear",
        ),
        decoder="transformer",
        decoder_conf=dict(attention_heads=2, linear_units=16, num_blocks=1),
        ctc_conf={},
        token_list=["<blank>", "<unk>", "<ara>", "<eng>", "<sos/eos>"],
        token_type="word",
        model_conf=dict(
            ctc_weight=0,
            lidseq_order_insensitive_loss=True,
            lidseq_order_insensitive_reduction="min",
        ),
    )


@pytest.mark.parametrize("task", ["lid", "asr"])
def test_relocated_native_model_load_without_source_paths(
    tmp_path, publisher, monkeypatch, task
):
    if task == "lid":
        from espnet2.tasks.lid import LIDTask as Task
    else:
        from espnet2.tasks.asr import ASRTask as Task

    args = inputs(tmp_path, publisher, task)
    config = tiny_config(task)
    args.config_file.write_text(yaml.safe_dump(config))
    original, _ = Task.build_model_from_file(args.config_file, None, "cpu")
    torch.save(original.state_dict(), args.model_file)
    output = publisher.prepare(args)
    original_config = args.config_file.read_bytes()
    moved = tmp_path / "elsewhere"
    args.config_file.parent.rename(moved)
    monkeypatch.chdir(tmp_path)
    restored, restored_args = Task.build_model_from_file(
        output / "config.yaml", moved / "15epoch.pth", "cpu"
    )
    restored.load_state_dict(
        torch.load(
            moved / "15epoch.pth", map_location="cpu", weights_only=True, mmap=True
        ),
        strict=True,
    )
    assert (moved / "config.yaml").read_bytes() == original_config
    for key, tensor in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor)
    if task == "asr":
        assert restored.pit_loss and restored.pit_loss_reduction == "min"
        assert restored_args.token_list == config["token_list"]
        from espnet2.bin.asr_inference import Speech2Text

        decoder = Speech2Text(
            asr_train_config=output / "config.yaml",
            asr_model_file=moved / "15epoch.pth",
            device="cpu",
            beam_size=1,
            ctc_weight=0,
            lm_weight=0,
            maxlenratio=0.2,
            token_type="word",
        )
        result = decoder(torch.randn(12, 4))
        assert result and len(result[0]) == 4
    else:
        monkeypatch.syspath_prepend(str(RECIPE / "lid1" / "local"))
        spec = importlib.util.spec_from_file_location(
            "publication_eval", RECIPE / "lid1" / "local" / "evaluate_lid.py"
        )
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        eval_args = SimpleNamespace(
            train_data_dir=moved, config_file=output / "config.yaml"
        )
        # An inventory placeholder is not enough for seen/unseen evaluation.
        with pytest.raises(FileNotFoundError):
            evaluator.training_inventory(eval_args, vars(restored_args))
        (moved / "utt2langs").write_text("private1 ara eng\nprivate2 jpn\n")
        labels, _, _, seen = evaluator.training_inventory(
            eval_args, vars(restored_args)
        )
        assert labels == ["jpn", "ara", "eng"]
        assert {frozenset(pair) for pair in seen} == {frozenset({"ara", "eng"})}
        samples = torch.randn(1, 512)
        original.eval()
        restored.eval()
        with torch.inference_mode():
            head = evaluator.head_spec(vars(restored_args), labels)
            expected = evaluator.model_logits(
                original, samples, torch.tensor([512]), head, 3
            )
            actual = evaluator.model_logits(
                restored, samples, torch.tensor([512]), head, 3
            )
        torch.testing.assert_close(actual, expected)
        assert actual.shape == (1, 3) and torch.isfinite(actual).all()


def test_public_training_sets_work_with_evaluator(tmp_path, publisher, monkeypatch):
    args = inputs(tmp_path, publisher)
    args.train_language_sets = tmp_path / "sets.txt"
    args.train_language_sets.write_text("ara eng\njpn\n")
    output = publisher.prepare(args)
    monkeypatch.syspath_prepend(str(RECIPE / "lid1" / "local"))
    spec = importlib.util.spec_from_file_location(
        "public_sets_eval", RECIPE / "lid1" / "local" / "evaluate_lid.py"
    )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    labels, _, _, seen = evaluator.training_inventory(
        SimpleNamespace(config_file=output / "config.yaml", train_data_dir=output),
        yaml.safe_load((output / "config.yaml").read_text()),
    )
    assert labels == ["jpn", "ara", "eng"]
    assert {frozenset(pair) for pair in seen} == {frozenset({"ara", "eng"})}
