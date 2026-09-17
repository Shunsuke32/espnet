"""Offline shell/config regressions; no ASR model or training is invoked."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "egs2/fleurs_cs_lid/asr1"
STEM = "train_lidseq_mms_transformer24_order_insensitive_min"
PHASES = [
    ("mixed", "frozen", 8, 4, 10, 1, 0.3, 3880, 2910),
    ("mixed", "unfrozen", 8, 4, 30, 3, 0.1, 3880, 2910),
    ("csall", "frozen", 4, 8, 30, 3, 0.1, 8240, 3090),
    ("csall", "unfrozen", 4, 8, 30, 3, 0.1, 8240, 3090),
]


def config_name(mode, phase):
    name = STEM + ("_csall" if mode == "csall" else "")
    return name + ("_unfrozen_lr5e6" if phase == "unfrozen" else "") + ".yaml"


def read_config(mode, phase):
    return yaml.safe_load((RECIPE / "conf" / config_name(mode, phase)).read_text())


@pytest.fixture
def work(tmp_path):
    (tmp_path / "utils").symlink_to(RECIPE / "utils", target_is_directory=True)
    (tmp_path / "local").symlink_to(RECIPE / "local", target_is_directory=True)
    (tmp_path / "conf").mkdir()
    for path in (RECIPE / "conf").glob("*.yaml"):
        (tmp_path / "conf" / path.name).symlink_to(path)
    for name in ("train_lidseq", "train_lidseq_csall_yodas"):
        directory = tmp_path / "data" / name
        directory.mkdir(parents=True)
        (directory / "wav.scp").write_text(
            "".join(f"u{i:03d} /nonexistent/{i}.wav\n" for i in range(32))
        )
    return tmp_path


def invoke(work, *args):
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        WRAPPER_PYTHON=sys.executable,
    )
    # Only the budget generator/validator and inline YAML guards may run Python.
    harness = r'''
forbidden() { printf 'FORBIDDEN: %s\n' "$*" >&2; exit 91; }
function local/data.sh() { forbidden local/data.sh; }
function local/verify_lid_data.sh() { :; }
function local/score_lidseq_decodes.sh() { forbidden local/score_lidseq_decodes.sh; }
function python3() {
    case "$1" in
        -|local/make_fixed_batch_config.py|local/validate_fixed_batch_budget.py)
            "$WRAPPER_PYTHON" "$@" ;;
        *) forbidden python3 "$@" ;;
    esac
}
function ./asr.sh() { printf 'TEMPLATE_ARG=%s\n' "$@"; }
source "$1" "${@:2}"
'''
    before = {p: p.read_bytes() for p in (work / "data").glob("*/wav.scp")}
    result = subprocess.run(
        [
            "bash", "-c", harness, "phase-test", str(RECIPE / "run.sh"),
            "--ngpu", "0", "--stage", "11", "--stop_stage", "11",
            "--asr_tag", "phase_test", "--auto_training_budget", "false",
            "--run_lidseq_scoring", "false", "--cs_root", str(work / "no_cs"),
            "--fleurs_download_dir", str(work / "no_fleurs"), *args,
        ],
        cwd=work, env=env, text=True, capture_output=True, timeout=30,
    )
    assert "FORBIDDEN:" not in result.stderr, result.stderr
    for path, contents in before.items():
        assert path.read_bytes() == contents
    return result


def forwarded(result):
    assert result.returncode == 0, result.stdout + result.stderr
    args = [
        line.removeprefix("TEMPLATE_ARG=") for line in result.stdout.splitlines()
        if line.startswith("TEMPLATE_ARG=")
    ]
    assert args and len(args) % 2 == 0
    return dict(zip(args[::2], args[1::2]))


def refused(result, message):
    assert result.returncode != 0, result.stdout + result.stderr
    assert message in result.stderr, result.stdout + result.stderr
    assert "TEMPLATE_ARG=" not in result.stdout


def weights(work, name="chosen_epoch.pth"):
    path = work / name
    path.write_bytes(b"existence-only fixture: never loaded")
    return str(path)


def saved_phase(work, mode, phase, config=None):
    directory = work / "exp/asr_phase_test"
    directory.mkdir(parents=True)
    config = read_config(mode, phase) if config is None else dict(config)
    config["resume"] = False
    config["output_dir"] = str(directory)
    (directory / "config.yaml").write_text(yaml.safe_dump(config))
    (directory / "checkpoint.pth").write_bytes(b"never deserialize this fixture")
    return directory


@pytest.mark.parametrize("mode,phase,batch,accum,epochs,passes,warmup,iters,steps", PHASES)
def test_phase_configs_and_historical_budget(
    mode, phase, batch, accum, epochs, passes, warmup, iters, steps
):
    cfg = read_config(mode, phase)
    assert cfg["encoder_conf"]["output_size"] == 256
    assert cfg["model_conf"]["pit_loss"] is True
    assert cfg["model_conf"]["pit_loss_reduction"] == "min"
    assert cfg["freeze_param"] == (["frontend.upstream"] if phase == "frozen" else [])
    assert cfg["optim_conf"]["lr"] == (0.001 if phase == "frozen" else 5e-6)
    assert not cfg.get("init_param")
    assert cfg.get("patience") is None
    assert (cfg["batch_size"], cfg["accum_grad"]) == (batch, accum)
    assert cfg["max_epoch"] == epochs
    assert cfg["num_iters_per_epoch"] == iters
    assert cfg["scheduler_conf"]["warmup_steps"] == steps
    if mode == "csall" or phase == "frozen":
        assert cfg["best_model_criterion"] == [["valid", "loss", "min"]]
    spec = importlib.util.spec_from_file_location(
        "phase_budget", RECIPE / "local/make_fixed_batch_config.py"
    )
    budget_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget_module)
    budget = budget_module.compute_budget(
        num_utts=308643 if mode == "mixed" else 329192,
        batch_size=batch, accum_grad=accum, effective_batch_size=32, ngpu=0,
        max_epoch=epochs, target_passes=passes, warmup_ratio=warmup,
        round_updates_per_epoch_to=10,
    )
    assert budget["num_iters_per_epoch"] == iters
    assert budget["warmup_steps"] == steps
    assert budget["total_optimizer_updates"] == epochs * iters // accum


@pytest.mark.parametrize("mode,phase,batch,accum,epochs,passes,warmup,iters,steps", PHASES)
def test_wrapper_phase_defaults_reach_auto_budget(
    work, mode, phase, batch, accum, epochs, passes, warmup, iters, steps
):
    options = forwarded(invoke(
        work, "--train_mode", mode, "--training_phase", phase,
        "--stage", "10", "--stop_stage", "10", "--auto_training_budget", "true",
    ))
    path = work / options["--asr_config"]
    budget = json.loads(path.with_suffix(".budget.json").read_text())
    assert Path(budget["base_config"]).name == config_name(mode, phase)
    assert (budget["batch_size"], budget["accum_grad"]) == (batch, accum)
    assert budget["max_epoch"] == epochs
    assert budget["target_passes"] == passes
    assert budget["warmup_ratio"] == warmup
    assert budget["num_train_utts"] == 32
    assert budget["ngpu"] == 0
    assert options["--feats_type"] == ("raw_copy" if mode == "csall" else "raw")
    data_args = shlex.split(options["--local_data_opts"])
    data = dict(zip(data_args[::2], data_args[1::2]))
    assert data["--skip_fleurs_download"] == "true"
    assert data["--cs_dev_ratio"] == "0.02"
    assert data["--cs_split_mode"] == "global_hash"


def test_explicit_probe_budget_is_not_replaced_by_phase_defaults(work):
    options = forwarded(invoke(
        work, "--stage", "10", "--stop_stage", "10",
        "--auto_training_budget", "true", "--train_batch_size", "16",
        "--accum_grad", "2", "--target_passes", "3",
        "--budget_max_epoch", "30", "--warmup_ratio", "0.1",
    ))
    budget_path = (work / options["--asr_config"]).with_suffix(".budget.json")
    budget = json.loads(budget_path.read_text())
    assert (budget["batch_size"], budget["accum_grad"]) == (16, 2)
    assert (
        budget["max_epoch"], budget["target_passes"], budget["warmup_ratio"]
    ) == (30, 3, 0.1)


@pytest.mark.parametrize("mode", ["mixed", "csall"])
@pytest.mark.parametrize("phase", ["frozen", "unfrozen"])
def test_new_phase_forwards_explicit_no_resume(work, mode, phase):
    init = weights(work) if phase == "unfrozen" else ""
    options = forwarded(invoke(
        work, "--train_mode", mode, "--training_phase", phase,
        "--pretrained_model", init,
    ))
    assert options["--pretrained_model"] == init
    assert shlex.split(options["--asr_args"])[-2:] == ["--resume", "false"]
    assert not (work / "exp").exists()


@pytest.mark.parametrize(
    "kind", ["missing", "empty", "checkpoint", "checkpoint_alias"]
)
def test_new_unfrozen_requires_existing_model_only_weights(work, kind):
    path = work / "chosen.pth"
    if kind == "empty":
        path.touch()
    elif kind in ("checkpoint", "checkpoint_alias"):
        checkpoint = Path(weights(work, "checkpoint.pth"))
        if kind == "checkpoint":
            path = checkpoint
        else:
            path.symlink_to(checkpoint)
    result = invoke(
        work, "--training_phase", "unfrozen", "--pretrained_model", str(path)
    )
    message = (
        "model-only" if kind.startswith("checkpoint") else "requires --pretrained_model"
    )
    refused(result, message)


def test_frozen_rejects_experiment_initialization(work):
    refused(invoke(work, "--pretrained_model", weights(work)), "frozen phase starts")


def test_explicit_best_alias_is_resolved_before_launch(work):
    target = Path(weights(work, "9epoch.pth"))
    alias = work / "valid.loss.best.pth"
    alias.symlink_to(target.name)
    options = forwarded(invoke(
        work, "--training_phase", "unfrozen", "--pretrained_model", str(alias)
    ))
    assert options["--pretrained_model"] == str(target.resolve())


@pytest.mark.parametrize("artifact", ["checkpoint.pth", "config.yaml"])
def test_new_run_never_overwrites_existing_experiment(work, artifact):
    directory = work / "exp/asr_phase_test"
    directory.mkdir(parents=True)
    path = directory / artifact
    path.write_bytes(b"original artifact")
    refused(invoke(work), "experiment already exists")
    assert path.read_bytes() == b"original artifact"


@pytest.mark.parametrize("phase", ["frozen", "unfrozen"])
def test_resume_uses_own_checkpoint_without_parent_initialization(work, phase):
    directory = saved_phase(work, "mixed", phase)
    before = {p: p.read_bytes() for p in directory.iterdir()}
    options = forwarded(invoke(work, "--training_phase", phase, "--resume", "true"))
    assert options["--pretrained_model"] == ""
    assert shlex.split(options["--asr_args"])[-2:] == ["--resume", "true"]
    for path, value in before.items():
        assert path.read_bytes() == value


@pytest.mark.parametrize("mode", ["mixed", "csall"])
@pytest.mark.parametrize("phase", ["frozen", "unfrozen"])
def test_resume_accepts_matching_generated_budget_without_rewriting(work, mode, phase):
    options = forwarded(invoke(
        work, "--train_mode", mode, "--training_phase", phase,
        "--stage", "10", "--stop_stage", "10", "--auto_training_budget", "true",
    ))
    config_path = work / options["--asr_config"]
    config = yaml.safe_load(config_path.read_text())
    saved_phase(work, mode, phase, config=config)
    paths = [
        config_path,
        config_path.with_suffix(".budget.json"),
        work / "exp/asr_phase_test/config.yaml",
        work / "exp/asr_phase_test/checkpoint.pth",
    ]
    before = {path: path.read_bytes() for path in paths}
    resumed = forwarded(invoke(
        work, "--train_mode", mode, "--training_phase", phase,
        "--resume", "true", "--auto_training_budget", "true",
    ))
    assert resumed["--asr_config"] == options["--asr_config"]
    assert resumed["--pretrained_model"] == ""
    assert shlex.split(resumed["--asr_args"])[-2:] == ["--resume", "true"]
    for path, value in before.items():
        assert path.read_bytes() == value


def test_resume_requires_own_checkpoint(work):
    refused(invoke(work, "--resume", "true"), "requires this phase's")


def test_resume_accepts_legacy_pit_keys_without_modifying_saved_config(work):
    config = read_config("mixed", "frozen")
    model = config["model_conf"]
    model["lidseq_order_insensitive_loss"] = model.pop("pit_loss")
    model["lidseq_order_insensitive_reduction"] = model.pop("pit_loss_reduction")
    directory = saved_phase(work, "mixed", "frozen", config)
    before = (directory / "config.yaml").read_bytes()
    options = forwarded(invoke(work, "--resume", "true"))
    assert shlex.split(options["--asr_args"])[-2:] == ["--resume", "true"]
    assert (directory / "config.yaml").read_bytes() == before


def test_resume_rejects_conflicting_pit_aliases(work):
    config = read_config("mixed", "frozen")
    config["model_conf"]["lidseq_order_insensitive_loss"] = False
    saved_phase(work, "mixed", "frozen", config)
    refused(invoke(work, "--resume", "true"), "conflicting pit_loss")


@pytest.mark.parametrize(
    "key,value",
    [
        ("report_cer", True), ("report_wer", True), ("ignore_id", -1),
        ("sym_space", "<space>"), ("autocast_frontend", False), ("aux_ctc", None),
        ("transducer_multi_blank_durations", []),
    ],
)
def test_resume_accepts_explicit_legacy_model_defaults_without_writes(work, key, value):
    config = read_config("mixed", "frozen")
    model = config["model_conf"]
    model["lidseq_order_insensitive_loss"] = model.pop("pit_loss")
    model["lidseq_order_insensitive_reduction"] = model.pop("pit_loss_reduction")
    assert key not in model
    model[key] = value
    directory = saved_phase(work, "mixed", "frozen", config)
    before = {p: p.read_bytes() for p in directory.iterdir()}
    options = forwarded(invoke(work, "--resume", "true"))
    assert shlex.split(options["--asr_args"])[-2:] == ["--resume", "true"]
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert not (work / "conf/generated").exists()


def test_resume_accepts_requested_defaults_omitted_from_saved_config(work):
    directory = saved_phase(work, "mixed", "frozen")
    config = read_config("mixed", "frozen")
    config["model_conf"].update(report_cer=True, report_wer=True)
    requested = work / "explicit_defaults.yaml"
    requested.write_text(yaml.safe_dump(config))
    paths = [requested, *directory.iterdir()]
    before = {p: p.read_bytes() for p in paths}
    forwarded(invoke(work, "--resume", "true", "--asr_config", str(requested)))
    assert all(p.read_bytes() == contents for p, contents in before.items())


@pytest.mark.parametrize("key,value", [("report_cer", False), ("report_wer", False), ("ignore_id", -2)])
def test_resume_rejects_saved_nondefaults_not_explicitly_requested(work, key, value):
    config = read_config("mixed", "frozen")
    assert key not in config["model_conf"]
    config["model_conf"][key] = value
    directory = saved_phase(work, "mixed", "frozen", config)
    before = (directory / "config.yaml").read_bytes()
    refused(invoke(work, "--resume", "true"), "saved model_conf")
    assert (directory / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("value", [None, 0.0])
def test_resume_rejects_unknown_saved_model_fields(work, value):
    config = read_config("mixed", "frozen")
    config["model_conf"]["future_objective_weight"] = value
    directory = saved_phase(work, "mixed", "frozen", config)
    before = (directory / "config.yaml").read_bytes()
    refused(invoke(work, "--resume", "true"), "unknown model_conf fields: future_objective_weight")
    assert (directory / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("key,value", [("lsm_weight", 0.2), ("length_normalized_loss", True), ("ctc_weight", 0.1)])
def test_resume_rejects_changed_explicit_nested_objective(work, key, value):
    directory = saved_phase(work, "mixed", "frozen")
    config = read_config("mixed", "frozen")
    assert config["model_conf"][key] != value
    config["model_conf"][key] = value
    requested = work / "changed_objective.yaml"
    requested.write_text(yaml.safe_dump(config))
    before = (directory / "config.yaml").read_bytes()
    refused(invoke(work, "--resume", "true", "--asr_config", str(requested)), "saved model_conf")
    assert (directory / "config.yaml").read_bytes() == before


def test_resume_default_completion_does_not_hide_reduction_alias_conflict(work):
    config = read_config("mixed", "frozen")
    config["model_conf"].update(report_cer=True, lidseq_order_insensitive_reduction="mean")
    directory = saved_phase(work, "mixed", "frozen", config)
    before = (directory / "config.yaml").read_bytes()
    refused(invoke(work, "--resume", "true"), "conflicting pit_loss_reduction")
    assert (directory / "config.yaml").read_bytes() == before


def test_resume_cannot_regenerate_data_or_budget(work):
    saved_phase(work, "mixed", "frozen")
    refused(invoke(
        work, "--resume", "true", "--stage", "1",
        "--auto_training_budget", "true",
    ), "--stage 11")
    assert not (work / "conf/generated").exists()
    assert not (work / "data/local").exists()


def test_resume_cannot_also_initialize_from_parent(work):
    saved_phase(work, "mixed", "unfrozen")
    refused(invoke(
        work, "--training_phase", "unfrozen", "--resume", "true",
        "--pretrained_model", weights(work),
    ), "not both")


def test_resume_cannot_change_freeze_phase(work):
    saved_phase(work, "mixed", "frozen")
    refused(
        invoke(work, "--training_phase", "unfrozen", "--resume", "true"),
        "saved freeze_param",
    )


def test_explicit_config_must_match_phase(work):
    refused(invoke(
        work, "--asr_config", "conf/" + config_name("mixed", "unfrozen"),
    ), "freeze_param does not match")


@pytest.mark.parametrize("phase", ["frozen", "unfrozen"])
@pytest.mark.parametrize("stage", [1, 10, 12])
def test_nontraining_stages_do_not_require_init_or_resume_artifacts(work, phase, stage):
    options = forwarded(invoke(
        work, "--training_phase", phase, "--stage", str(stage),
        "--stop_stage", str(stage),
    ))
    assert "--resume" not in shlex.split(options["--asr_args"])
    assert options["--pretrained_model"] == ""


@pytest.mark.parametrize(
    "option", ["--resume true", "--init_param fake.pth", "--freeze_param encoder"]
)
def test_phase_controls_cannot_be_overridden_via_asr_args(work, option):
    refused(
        invoke(work, "--asr_args", option), "training --asr_args is not supported"
    )


@pytest.mark.parametrize("option", ["--output_dir elsewhere", "--config replacement.yaml"])
def test_asr_args_cannot_bypass_validated_config_or_destination(work, option):
    refused(invoke(work, "--asr_args", option), "Error:")


@pytest.mark.parametrize(
    "option",
    [
        "--output_d other_output",
        "--model_conf lidseq_order_insensitive_loss=false",
        "--model_c lidseq_order_insensitive_loss=false",
        "--scheduler_conf warmup_steps=1",
        "--scheduler_c warmup_steps=1",
    ],
)
def test_training_rejects_runtime_config_overrides_and_abbreviations(work, option):
    result = invoke(work, "--asr_args", option)
    assert result.returncode == 2, result.stdout + result.stderr
    refused(result, "training --asr_args is not supported")
    assert "use the explicit config and phase/resume options" in result.stderr


def test_frozen_config_cannot_hide_experiment_initialization(work):
    config = read_config("mixed", "frozen")
    config["init_param"] = [weights(work)]
    path = work / "hidden_init.yaml"
    path.write_text(yaml.safe_dump(config))
    refused(invoke(work, "--asr_config", str(path)), "init_param")


def test_invalid_phase_is_rejected_before_data_preparation(work):
    result = invoke(
        work, "--stage", "1", "--auto_training_budget", "true",
        "--asr_config", "conf/" + config_name("mixed", "unfrozen"),
    )
    refused(result, "freeze_param does not match")
    assert not (work / "conf/generated").exists()
    assert not (work / "data/local").exists()
