import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest


RECIPE = Path(__file__).resolve().parents[3] / "egs2/fleurs_cs_lid"


def run_wrapper(
    tmp_path,
    recipe,
    *args,
    preprocessor="lid",
    allow_scoring=False,
    duration_fixture=False,
):
    work = tmp_path / recipe
    work.mkdir()
    (work / "utils").symlink_to(RECIPE / recipe / "utils", target_is_directory=True)
    config = work / "synthetic.yaml"
    config.write_text(
        f"preprocessor: {preprocessor}\nmax_epoch: 30\nbatch_type: catbel\n"
        + (
            "freeze_param: [frontend.upstream]\nmodel_conf:\n"
            "  lidseq_order_insensitive_loss: true\n"
            "  lidseq_order_insensitive_reduction: min\n"
            if recipe == "asr1"
            else ""
        )
    )
    sentinel = work / "data/existing/wav.scp"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("u1 /nonexistent/unchanged.wav\n")
    if duration_fixture:
        (tmp_path / "local").symlink_to(RECIPE / "local", target_is_directory=True)
        for split in ("train", "valid"):
            name = f"{split}_fleurs_lid"
            files = {
                "wav.scp": "u1 /audio/a.wav\nu2 /audio/b.wav\n",
                "text": "u1 <eng>\nu2 <fas>\n",
                "utt2spk": "u1 u1\nu2 u2\n",
                "utt2lang": "u1 eng\nu2 fas\n",
                "utt2langs": "u1 eng\nu2 fas\n",
                "spk2utt": "u1 u1\nu2 u2\n",
                "lang2utt": "eng u1\nfas u2\n",
                "utt2category": "u1 fleurs\nu2 fleurs\n",
                "utt2dur": "u1 2.000000\nu2 1.023062\n",
                "utt2num_samples": "u1 32000\nu2 15345\n",
            }
            for root in (
                work / "data",
                work / "legacy_dump/raw",
                work / "legacy_dump/raw_copy",
            ):
                dest = root / name
                dest.mkdir(parents=True)
                for key, value in files.items():
                    if key == "utt2num_samples" and root.name != "raw":
                        value = "u1 32000\nu2 16368\n"
                    (dest / key).write_text(value)
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        CS_FLEURS_ROOT=str(work / "synthetic_cs"),
        FLEURS=str(work / "synthetic_fleurs"),
        PYTHONDONTWRITEBYTECODE="1",
        WRAPPER_PYTHON=sys.executable,
        WRAPPER_ALLOW_SCORING=str(int(allow_scoring)),
        WRAPPER_DURATION_FIXTURE=str(int(duration_fixture)),
    )
    # Run the real wrapper/parser, but never reach data preparation or ESPnet.
    harness = r"""
forbidden() { printf 'FORBIDDEN: %s\n' "$*" >&2; exit 91; }
function local/data.sh() { forbidden local/data.sh; }
function local/verify_lid_data.sh() {
    [ "$WRAPPER_DURATION_FIXTURE" = 1 ] || forbidden local/verify_lid_data.sh
    printf 'PREFLIGHT\n'
}
function local/score_lidseq_decodes.sh() {
    [ "$WRAPPER_ALLOW_SCORING" = 1 ] || forbidden local/score_lidseq_decodes.sh
    printf 'SCORE_ARG=%s\n' "$@"
}
function mkdir() { forbidden mkdir; }
function python3() {
    if [ "$1" = - ] || { [ "$WRAPPER_DURATION_FIXTURE" = 1 ] && [ "$1" = ../local/finalize_duration_view.py ]; }; then
        "$WRAPPER_PYTHON" "$@"
    else
        forbidden python3 "$@"
    fi
}
function ./asr.sh() { printf 'TEMPLATE_ARG=%s\n' "$@"; }
function ./lid.sh() { printf 'TEMPLATE_ARG=%s\n' "$@"; }
source "$1" "${@:2}"
"""
    config_option = "--asr_config" if recipe == "asr1" else "--lid_config"
    result = subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "wrapper-test",
            str(RECIPE / recipe / "run.sh"),
            "--ngpu",
            "0",
            config_option,
            str(config),
            *args,
        ],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "FORBIDDEN:" not in result.stderr, result.stderr
    assert sentinel.read_text() == "u1 /nonexistent/unchanged.wav\n"
    assert not (work / "conf/generated").exists()
    assert not (work / "data/local").exists()
    return result


def template_options(result):
    assert result.returncode == 0, result.stdout + result.stderr
    args = [
        line.removeprefix("TEMPLATE_ARG=")
        for line in result.stdout.splitlines()
        if line.startswith("TEMPLATE_ARG=")
    ]
    assert args and len(args) % 2 == 0, result.stdout
    return dict(zip(args[::2], args[1::2]))


@pytest.mark.parametrize("preprocessor,profile,batch,accum", [
    ("lid", "mixed", 8, 4),
    ("lid", "csall", 4, 8),
    ("lid_softlabel", "mixed", 8, 4),
    ("lid_softlabel", "csall", 8, 4),
    ("lid_multilabel", "mixed", 4, 8),
    ("lid_multilabel", "csall", 8, 4),
    ("lid_multilabel", "fleurs_only", 16, 2),
])
def test_lid_reference_batch_defaults(tmp_path, preprocessor, profile, batch, accum):
    result = run_wrapper(
        tmp_path, "lid1", "--stage", "1", "--stop_stage", "1",
        "--profile", profile, "--cs_yodas_root", str(tmp_path / "no_yodas"),
        preprocessor=preprocessor,
    )
    options = template_options(result)
    assert f"_bs{batch}_ag{accum}_eb32_" in options["--lid_tag"]


@pytest.mark.parametrize("recipe", ["asr1", "lid1"])
def test_rawcopy_cannot_reinterpret_wav_as_archive(recipe):
    result = subprocess.run(
        [
            "bash",
            "run.sh",
            "--feats_type",
            "raw_copy",
            "--audio_format",
            "wav.ark",
            "--stage",
            "3",
            "--stop_stage",
            "3",
            "--ngpu",
            "0",
        ],
        cwd=RECIPE / recipe,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not an archive format" in result.stderr


@pytest.mark.parametrize("recipe", ["asr1", "lid1"])
def test_stage1_rejects_whitespace_audio_root_before_preparation(tmp_path, recipe):
    result = run_wrapper(
        tmp_path,
        recipe,
        "--stage",
        "1",
        "--stop_stage",
        "1",
        "--fleurs_audio_root",
        "/new audio root",
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "wrapper Stage 1 does not support whitespace in input paths" in result.stderr
    assert "Run local/data.sh directly with quoted paths" in result.stderr
    assert "start the wrapper at Stage 3" in result.stderr
    assert "FORBIDDEN:" not in result.stderr
    assert "TEMPLATE_ARG=" not in result.stdout


def test_copy_preserves_labels_without_language_prefixed_ids(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    files = {
        "wav.scp": "u1 /nonexistent/audio.wav\nu2 /nonexistent/audio.wav\n",
        "text": "u1 <eng>\nu2 <ara-eng>\n",
        "utt2spk": "u1 u1\nu2 u2\n",
        "spk2utt": "u1 u1\nu2 u2\n",
        "utt2lang": "u1 eng\nu2 ara-eng\n",
        "utt2langs": "u1 eng\nu2 ara-eng\n",
        "lang2utt": "ara-eng u2\neng u1\n",
        "utt2num_samples": "u1 16000\nu2 32000\n",
    }
    for name, value in files.items():
        (source / name).write_text(value)
        (source / name).chmod(0o444)
    for attempt in range(2):
        result = subprocess.run(
            [
                "bash",
                "local/copy_data_dir.sh",
                "--validate_opts",
                "--non-print",
                str(source),
                str(destination),
            ],
            cwd=RECIPE / "lid1",
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        for name, value in files.items():
            assert (source / name).read_text() == value
            assert stat.S_IMODE((source / name).stat().st_mode) == 0o444
            assert (destination / name).read_text() == value
        assert (destination / "utt2langs").stat().st_mode & stat.S_IWUSR
        category2utt = destination / "category2utt"
        assert category2utt.read_text() == files["lang2utt"]
        assert category2utt.stat().st_mode & stat.S_IWUSR
        # TEMPLATE/lid1 repeats this cp immediately after the local helper.
        subprocess.run(
            ["cp", str(source / "lang2utt"), str(category2utt)],
            check=True,
            capture_output=True,
            timeout=30,
        )
        assert category2utt.read_text() == files["lang2utt"]
        assert category2utt.stat().st_mode & stat.S_IWUSR
        if attempt == 0:
            # Also cover a shared dump left read-only by an older template copy.
            category2utt.chmod(0o444)


@pytest.mark.parametrize(
    "recipe,stage,stop_stage",
    [("asr1", 2, 10), ("asr1", 10, 10), ("lid1", 2, 5), ("lid1", 5, 5)],
)
def test_resume_missing_data_never_regenerates(tmp_path, recipe, stage, stop_stage):
    profile_option = "--train_mode" if recipe == "asr1" else "--profile"
    result = run_wrapper(
        tmp_path,
        recipe,
        "--stage",
        str(stage),
        "--stop_stage",
        str(stop_stage),
        profile_option,
        "csall",
        "--cs_yodas_root",
        "",
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "missing prepared data" in result.stderr
    assert "run Stage 1 explicitly" in result.stderr
    assert "TEMPLATE_ARG=" not in result.stdout


@pytest.mark.parametrize("recipe", ["asr1", "lid1"])
@pytest.mark.parametrize("yodas_root", ["", "synthetic_yodas"])
def test_stage1_preserves_optional_data_arguments(tmp_path, recipe, yodas_root):
    result = run_wrapper(
        tmp_path,
        recipe,
        "--stage",
        "1",
        "--stop_stage",
        "1",
        "--cs_yodas_root",
        yodas_root,
        "--max_train_duration_sec",
        "20",
    )
    options = template_options(result)
    forwarded = shlex.split(options["--local_data_opts"])
    assert len(forwarded) % 2 == 0
    data_options = dict(zip(forwarded[::2], forwarded[1::2]))
    assert data_options["--max_train_duration_sec"] == "20"
    assert data_options["--cs_yodas_train_ratio"] == "0.8"
    if yodas_root:
        assert data_options["--cs_yodas_root"] == yodas_root
    else:
        assert "--cs_yodas_root" not in data_options
    assert options["--stage"] == "1"


@pytest.mark.parametrize("gpu_inference", [None, "true", "false"])
def test_asr_ngpu_zero_always_decodes_on_cpu(tmp_path, gpu_inference):
    extra = [] if gpu_inference is None else ["--gpu_inference", gpu_inference]
    result = run_wrapper(
        tmp_path,
        "asr1",
        "--stage",
        "12",
        "--stop_stage",
        "12",
        "--run_lidseq_scoring",
        "false",
        *extra,
    )
    options = template_options(result)
    assert options["--ngpu"] == "0"
    assert options["--gpu_inference"] == "false"
    assert options["--stage"] == "12"


@pytest.mark.parametrize("stage,stop_stage", [(12, 12), (12, 13), (13, 13)])
@pytest.mark.parametrize("auto_budget", ["true", "false"])
def test_asr_eval_does_not_require_training_budget(
    tmp_path, stage, stop_stage, auto_budget
):
    # ngpu only exercises shell policy routing; the template is a CPU-only stub.
    result = run_wrapper(
        tmp_path,
        "asr1",
        "--stage",
        str(stage),
        "--stop_stage",
        str(stop_stage),
        "--ngpu",
        "2",
        "--require_cuda_visible_devices",
        "false",
        "--auto_training_budget",
        auto_budget,
        "--train_batch_size",
        "2",
        "--accum_grad",
        "16",
        "--budget_max_epoch",
        "7",
        "--asr_tag",
        "existing_run",
        "--run_lidseq_scoring",
        "false",
    )
    options = template_options(result)
    assert options["--asr_tag"] == "existing_run"
    assert options["--stage"] == str(stage)
    assert options["--stop_stage"] == str(stop_stage)
    assert Path(options["--asr_config"]).name == "synthetic.yaml"


@pytest.mark.parametrize(
    "auto_budget,error",
    [("false", "requires --auto_training_budget true"), ("true", "allowed fixed ASR")],
)
def test_asr_training_still_enforces_budget_policy(tmp_path, auto_budget, error):
    result = run_wrapper(
        tmp_path,
        "asr1",
        "--stage",
        "11",
        "--stop_stage",
        "11",
        "--ngpu",
        "2",
        "--require_cuda_visible_devices",
        "false",
        "--auto_training_budget",
        auto_budget,
        "--train_batch_size",
        "2",
        "--accum_grad",
        "16",
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert error in result.stderr
    assert "TEMPLATE_ARG=" not in result.stdout


def test_asr_training_resume_still_requires_generated_config(tmp_path):
    result = run_wrapper(tmp_path, "asr1", "--stage", "11", "--stop_stage", "11")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "generated ASR config missing for resume" in result.stderr
    assert "TEMPLATE_ARG=" not in result.stdout


@pytest.mark.parametrize("stage", [12, 13])
def test_asr_existing_decodes_can_be_scored_without_training_budget(tmp_path, stage):
    decode_dir = str(tmp_path / "asr1/data/existing")
    result = run_wrapper(
        tmp_path,
        "asr1",
        "--stage",
        str(stage),
        "--stop_stage",
        "13",
        "--decode_dir",
        decode_dir,
        "--asr_tag",
        "existing_run",
        allow_scoring=True,
    )
    options = template_options(result)
    assert options["--stage"] == str(stage)
    assert options["--stop_stage"] == "12"
    assert options["--gpu_inference"] == "false"
    score_args = [
        line.removeprefix("SCORE_ARG=")
        for line in result.stdout.splitlines()
        if line.startswith("SCORE_ARG=")
    ]
    assert score_args[:2] == ["--decode_dir", decode_dir]


@pytest.mark.parametrize("profile", ["mixed", "csall"])
@pytest.mark.parametrize("preprocessor", ["lid", "lid_softlabel", "lid_multilabel"])
def test_lid_explicit_legacy_tag_and_stats_are_preserved(
    tmp_path, profile, preprocessor
):
    result = run_wrapper(
        tmp_path,
        "lid1",
        "--stage",
        "4",
        "--stop_stage",
        "4",
        "--lid_tag",
        "legacy_run",
        "--lid_stats_dir",
        "legacy_exp/lid_stats_16k",
        "--expdir",
        "legacy_exp",
        "--dumpdir",
        "legacy_dump",
        "--profile",
        profile,
        "--train_set",
        "train_fleurs_lid",
        "--valid_set",
        "valid_fleurs_lid",
        duration_fixture=True,
        preprocessor=preprocessor,
    )
    options = template_options(result)
    assert options["--lid_tag"] == "legacy_run"
    assert options["--lid_stats_dir"] == "legacy_exp/lid_stats_16k"
    assert options["--expdir"] == "legacy_exp"
    raw = profile == "mixed" and preprocessor != "lid_softlabel"
    assert options["--dumpdir"] == (
        "legacy_dump/old_duration" if raw else "legacy_dump"
    )
    assert "PREFLIGHT" in result.stdout
    if raw:
        assert (
            tmp_path / "lid1/legacy_dump/old_duration/raw/train_fleurs_lid/wav.scp"
        ).read_text() == "u1 /audio/a.wav\n"
    else:
        assert not (tmp_path / "lid1/legacy_dump/old_duration").exists()
        assert options["--feats_type"] == "raw_copy"


@pytest.mark.parametrize(
    "preprocessor,train_set,label_file",
    [
        ("lid", "train_lid_pair_cs", "utt2lang"),
        ("lid_softlabel", "train_lidseq", "utt2langs"),
        ("lid_multilabel", "train_lidseq", "utt2langs"),
    ],
)
def test_lid_mixed_target_routing(tmp_path, preprocessor, train_set, label_file):
    result = run_wrapper(
        tmp_path,
        "lid1",
        "--stage",
        "1",
        "--stop_stage",
        "1",
        "--profile",
        "mixed",
        preprocessor=preprocessor,
    )
    options = template_options(result)
    assert options["--train_set"] == train_set
    assert options["--valid_set"] == train_set.replace("train_", "valid_", 1)
    assert options["--lid_label_file"] == label_file
