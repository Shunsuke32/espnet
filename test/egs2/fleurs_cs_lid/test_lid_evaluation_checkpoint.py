"""Real ESPnet component checkpoints, without training or pretrained downloads."""

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml


RECIPE = Path(__file__).resolve().parents[3] / "egs2" / "fleurs_cs_lid"
LOCAL = RECIPE / "lid1" / "local"


@pytest.fixture
def evaluator(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(LOCAL))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba-cache"))
    spec = importlib.util.spec_from_file_location(
        "checkpoint_eval", LOCAL / "evaluate_lid.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("reporter_format", ["current", "legacy", "object"])
def test_real_trainer_schema_after_workspace_move(
    tmp_path, evaluator, monkeypatch, reporter_format
):
    from espnet2.schedulers.warmup_lr import WarmupLR
    from espnet2.tasks.lid import LIDTask
    from espnet2.train.reporter import Reporter

    source = tmp_path / "original-workspace"
    source.mkdir()
    train = source / "train"
    train.mkdir()
    (train / "lang2utt").write_text("ara t1 t4\neng t2 t4\njpn t3\n")
    (train / "utt2langs").write_text("t1 ara\nt2 eng\nt3 jpn\nt4 ara eng\n")
    (source / "lang2utt").write_bytes((train / "lang2utt").read_bytes())
    config = dict(
        frontend="default",
        frontend_conf=dict(fs=16000, n_fft=128, hop_length=32, n_mels=8),
        specaug=None,
        normalize=None,
        encoder="identity",
        encoder_conf={},
        pooling="mean",
        pooling_conf={},
        projector="rawnet3",
        projector_conf=dict(output_size=4),
        loss="arc_margin_subcenter_intertopk_multilabel_bce",
        loss_conf=dict(K=3, k_top=0),
        lang_num=3,
        model_conf={},
        init=None,
        use_preprocessor=True,
        preprocessor="lid_multilabel",
        preprocessor_conf={},
        lang2utt=str(train / "lang2utt"),
    )
    (source / "config.yaml").write_text(yaml.safe_dump(config))
    original, _ = LIDTask.build_model_from_file(source / "config.yaml", None, "cpu")
    original.eval()
    reporter = Reporter(epoch=1)
    with reporter.observe("valid") as observation:
        observation.register({"loss": 0.75})
        observation.next()
    # Match Trainer's schema using real components; no optimizer/training step.
    optimizer = torch.optim.Adam(original.parameters(), lr=0.001)
    scheduler = WarmupLR(optimizer, warmup_steps=2)
    report_state = {
        "current": reporter.state_dict(),
        "legacy": {"epoch": reporter.get_epoch(), "stats": reporter.stats},
        "object": reporter,
    }[reporter_format]
    checkpoint = dict(
        model=original.state_dict(),
        reporter=report_state,
        optimizers=[optimizer.state_dict()],
        schedulers=[scheduler.state_dict()],
        scaler=None,
    )
    torch.save(checkpoint, source / "checkpoint.pth")
    moved = tmp_path / "relocated-workspace"
    source.rename(moved)
    assert not Path(config["lang2utt"]).exists()
    before_config = (moved / "config.yaml").read_bytes()
    before_inventory = (moved / "lang2utt").read_bytes()

    data = moved / "data" / "test_fleurs_lid"
    data.mkdir(parents=True)
    samples = (0.1 * np.sin(np.arange(2048) * 0.2)).astype(np.float32)
    sf.write(data / "u.wav", samples, 16000, subtype="FLOAT")
    (data / "wav.scp").write_text(f"u {data / 'u.wav'}\n")
    (data / "utt2langs").write_text("u eng\n")
    elsewhere = tmp_path / "unrelated-working-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    args = evaluator.get_parser().parse_args(
        [
            "--model_file",
            str(moved / "checkpoint.pth"),
            "--config_file",
            str(moved / "config.yaml"),
            "--train_data_dir",
            str(moved / "train"),
            "--data_dir",
            str(moved / "data"),
            "--test_sets",
            "test_fleurs_lid",
            "--output_dir",
            str(moved / "evaluation"),
            "--ngpu",
            "0",
            "--batch_size",
            "1",
            "--thresholds",
            "0.5",
            "--no_plots",
        ]
    )
    if reporter_format != "current":
        with pytest.raises(ValueError, match="--trusted_checkpoint"):
            evaluator.run(args)
        assert not (args.output_dir / "scores.npz").exists()
        args.trusted_checkpoint = True
    evaluator.run(args)
    with torch.inference_mode():
        expected = evaluator.model_logits(
            original,
            torch.from_numpy(samples)[None],
            torch.tensor([len(samples)]),
            evaluator.head_spec(config, ["ara", "eng", "jpn"]),
            3,
        ).numpy()
    with np.load(args.output_dir / "scores.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["logits"], expected, rtol=1e-5, atol=1e-6)
        metadata = json.loads(str(saved["metadata"]))
    kind = "trainer_checkpoint.model"
    if reporter_format != "current":
        kind += ":trusted_pickle"
    assert metadata["checkpoint_format"] == kind
    assert (moved / "config.yaml").read_bytes() == before_config
    assert (moved / "lang2utt").read_bytes() == before_inventory
    assert not source.exists()


def test_trusted_flag_keeps_safe_first_and_never_relaxes_state_keys(
    tmp_path, evaluator, monkeypatch
):
    from espnet2.tasks.lid import LIDTask
    from espnet2.train.reporter import Reporter

    path = tmp_path / "checkpoint.pth"
    model = torch.nn.Linear(3, 2)
    torch.save(dict(model=model.state_dict(), reporter={}), path)
    loader = torch.load
    calls = []

    def record_load(*args, **kwargs):
        calls.append(kwargs["weights_only"])
        return loader(*args, **kwargs)

    monkeypatch.setattr(torch, "load", record_load)
    monkeypatch.setattr(
        LIDTask,
        "build_model_from_file",
        lambda *_: (torch.nn.Linear(3, 2), argparse.Namespace()),
    )
    evaluator.load_model(tmp_path / "config.yaml", path, "cpu", trusted_checkpoint=True)
    assert calls == [True]
    calls.clear()
    torch.save(dict(model={"weight": model.weight}, reporter=Reporter()), path)
    with pytest.raises(RuntimeError, match="Missing key"):
        evaluator.load_model(
            tmp_path / "config.yaml", path, "cpu", trusted_checkpoint=True
        )
    assert calls == [True, False]


def test_evaluation_readme_flags_and_command_match_help(evaluator):
    readme = (
        (RECIPE / "evaluation" / "README.md")
        .read_text()
        .split("## ASR Aggregate Details")[0]
    )
    command = readme.split("```bash\n", 1)[1].split("```", 1)[0]
    words = shlex.split(command.replace("\\\n", " "))
    assert words[0] == "local/evaluate.sh"
    args = evaluator.get_parser().parse_args(words[1:])
    assert args.ngpu == 0 and args.batch_size == 1 and not args.trusted_checkpoint
    help_text = evaluator.get_parser().format_help()
    for flag in re.findall(r"--[a-z_]+", readme):
        assert flag in help_text
    assert "can execute code" in " ".join(help_text.split())
