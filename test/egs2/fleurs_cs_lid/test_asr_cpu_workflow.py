"""Offline ASR API smoke, NOT numerical reproduction of historical experiments.

With the ESPnet Python environment active, run without pytest::

    source tools/activate_python.sh  # Or use an already active ESPnet environment.
    export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    export NUMBA_CACHE_DIR=/tmp/cs-lid-asr-numba
    python3 test/egs2/fleurs_cs_lid/test_asr_cpu_workflow.py --work-dir /tmp/asr-smoke

The work directory must not exist. Synthetic WAVs, configs, logs, checkpoints,
decoded text, scores and report.json are retained there. Ordinary pytest/unittest
discovery skips the workflow unless CS_LID_ASR_CPU_SMOKE=1. Both ordered and
order-insensitive losses train for exactly one optimizer update. This exercises
ASRTask, Speech2Text and the recipe scorer, not the recipe shell stage wrappers.
The ordered case is a shared-core compatibility regression, not a selected recipe
experiment. Only order-insensitive ASR is part of the recipe selection.
No MMS, dataset download, calibration, historical preset or GPU is involved.
"""

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve()
CASES = ("ordered", "insensitive")
TOKENS = ("<blank>", "<unk>", "<ara>", "<eng>", "<sos/eos>")
SCORER = ROOT / "egs2/fleurs_cs_lid/asr1/local/score_lidseq.py"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_kv(path):
    rows = [line.split(maxsplit=1) for line in path.read_text().splitlines()]
    assert rows and all(len(row) == 2 for row in rows), path
    assert len({row[0] for row in rows}) == len(rows), path
    return dict(rows)


def offline_environment(work):
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONPATH=str(ROOT),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(work / "cache/huggingface"),
        TORCH_HOME=str(work / "cache/torch"),
        MPLCONFIGDIR=str(work / "cache/matplotlib"),
        NUMBA_CACHE_DIR=str(work / "cache/numba"),
        XDG_CACHE_HOME=str(work / "cache"),
    )
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    return env


def create_fixtures(work):
    import numpy as np
    import soundfile as sf

    (work / "tokens.txt").write_text("\n".join(TOKENS) + "\n")
    targets = ("<eng>", "<ara>", "<eng> <ara>", "<ara> <eng>")
    for split in ("train", "valid", "test"):
        directory = work / "data" / split
        directory.mkdir(parents=True)
        wavs, text = [], []
        for index, target in enumerate(targets):
            uid = f"{split}_{index}"
            path = directory / (uid + ".wav")
            samples = 1600 + index * 160
            t = np.arange(samples, dtype=np.float64) / 16000
            signal = 0.1 * np.sin(2 * np.pi * (180 + 30 * index) * t)
            signal += 0.02 * np.cos(2 * np.pi * 71 * t)
            sf.write(path, signal.astype(np.float32), 16000, subtype="PCM_16")
            wavs.append(f"{uid} {path}\n")
            text.append(f"{uid} {target}\n")
        (directory / "wav.scp").write_text("".join(wavs))
        (directory / "text").write_text("".join(text))


def tiny_config(insensitive, work):
    return dict(
        token_list=str(work / "tokens.txt"),
        token_type="word",
        use_preprocessor=True,
        frontend="default",
        frontend_conf=dict(fs=16000, n_fft=128, hop_length=64, n_mels=8),
        normalize="utterance_mvn",
        normalize_conf=dict(norm_vars=False),
        specaug=None,
        encoder="transformer",
        encoder_conf=dict(
            output_size=16,
            attention_heads=2,
            linear_units=32,
            num_blocks=1,
            input_layer="linear",
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            attention_dropout_rate=0.0,
        ),
        decoder="transformer",
        decoder_conf=dict(
            attention_heads=2,
            linear_units=32,
            num_blocks=1,
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            self_attention_dropout_rate=0.0,
            src_attention_dropout_rate=0.0,
        ),
        model_conf=dict(
            ctc_weight=0.0,
            lsm_weight=0.1,
            length_normalized_loss=False,
            lidseq_order_insensitive_loss=insensitive,
            lidseq_order_insensitive_reduction="min",
            extract_feats_in_collect_stats=True,
            report_cer=False,
            report_wer=False,
        ),
        ngpu=0,
        seed=13,
        num_workers=0,
        max_epoch=1,
        num_iters_per_epoch=1,
        batch_size=2,
        valid_batch_size=2,
        accum_grad=1,
        batch_type="sorted",
        iterator_type="sequence",
        drop_last_iter=True,
        optim="adam",
        optim_conf=dict(lr=0.001, betas=[0.9, 0.98]),
        scheduler="warmuplr",
        scheduler_conf=dict(warmup_steps=1),
        freeze_param=[],
        init_param=[],
        resume=False,
        use_amp=False,
        cudnn_enabled=False,
        use_tensorboard=False,
        use_matplotlib=False,
        num_att_plot=0,
        log_interval=1,
        keep_nbest_models=1,
        best_model_criterion=[["valid", "loss", "min"]],
        patience=None,
    )


def worker(case, work):
    # Pin source before importing tasks: an editable live installation must not win.
    sys.path.insert(0, str(ROOT))

    def deny_network(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise RuntimeError(f"Offline smoke attempted network access: {event}")

    sys.addaudithook(deny_network)
    import numpy as np
    import soundfile as sf
    import torch
    import yaml
    from espnet2.bin.asr_inference import Speech2Text
    from espnet2.tasks.asr import ASRTask

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    modules = {}
    for name in (
        "espnet2.tasks.asr",
        "espnet2.asr.espnet_model",
        "espnet2.asr.frontend.default",
        "espnet2.asr.encoder.transformer_encoder",
        "espnet2.asr.decoder.transformer_decoder",
        "espnet2.bin.asr_inference",
        "espnet2.train.trainer",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        assert path.is_relative_to(ROOT), (name, path, ROOT)
        modules[name] = str(path)

    directory = work / case
    directory.mkdir()
    config_path = directory / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(tiny_config(case == "insensitive", work)))
    common = ["--config", str(config_path)]
    for split in ("train", "valid"):
        data = work / "data" / split
        for filename, name, dtype in (
            ("wav.scp", "speech", "sound"),
            ("text", "text", "text"),
        ):
            common.extend(
                [
                    f"--{split}_data_path_and_name_and_type",
                    f"{data / filename},{name},{dtype}",
                ]
            )
    stats = directory / "stats"
    ASRTask.main(cmd=common + ["--collect_stats", "true", "--output_dir", str(stats)])
    shapes = []
    for split in ("train", "valid"):
        expected = read_kv(work / "data" / split / "text")
        for name in ("speech", "text"):
            path = stats / split / (name + "_shape")
            entries = read_kv(path)
            assert entries.keys() == expected.keys(), (case, split, name)
            for uid, shape in entries.items():
                length = int(shape.split(",")[0])
                if name == "text":
                    assert length == len(expected[uid].split())
                else:
                    assert length == 1600 + int(uid.rsplit("_", 1)[1]) * 160
            shapes.extend([f"--{split}_shape_file", str(path)])
        with np.load(stats / split / "feats_stats.npz") as features:
            assert features["count"] > 0
            assert np.isfinite(features["sum"]).all()

    exp = directory / "exp"
    ASRTask.main(cmd=common + shapes + ["--output_dir", str(exp)])
    # Only load checkpoints created by this test in its new, private directory.
    checkpoint = torch.load(
        exp / "checkpoint.pth", map_location="cpu", weights_only=False
    )
    states = list(checkpoint["optimizers"][0]["state"].values())
    steps = [float(state["step"]) for state in states]
    assert steps and set(steps) == {1.0}, steps
    assert all(torch.isfinite(state["exp_avg"]).all() for state in states)
    assert any(torch.count_nonzero(state["exp_avg"]) for state in states)
    for value in checkpoint["model"].values():
        assert value.device.type == "cpu" and torch.isfinite(value).all()

    model, args = ASRTask.build_model_from_file(
        str(exp / "config.yaml"), str(exp / "1epoch.pth"), "cpu"
    )
    assert args.frontend == "default"
    assert args.encoder == args.decoder == "transformer"
    assert model.lidseq_order_insensitive_loss == (case == "insensitive")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint["model"][key], rtol=0, atol=0)

    data = work / "data/train"
    targets = read_kv(data / "text")
    preprocess = ASRTask.build_preprocess_fn(args, train=True)
    examples = []
    for uid, wav in read_kv(data / "wav.scp").items():
        speech, rate = sf.read(wav, dtype="float32")
        assert rate == 16000
        example = preprocess(uid, dict(speech=speech, text=targets[uid]))
        assert [TOKENS[i] for i in example["text"]] == targets[uid].split()
        examples.append((uid, example))
    _, batch = ASRTask.build_collate_fn(args, train=True)(examples)
    model.train()
    loss, _, _ = model(**batch)
    assert torch.isfinite(loss)
    loss.backward()
    for prefix in ("encoder.", "decoder."):
        gradients = [
            p.grad
            for name, p in model.named_parameters()
            if name.startswith(prefix) and p.grad is not None
        ]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients), prefix
    model.eval()
    swapped = dict(batch)
    swapped["text"] = batch["text"].clone()
    pairs = batch["text_lengths"] == 2
    swapped["text"][pairs, :2] = swapped["text"][pairs, :2].flip(1)
    with torch.inference_mode():
        original_loss = model(**batch)[0]
        swapped_loss = model(**swapped)[0]
    assert torch.isfinite(original_loss) and torch.isfinite(swapped_loss)
    if case == "insensitive":
        torch.testing.assert_close(original_loss, swapped_loss, rtol=1e-5, atol=1e-6)

    decoder = Speech2Text(
        asr_train_config=str(exp / "config.yaml"),
        asr_model_file=str(exp / "1epoch.pth"),
        device="cpu",
        beam_size=2,
        nbest=1,
        ctc_weight=0.0,
        lm_weight=0.0,
        maxlenratio=-3,
        minlenratio=0.0,
    )
    assert decoder.beam_search.beam_size == 2
    hypotheses = []
    for uid, wav in read_kv(work / "data/test/wav.scp").items():
        speech, _ = sf.read(wav, dtype="float32")
        results = decoder(speech)
        assert len(results) == 1
        text, tokens, token_ids, hypothesis = results[0]
        assert isinstance(text, str) and len(tokens) == len(token_ids)
        assert set(tokens) <= set(TOKENS)
        assert math.isfinite(float(hypothesis.score))
        hypotheses.append(uid + " " + " ".join(tokens) + "\n")
    (directory / "decoded.txt").write_text("".join(hypotheses))
    assert not torch.cuda.is_initialized()
    write_json(
        directory / "result.json",
        dict(
            case=case,
            module_files=modules,
            collect_stats=True,
            optimizer_updates=1,
            optimizer_parameter_states=len(steps),
            checkpoint=str(exp / "1epoch.pth"),
            checkpoint_reload_exact=True,
            forward_backward_loss=float(loss.detach()),
            original_loss=float(original_loss),
            swapped_loss=float(swapped_loss),
            order_invariance_checked=(case == "insensitive"),
            beam_size=2,
            decoded_utterances=len(hypotheses),
            cpu_threads=torch.get_num_threads(),
            gpu_initialized=False,
        ),
    )


def run_workflow(work):
    work = work.resolve()
    if not work.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise ValueError("Smoke artifacts must be in a new temporary directory")
    work.mkdir(parents=True, exist_ok=False)
    env = offline_environment(work)
    report = dict(
        scope=(
            "synthetic offline CPU ASR API smoke; NOT experiment reproduction; "
            "ordered case is shared-core compatibility only, not recipe selection"
        ),
        checkout=str(ROOT),
        wrapper_stages_tested=False,
        commands=[],
        cases={},
    )

    def command(name, argv):
        log = work / (name + ".log")
        start = time.monotonic()
        with log.open("w") as handle:
            result = subprocess.run(
                list(map(str, argv)),
                cwd=work,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
        report["commands"].append(
            dict(
                argv=list(map(str, argv)),
                log=str(log),
                returncode=result.returncode,
                seconds=round(time.monotonic() - start, 2),
            )
        )
        write_json(work / "report.json", report)
        if result.returncode:
            tail = "\n".join(log.read_text().splitlines()[-60:])
            raise AssertionError(f"{name}: {log}\n{tail}")

    try:
        # Fixture creation also runs with the same isolated cache/thread environment.
        command("fixtures", [sys.executable, SCRIPT, "--fixtures", "--work-dir", work])
        for case in CASES:
            print(f"ASR CPU smoke: {case}", flush=True)
            command(
                case, [sys.executable, SCRIPT, "--worker", case, "--work-dir", work]
            )
            directory = work / case
            command(
                case + "_score",
                [
                    sys.executable,
                    SCORER,
                    "--ref",
                    work / "data/test/text",
                    "--hyp",
                    directory / "decoded.txt",
                    "--out",
                    directory / "score.json",
                    "--details_out",
                    directory / "details.tsv",
                ],
            )
            metrics = json.loads((directory / "score.json").read_text())
            assert metrics["num_ref"] == metrics["num_hyp"] == 4
            assert metrics["num_missing_hyp"] == metrics["num_extra_hyp"] == 0
            for key in (
                "sequence_exact_accuracy",
                "unordered_set_exact_accuracy",
                "length_accuracy",
            ):
                assert 0.0 <= metrics[key] <= 1.0
            assert len((directory / "details.tsv").read_text().splitlines()) == 5
            report["cases"][case] = json.loads((directory / "result.json").read_text())
            report["cases"][case]["scorer_complete"] = True
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(work / "report.json", report)
    print(f"PASS: {work / 'report.json'}", flush=True)
    return report


@unittest.skipUnless(
    os.environ.get("CS_LID_ASR_CPU_SMOKE") == "1",
    "offline ASR integration smoke is opt-in",
)
class TestASRCPUWorkflow(unittest.TestCase):
    def test_workflow(self):
        with tempfile.TemporaryDirectory(prefix="cs-lid-asr-test-") as temporary:
            report = run_workflow(Path(temporary) / "workflow")
            self.assertEqual(set(report["cases"]), set(CASES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--fixtures", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.work_dir is None:
        if args.worker or args.fixtures:
            parser.error("internal workers require --work-dir")
        args.work_dir = Path(tempfile.mkdtemp(prefix="cs-lid-asr-smoke-")) / "workflow"
    if args.fixtures:
        create_fixtures(args.work_dir)
    elif args.worker:
        worker(args.worker, args.work_dir)
    else:
        run_workflow(args.work_dir)


if __name__ == "__main__":
    main()
