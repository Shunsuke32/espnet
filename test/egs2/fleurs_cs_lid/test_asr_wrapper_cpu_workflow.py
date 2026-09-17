"""Opt-in, offline ASR wrapper/template integration, NOT paper reproduction.

Standalone invocation (no pytest required)::

    source tools/activate_python.sh  # Or use an already active ESPnet environment.
    export CUDA_VISIBLE_DEVICES=''
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    export NUMBA_CACHE_DIR=/tmp/asr-wrapper-numba
    python3 test/egs2/fleurs_cs_lid/test_asr_wrapper_cpu_workflow.py \
        --work-dir /tmp/asr-wrapper-smoke

The work directory must not exist. Only synthetic audio and new checkpoints are
used. Stages 3/4/5/10/11/12 and the wrapper's custom scoring run through unmodified
recipe shell scripts. Stage 1 and corpus downloads are deliberately not run:
source manifests use the production writer with the existing ASR API fixtures.
Each of raw and raw_copy gets one unordered tiny Transformer optimizer update.
The raw_copy run initializes explicitly from this test's raw-run model weights;
this checks the phase protocol, not MMS freezing or historical d256 numerics.
Ordinary pytest/unittest discovery skips unless CS_LID_ASR_WRAPPER_SMOKE=1.
An optional CS_LID_SMOKE_ACTIVATE path reactivates an external environment in
subprocesses; otherwise the currently active environment is inherited.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve()

_spec = importlib.util.spec_from_file_location(
    "asr_api_smoke_fixtures", SCRIPT.with_name("test_asr_cpu_workflow.py")
)
api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api)


def run_command(argv, cwd, env, log, timeout=240):
    # Active environments work without a checkout-local activation script.
    activate = env.get("CS_LID_SMOKE_ACTIVATE")
    if not activate and (ROOT / "tools/activate_python.sh").is_file():
        activate = str(ROOT / "tools/activate_python.sh")
    prefix = f"source {shlex.quote(activate)} && " if activate else ""
    command = (
        prefix + f"export CS_FLEURS_ROOT={shlex.quote(env['CS_FLEURS_ROOT'])} && "
        "export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && "
        f"export NUMBA_CACHE_DIR={shlex.quote(env['NUMBA_CACHE_DIR'])} && "
        "exec " + shlex.join([str(arg) for arg in argv])
    )
    with log.open("w") as output:
        output.write(command + "\n")
        output.flush()
        process = subprocess.Popen(
            ["bash", "-c", command],
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if returncode:
        tail = "\n".join(log.read_text().splitlines()[-35:])
        raise RuntimeError(f"exit {returncode}: {log}\n{tail}")


def setup(work):
    import yaml

    source = ROOT / "egs2/fleurs_cs_lid"
    destination = work / "egs2/fleurs_cs_lid"
    destination.mkdir(parents=True)
    for name in ("asr1", "lid1", "local"):
        shutil.copytree(
            source / name,
            destination / name,
            symlinks=True,
            ignore=shutil.ignore_patterns(
                "data", "dump", "exp", "downloads", "hub", "__pycache__"
            ),
        )
    (work / "egs2/TEMPLATE").symlink_to(
        ROOT / "egs2/TEMPLATE", target_is_directory=True
    )
    (work / "tools").symlink_to(ROOT / "tools", target_is_directory=True)
    recipe = destination / "asr1"
    fixture_dir = work / "fixtures"
    fixture_dir.mkdir()
    api.create_fixtures(fixture_dir)
    spec = importlib.util.spec_from_file_location(
        "asr_wrapper_source_prep",
        destination / "lid1/local/prepare_fleurs_cs_lid_data.py",
    )
    prep = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = prep
    spec.loader.exec_module(prep)
    for split in ("train", "valid", "test"):
        manifest = fixture_dir / "data" / split
        text = api.read_kv(manifest / "text")
        examples = []
        for uid, wav in api.read_kv(manifest / "wav.scp").items():
            labels = tuple(token.strip("<>") for token in text[uid].split())
            samples = 1600 + int(uid.rsplit("_", 1)[1]) * 160
            examples.append(
                prep.Example(
                    uttid=uid,
                    wav=wav,
                    labels=labels,
                    speaker=uid,
                    source="fleurs" if len(labels) == 1 else "cs",
                    subset="synthetic",
                    duration_sec=samples / 16000,
                    num_samples=samples,
                )
            )
        prep.write_data_dir(split + "_smoke", examples, recipe / "data", "angle")
        assert api.read_kv(recipe / "data" / (split + "_smoke") / "text") == text
    (recipe / "data/local").mkdir()
    (recipe / "data/local/label_map.used.tsv").write_text("ara\tara\neng\teng\n")
    (recipe / "data/nlsyms.txt").write_text("<ara>\n<eng>\n")
    config = api.tiny_config(True, fixture_dir)
    model = config["model_conf"]
    model["pit_loss"] = model.pop("lidseq_order_insensitive_loss")
    model["pit_loss_reduction"] = model.pop("lidseq_order_insensitive_reduction")
    del config["token_list"]  # Stage 5, not the API fixture, owns the vocabulary.
    (recipe / "conf/train_cpu_smoke.yaml").write_text(yaml.safe_dump(config))
    config["optim_conf"]["lr"] = 5e-6
    (recipe / "conf/train_cpu_unfrozen_smoke.yaml").write_text(yaml.safe_dump(config))
    decode = dict(
        beam_size=2,
        nbest=1,
        ctc_weight=0.0,
        lm_weight=0.0,
        maxlenratio=-3.0,
        minlenratio=0.0,
        batch_size=1,
        num_workers=0,
    )
    (recipe / "conf/decode_cpu_smoke.yaml").write_text(yaml.safe_dump(decode))
    return recipe


def verify_case(recipe, feats_type, pretrained):
    import torch
    import yaml
    from espnet2.tasks.asr import ASRTask

    dumped = recipe / ("dump_" + feats_type) / feats_type
    stats = recipe / ("exp/stats_" + feats_type)
    exp = recipe / ("exp/asr_cpu_" + feats_type)
    for split in ("train", "valid", "test"):
        source = recipe / "data" / (split + "_smoke")
        durations = api.read_kv(source / "utt2dur")
        sample_counts = api.read_kv(source / "utt2num_samples")
        assert set(durations) == set(sample_counts) == set(api.read_kv(source / "text"))
        for uid, duration in durations.items():
            assert int(sample_counts[uid]) == int(float(duration) * 16000)
    for split in ("train", "valid"):
        expected = {split + "_1", split + "_2"}
        original = dumped / "org" / (split + "_smoke")
        filtered = dumped / (split + "_smoke")
        assert len(api.read_kv(original / "wav.scp")) == 4
        for name in ("wav.scp", "text", "utt2num_samples"):
            assert set(api.read_kv(filtered / name)) == expected, (filtered, name)
        for name in ("speech_shape", "text_shape.word"):
            assert set(api.read_kv(stats / split / name)) == expected
        assert (filtered / "feats_type").read_text().strip() == "raw"
        samples = api.read_kv(filtered / "utt2num_samples")
        assert [int(samples[uid]) for uid in sorted(samples)] == [1760, 1920]
        if feats_type == "raw_copy":
            assert api.read_kv(original / "wav.scp") == api.read_kv(
                recipe / "data" / (split + "_smoke") / "wav.scp"
            )
            sidecar = recipe / "data" / (split + "_smoke") / "utt2dur"
            assert sidecar.is_file(), "production data writer must emit utt2dur"
            assert set(api.read_kv(sidecar)) == set(api.read_kv(original / "wav.scp"))
    tokens = list(
        (recipe / "data/lidseq_token_list/word/tokens.txt").read_text().splitlines()
    )
    assert set(tokens) == set(api.TOKENS) and len(tokens) == len(api.TOKENS)
    assert tokens[0] == "<blank>" and tokens[1] == "<unk>"
    assert tokens[-1] == "<sos/eos>"
    saved = yaml.safe_load((exp / "config.yaml").read_text())
    assert saved["ngpu"] == 0 and saved["num_workers"] == 0
    assert (
        saved["max_epoch"] == saved["num_iters_per_epoch"] == saved["accum_grad"] == 1
    )
    assert saved["batch_size"] == 2 and saved["resume"] is False
    assert saved["frontend"] == "default"
    assert saved["encoder"] == saved["decoder"] == "transformer"
    assert saved["model_conf"]["pit_loss"] is True
    assert saved["model_conf"]["pit_loss_reduction"] == "min"
    assert saved["token_list"] == tokens
    assert saved["init_param"] == ([str(pretrained)] if pretrained else [])
    assert saved["freeze_param"] == []
    assert saved["optim_conf"]["lr"] == (5e-6 if pretrained else 0.001)
    # Load only this run's new synthetic checkpoints, never historical tensors.
    checkpoint = torch.load(
        exp / "checkpoint.pth", map_location="cpu", weights_only=False
    )
    states = list(checkpoint["optimizers"][0]["state"].values())
    assert states and {float(state["step"]) for state in states} == {1.0}
    assert all(torch.isfinite(state["exp_avg"]).all() for state in states)
    assert any(torch.count_nonzero(state["exp_avg"]) for state in states)
    model, _ = ASRTask.build_model_from_file(
        str(exp / "config.yaml"), str(exp / "1epoch.pth"), "cpu"
    )
    for key, value in model.state_dict().items():
        assert value.device.type == "cpu" and torch.isfinite(value).all()
        torch.testing.assert_close(value, checkpoint["model"][key], rtol=0, atol=0)
    decode = exp / "cpu_beam2/test_smoke"
    decode_log = (decode / "logdir/asr_inference.1.log").read_text()
    assert "Decoding device=cpu" in decode_log
    assert "Beam_search:" in decode_log
    assert "PermissionError" not in decode_log and "Traceback" not in decode_log
    if pretrained:
        parent_weights = torch.load(pretrained, map_location="cpu", weights_only=False)
        assert any(
            not torch.equal(value, parent_weights[key])
            for key, value in model.state_dict().items()
        ), "unfrozen phase must update its explicitly chosen parent weights"
    hypotheses = [
        line.split(maxsplit=1) for line in (decode / "text").read_text().splitlines()
    ]
    ids = [row[0] for row in hypotheses]
    assert len(ids) == 4 and set(ids) == set(
        api.read_kv(recipe / "data/test_smoke/text")
    )
    metrics = json.loads((decode / "lidseq_score.json").read_text())
    assert metrics["num_ref"] == metrics["num_hyp"] == 4
    assert metrics["num_missing_hyp"] == metrics["num_extra_hyp"] == 0
    assert len((decode / "lidseq_details.tsv").read_text().splitlines()) == 5
    assert not torch.cuda.is_initialized() and torch.get_num_threads() == 1
    return dict(
        feats_type=feats_type,
        phase="unfrozen" if pretrained else "frozen",
        initialization=str(pretrained) if pretrained else None,
        resume=False,
        base_learning_rate=saved["optim_conf"]["lr"],
        checkpoint=str(exp / "1epoch.pth"),
        train_valid_before_filter=4,
        train_valid_after_filter=2,
        optimizer_updates=1,
        optimizer_parameter_states=len(states),
        checkpoint_reload_exact=True,
        unordered_loss=True,
        decoded_utterances=4,
        scorer_complete=True,
        source_duration_sidecar=(recipe / "data/train_smoke/utt2dur").is_file(),
        gpu_initialized=False,
        cpu_threads=torch.get_num_threads(),
    )


def worker(work):
    sys.path.insert(0, str(ROOT))
    import torch
    import espnet2.tasks.asr

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    module = Path(espnet2.tasks.asr.__file__).resolve()
    assert module.is_relative_to(ROOT), module
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    report = dict(
        status="running",
        scope="synthetic CPU workflow; NOT historical reproduction",
        stages=[3, 4, 5, 10, 11, 12, "13-custom"],
        source_manifest_api="prepare_fleurs_cs_lid_data.write_data_dir",
        source_module=str(module),
        cases={},
    )
    env = api.offline_environment(work)
    try:
        recipe = setup(work)
        report["snapshot_sha256"] = {
            str(path.relative_to(work)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                recipe / "run.sh",
                work / "egs2/TEMPLATE/asr1/asr.sh",
                recipe.parent / "lid1/local/prepare_fleurs_cs_lid_data.py",
            )
        }
        pretrained = None
        for feats_type, mode in (("raw", "mixed"), ("raw_copy", "csall")):
            options = dict(
                stage=3,
                stop_stage=13,
                ngpu=0,
                num_nodes=1,
                nj=1,
                inference_nj=1,
                gpu_inference="false",
                train_mode=mode,
                training_phase="unfrozen" if pretrained else "frozen",
                resume="false",
                auto_training_budget="false",
                enforce_training_policy="false",
                require_cuda_visible_devices="false",
                train_batch_size=2,
                accum_grad=1,
                effective_batch_size=2,
                target_passes=3,
                budget_max_epoch=1,
                warmup_ratio=0.1,
                train_set="train_smoke",
                valid_set="valid_smoke",
                test_sets="test_smoke",
                cs_root=work / "fixtures",
                asr_config="conf/train_cpu_smoke.yaml",
                inference_config="conf/decode_cpu_smoke.yaml",
                inference_args="--num_workers 0",
                dumpdir="dump_" + feats_type,
                asr_tag="cpu_" + feats_type,
                asr_stats_dir="exp/stats_" + feats_type,
                inference_tag="cpu_beam2",
                inference_asr_model="1epoch.pth",
                min_wav_duration=0.105,
                max_wav_duration=0.125,
                feats_normalize="uttmvn",
                audio_format="wav",
                nlsyms_txt="data/nlsyms.txt",
                score_label_map="data/local/label_map.used.tsv",
                run_lidseq_scoring="true",
            )
            # Leave feats_type unset to test mixed/raw and CSall/raw_copy defaults.
            if pretrained:
                options["pretrained_model"] = pretrained
                options["asr_config"] = "conf/train_cpu_unfrozen_smoke.yaml"
            argv = ["bash", "./run.sh"]
            for key, value in options.items():
                argv.extend(["--" + key, str(value)])
            print(f"RUN {feats_type}: actual wrapper stages 3..13", flush=True)
            start = time.monotonic()
            log = work / (feats_type + ".log")
            run_command(argv, recipe, env, log)
            result = verify_case(recipe, feats_type, pretrained)
            result.update(seconds=round(time.monotonic() - start, 2), log=str(log))
            report["cases"][feats_type] = result
            api.write_json(work / "report.json", report)
            pretrained = Path(result["checkpoint"])
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        api.write_json(work / "report.json", report)
    print(f"PASS: {work / 'report.json'}", flush=True)


def run_workflow(work):
    work = work.resolve()
    if not work.is_relative_to(Path("/tmp")):
        raise ValueError("Use a new isolated work directory under /tmp")
    work.mkdir(parents=True, exist_ok=False)
    env = api.offline_environment(work)
    env["CS_FLEURS_ROOT"] = os.environ.get("CS_FLEURS_ROOT", str(work / "fixtures"))
    run_command(
        [sys.executable, SCRIPT, "--worker", work],
        ROOT,
        env,
        work / "workflow.log",
        timeout=600,
    )
    report = json.loads((work / "report.json").read_text())
    assert report["status"] == "passed"
    print(f"PASS: {work / 'report.json'}", flush=True)
    return report


@unittest.skipUnless(
    os.environ.get("CS_LID_ASR_WRAPPER_SMOKE") == "1",
    "offline ASR wrapper/template integration is opt-in",
)
class TestASRWrapperCPUWorkflow(unittest.TestCase):
    def test_workflow(self):
        parent = Path(tempfile.mkdtemp(prefix="asr-wrapper-test-"))
        run_workflow(parent / "workflow")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker.resolve())
    elif args.work_dir:
        run_workflow(args.work_dir)
    else:
        parser.error("--work-dir is required")


if __name__ == "__main__":
    main()
