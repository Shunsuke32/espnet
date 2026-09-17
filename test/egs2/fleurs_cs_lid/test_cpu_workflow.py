"""Offline CPU workflow smoke test, NOT numerical MMS experiment reproduction.

Run with the ESPnet Python environment active (pytest is not required)::

    python3 test/egs2/fleurs_cs_lid/test_cpu_workflow.py --work-dir /tmp/cs-lid-cpu

The output directory must not exist. Commands, logs, fixtures, checkpoints, and
report.json are retained there. Ordinary test discovery skips this expensive
integration test unless CS_LID_CPU_SMOKE=1. No pretrained weights are downloaded.
Only the frontend, model dimensions, and training budget are reduced; the three
selected recipe target/loss combinations (hard, KL, AAM+BCE) and their sampler
settings are retained. The excluded plain Linear+BCE head is not required.
"""

import argparse
import csv
import hashlib
import importlib
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
RECIPE = ROOT / "egs2/fleurs_cs_lid"
CONFIGS = (
    ("hard", "train_fleurs_lid_mms_ecapa.yaml", "lid", "aamsoftmax_sc_topk"),
    (
        "kl",
        "train_lidseq_mms_ecapa_softtarget.yaml",
        "lid_softlabel",
        "aamsoftmax_sc_topk_softtarget",
    ),
    (
        "aam_bce",
        "train_mms_ecapa_multilabel_aam_bce_posw50.yaml",
        "lid_multilabel",
        "arc_margin_subcenter_intertopk_multilabel_bce",
    ),
)
LANGUAGES = (
    ("ara", "ar_eg", "Arabic"),
    ("cmn", "cmn_hans_cn", "Chinese"),
    ("eng", "en_us", "English"),
    ("fra", "fr_fr", "French"),
    ("hin", "hi_in", "Hindi"),
    ("jpn", "ja_jp", "Japanese"),
    ("rus", "ru_ru", "Russian"),
)
CS_SUBSETS = ("read/test", "xtts/test1", "xtts/test2", "mms/test")
TEST_SETS = (
    ["test_fleurs_lid"]
    + ["test_cs_" + name.replace("/", "_") for name in CS_SUBSETS]
    + ["test_yodas_lidseq"]
)


def read_kv(path):
    rows = [line.split(maxsplit=1) for line in path.read_text().splitlines()]
    assert rows and all(len(row) == 2 for row in rows), path
    assert len({row[0] for row in rows}) == len(rows), path
    return dict(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def options(**kwargs):
    result = []
    for key, value in kwargs.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        result.append("--" + key)
        result.extend(
            str(item).lower() if isinstance(item, bool) else str(item)
            for item in values
        )
    return result


def provenance():
    """Fail before any training if an editable installation wins resolution."""
    modules = {}
    for name in (
        "espnet2",
        "espnet2.tasks.lid",
        "espnet2.lid.espnet_model",
        "espnet2.train.lid_trainer",
        "espnet2.train.preprocessor",
        "espnet2.spk.loss.aamsoftmax_subcenter_intertopk",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        assert path.is_relative_to(ROOT), (name, str(path), str(ROOT))
        modules[name] = str(path)
    return modules


def create_fixtures(work):
    import numpy as np
    import soundfile as sf

    prep = load_local_module(
        "cpu_smoke_fleurs_prep", RECIPE / "lid1/local/prepare_fleurs_cs_lid_data.py"
    )

    fixtures = work / "fixtures"
    fleurs, cs, yodas = [fixtures / name for name in ("fleurs", "cs-fleurs", "yodas")]
    for directory in (fleurs, cs, yodas / "metadata", yodas / "audio"):
        directory.mkdir(parents=True)

    def audio(path, index):
        # At most 1.125 s per file, with no codecs, pipes, or external audio reads.
        samples = 16000 + (index % 3) * 1000
        t = np.arange(samples, dtype=np.float64) / 16000
        signal = 0.12 * np.sin(2 * np.pi * (180 + index * 7) * t)
        signal += 0.02 * np.sin(2 * np.pi * 51 * t)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), signal.astype(np.float32), 16000, subtype="PCM_16")
        return samples / 16000

    for split in ("train", "dev", "test"):
        with (fleurs / (split + ".tsv")).open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("id", "path", "lang_id_name", "speaker", "duration"))
            for i, (_, raw, _) in enumerate(LANGUAGES):
                for j in range(2):
                    uid = f"{split}-{raw}-{j}"
                    wav = fleurs / "audio" / (uid + ".wav")
                    duration = audio(wav, 2 * i + j)
                    writer.writerow((uid, str(wav), raw, uid, duration))

    for subset in ("xtts/train", *CS_SUBSETS):
        directory = cs / subset
        directory.mkdir(parents=True)
        indices = list(range(2))
        if subset == "xtts/train":
            training, validation = [], []
            for candidate in range(10000):
                uid = prep.sanitize_id(f"cs_xtts_train_xtts-train-{candidate}")
                group = validation if prep.stable_fraction(uid) < 0.02 else training
                group.append(candidate)
                if len(training) >= 18 and len(validation) >= 2:
                    break
            assert len(validation) >= 2
            indices = training[:18] + validation[:2]
        with (directory / "metadata.jsonl").open("w") as handle:
            for index in indices:
                wav = directory / "audio" / f"{index}.wav"
                duration = audio(wav, index)
                record = dict(
                    id=subset.replace("/", "-") + f"-{index}",
                    file_name=f"audio/{index}.wav",
                    language="English + Arabic" if index % 2 else "Arabic + English",
                    duration=duration,
                )
                handle.write(json.dumps(record) + "\n")

    for lang, _, name in LANGUAGES:
        if lang == "eng":
            continue
        with (yodas / "metadata" / (lang + ".jsonl")).open("w") as handle:
            for index in range(8):
                uid = f"{lang}-video{index}"
                duration = audio(yodas / "audio" / (uid + ".wav"), index)
                record = dict(
                    id=uid + "_asr_000000000_000001000",
                    kaldi_uttid=f"{uid}_{lang}_000000000_{int(duration * 1000):09d}",
                    wav_path=uid + ".wav",
                    languages=["English", name] if index % 2 else [name, "English"],
                )
                handle.write(json.dumps(record) + "\n")
    return fleurs, cs, yodas


def smoke_config(source, destination):
    import yaml

    config = yaml.safe_load(source.read_text())
    config.update(
        frontend="default",
        frontend_conf=dict(fs=16000, n_fft=256, hop_length=160, n_mels=16),
        encoder_conf=dict(model_scale=2, ndim=16, output_size=24),
        projector_conf=dict(output_size=8),
        max_epoch=1,
        num_iters_per_epoch=1,
        batch_size=2,
        valid_batch_size=2,
        accum_grad=1,
        num_workers=0,
        use_amp=False,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        log_interval=1,
        keep_nbest_models=1,
        use_tensorboard=False,
    )
    destination.write_text(
        "# CPU SMOKE TEST ONLY: not MMS numerical experiment reproduction.\n"
        + yaml.safe_dump(config, sort_keys=False)
    )
    return config


class Workflow:
    def __init__(self, work):
        self.work = work.resolve()
        self.work.mkdir(parents=True, exist_ok=False)
        self.report = dict(
            scope="offline CPU workflow smoke only; not MMS numerical reproduction",
            checkout=str(ROOT),
            python=sys.executable,
            commands=[],
            cases={},
        )
        self.recipe = self.work / "egs2/fleurs_cs_lid/lid1"
        self.env = dict(os.environ)
        self.env.update(
            CS_FLEURS_ROOT=str(self.work / "fixtures/cs-fleurs"),
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONPATH=str(ROOT),
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HOME=str(self.work / "cache/huggingface"),
            TORCH_HOME=str(self.work / "cache/torch"),
            MPLCONFIGDIR=str(self.work / "cache/matplotlib"),
            NUMBA_CACHE_DIR=str(self.work / "cache/numba"),
            XDG_CACHE_HOME=str(self.work / "cache"),
        )

    def save_report(self):
        write_json(self.work / "report.json", self.report)

    def command(self, name, argv):
        log = self.work / (name + ".log")
        command = shlex.join([str(x) for x in argv])
        print(f"[{name}] {command}", flush=True)
        # Activate before every recipe shell, then pin the isolated source again.
        activate = Path(sys.prefix).parents[1] / "bin/activate"
        prefix = "set -euo pipefail\n"
        if activate.is_file():
            prefix += f"source {shlex.quote(str(activate))} {shlex.quote(sys.prefix)}\n"
        for key in (
            "CS_FLEURS_ROOT",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "PYTHONPATH",
        ):
            prefix += f"export {key}={shlex.quote(self.env[key])}\n"
        started = time.monotonic()
        with log.open("w") as handle:
            proc = subprocess.Popen(
                ["bash", "-c", prefix + "exec " + command],
                cwd=self.recipe,
                env=self.env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise AssertionError(f"{name} timed out; see {log}")
        self.report["commands"].append(
            dict(
                name=name,
                command=command,
                cwd=str(self.recipe),
                log=str(log),
                returncode=code,
                seconds=round(time.monotonic() - started, 2),
            )
        )
        self.save_report()
        assert code == 0, f"{log}\n" + "\n".join(log.read_text().splitlines()[-90:])

    def setup(self):
        os.environ.update(self.env)
        self.report["module_files"] = provenance()
        self.report["tools"] = {
            name: shutil.which(name)
            for name in ("bash", "perl", "flac", "ffmpeg", "sox")
        }
        for name in ("bash", "perl"):
            assert self.report["tools"][name], f"Required tool missing: {name}"
        for directory in ("lid1", "local", "evaluation"):
            shutil.copytree(
                RECIPE / directory,
                self.recipe.parent / directory,
                symlinks=True,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "dump", "exp", "data", "hub"
                ),
            )
        # Keep the original helper symlinks and relative layout. Never fabricate a
        # missing helper: a fresh checkout must actually supply all of them.
        (self.work / "egs2/TEMPLATE").symlink_to(ROOT / "egs2/TEMPLATE")
        (self.work / "tools").symlink_to(ROOT / "tools")
        for path in (
            "run.sh",
            "lid.sh",
            "local/path.sh",
            "utils/run.pl",
            "local/copy_data_dir.sh",
            "local/prepare_ood_test.sh",
            "local/prepare_ood_test.py",
            "local/perturb_lid_data_dir_speed.sh",
            "local/evaluate.sh",
            "../local/finalize_duration_view.py",
        ):
            assert (self.recipe / path).is_file(), f"Missing checkout helper: {path}"
        self.fleurs, self.cs, self.yodas = create_fixtures(self.work)
        self.env["CS_FLEURS_ROOT"] = str(self.cs)
        os.environ.update(self.env)
        self.command(
            "provenance",
            [
                sys.executable,
                "-c",
                "import pathlib,sys; import espnet2.tasks.lid as m; "
                f"assert pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path({str(ROOT)!r})); "
                "print(sys.executable); print(m.__file__)",
            ],
        )

    def verify_data(self):
        data = self.recipe / "data"
        assert set(read_kv(data / "train_fleurs_lid/lang2utt")) == {
            lang for lang, _, _ in LANGUAGES
        }
        nlsyms = (data / "nlsyms.txt").read_text().splitlines()
        assert "<eng>" in nlsyms and "<ara>" in nlsyms
        assert not any("_" in token or "-" in token for token in nlsyms)
        for suffix in ("fleurs_lid", "lidseq", "lid_pair_cs", "lidseq_csall_yodas"):
            train = read_kv(data / f"train_{suffix}/wav.scp")
            valid = read_kv(data / f"valid_{suffix}/wav.scp")
            assert set(train).isdisjoint(valid), suffix
        pairs = set(read_kv(data / "train_lid_pair_cs/utt2lang").values())
        assert "ara-eng" in pairs and "eng-ara" not in pairs
        sequences = read_kv(data / "train_lidseq/utt2langs")
        assert any(set(value.split()) == {"ara", "eng"} for value in sequences.values())
        yodas = read_kv(data / "test_yodas_lidseq/utt2langs")
        assert {tuple(value.split()) for value in yodas.values()} == {
            (lang, "eng") for lang, _, _ in LANGUAGES if lang != "eng"
        }
        self.report["fixture_sets"] = {
            path.parent.name: len(read_kv(path)) for path in data.glob("*/wav.scp")
        }

    def snapshot_prepared_data(self):
        data = self.recipe / "data"
        return {
            str(path.relative_to(data)): (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            )
            for path in data.rglob("*")
            if path.is_file()
        }

    def protect_prepared_data(self):
        data = self.recipe / "data"
        self.source_snapshot = self.snapshot_prepared_data()
        self.source_modes = {
            path: path.stat().st_mode & 0o777
            for path in [data, *data.rglob("*")]
            if not path.is_symlink()
        }
        for path, mode in self.source_modes.items():
            path.chmod(mode & ~0o222)
        self.report["prepared_data_readonly"] = True

    def verify_prepared_data_unchanged(self):
        unchanged = self.snapshot_prepared_data() == self.source_snapshot
        self.report["prepared_data_unchanged"] = unchanged
        assert unchanged, "Recipe modified its read-only source data"

    def prepare_yodas_fixture(self):
        module = load_local_module(
            "cpu_smoke_yodas_prep", RECIPE / "local/prepare_cs_yodas_views.py"
        )
        # The production CLI deliberately accepts only pinned official metadata.
        # Exercise its actual manifest APIs without spoofing hashes or provenance.
        try:
            module.verify_metadata(self.yodas / "metadata")
        except ValueError as error:
            assert "checksum mismatch" in str(error)
        else:
            raise AssertionError("Synthetic YODAS must not pass official checksums")
        examples = module.load_yodas_records(
            self.yodas / "metadata", self.yodas / "audio"
        )
        split = module.split_yodas(examples, 0.5, 0.25, "cs-yodas-v1")
        module.ensure_no_split_leak(split)
        data = self.recipe / "data"
        for name, items in split.items():
            if name != "test":
                items, dropped = module.duration_cap(items, 70.0)
                assert not dropped
            assert items
            module.write_data_dir(name + "_yodas_lidseq", items, data)
            if name != "test":
                existing = module.read_existing_data_dir(data / (name + "_lidseq"))
                module.write_data_dir(
                    name + "_lidseq_csall_yodas", existing + items, data
                )
        self.report["limitations"] = [
            "YODAS CLI requires official checksums: synthetic fixture uses the real "
            "load_yodas_records/split_yodas/duration_cap/write_data_dir APIs, not CLI main.",
            "TSVs are synthetic, not revision-verified FLEURS; no datasets or models downloaded.",
        ]

    def verify_checkpoint(self, tag, config, train_set):
        import numpy as np
        import soundfile as sf
        import torch
        import yaml
        from espnet2.tasks.lid import LIDTask

        exp = self.recipe / "exp" / ("lid_smoke_" + tag)
        saved = yaml.safe_load((exp / "config.yaml").read_text())
        for key in (
            "preprocessor",
            "loss",
            "loss_conf",
            "iterator_type",
            "valid_iterator_type",
            "batch_type",
            "drop_last_iter",
        ):
            assert saved[key] == config[key], (tag, key)
        assert saved["frontend"] == "default" and not saved["use_amp"]
        checkpoint = torch.load(
            exp / "checkpoint.pth", map_location="cpu", weights_only=False
        )
        steps = [
            float(state["step"])
            for state in checkpoint["optimizers"][0]["state"].values()
        ]
        assert steps and set(steps) == {1.0}, (tag, steps)
        for state in checkpoint["model"].values():
            assert torch.isfinite(state).all(), tag
        model, args = LIDTask.build_model_from_file(
            str(exp / "config.yaml"), str(exp / "1epoch.pth"), "cpu"
        )
        assert type(model).__module__ == "espnet2.lid.espnet_model"
        for key, state in model.state_dict().items():
            torch.testing.assert_close(state, checkpoint["model"][key], rtol=0, atol=0)
        model.eval()
        data_root = self.recipe / "dump" / ("raw_copy" if tag == "kl" else "raw")
        training_root = (
            data_root if tag == "kl" else self.recipe / "dump/old_duration/raw"
        )
        training_dirs = {}
        membership = {}
        for split, name in (
            ("train", train_set),
            ("valid", train_set.replace("train_", "valid_", 1)),
        ):
            inputs = {
                key: (self.recipe / path).resolve()
                for path, key, _ in saved[f"{split}_data_path_and_name_and_type"]
            }
            formatted = inputs["speech"].parent
            assert formatted == (training_root / name).resolve(), (tag, inputs)
            assert inputs["speech"] == formatted / "wav.scp"
            label = "utt2lang" if tag == "hard" else "utt2langs"
            assert inputs["lid_labels"] == formatted / label
            training_dirs[split] = formatted
            shape_files = saved[f"{split}_shape_file"]
            assert len(shape_files) == 1, (tag, shape_files)
            shapes = read_kv(self.recipe / shape_files[0])
            samples = read_kv(formatted / "utt2num_samples")
            assert set(shapes) == set(samples) == set(read_kv(formatted / "wav.scp"))
            assert all(16000 <= int(value) <= 18000 for value in samples.values())
            assert all(
                int(shapes[key].split(",")[0]) == int(value)
                for key, value in samples.items()
            )
            unfiltered = data_root / name
            input_wavs = read_kv(unfiltered / "wav.scp")
            input_samples = read_kv(unfiltered / "utt2num_samples")
            prepared_wavs = read_kv(self.recipe / "data" / name / "wav.scp")
            assert set(input_wavs) == set(input_samples) == set(prepared_wavs)
            boundary_ids = {uid for uid, n in input_samples.items() if int(n) == 16000}
            assert boundary_ids, (tag, "fixture must exercise the 1-second boundary")
            if tag == "kl":
                assert set(samples) == set(input_wavs)
                assert input_wavs == prepared_wavs
                durations = read_kv(unfiltered / "utt2dur")
                assert {uid: int(n) for uid, n in samples.items()} == {
                    uid: int(float(seconds) * 16000)
                    for uid, seconds in durations.items()
                }
            else:
                kept = {uid for uid, n in input_samples.items() if int(n) > 16000}
                assert set(samples) == kept and not boundary_ids.intersection(samples)
                assert read_kv(formatted / "wav.scp") == {
                    uid: input_wavs[uid] for uid in kept
                }
                assert not list(formatted.rglob("*.wav"))
                audit = json.loads((formatted / "duration_view.json").read_text())
                assert audit["mode"] == "raw"
                assert audit["num_input"] == len(input_wavs)
                assert audit["num_kept"] == len(kept)
                assert audit["num_dropped"] == len(boundary_ids)
            membership[split] = dict(
                prepared=len(prepared_wavs),
                training=len(samples),
                one_second_retained=len(boundary_ids.intersection(samples)),
            )
        source = training_dirs["train"]
        assert (self.recipe / saved["lang2utt"]).resolve() == source / "lang2utt"
        wavs = read_kv(source / "wav.scp")
        labels = read_kv(source / ("utt2lang" if tag == "hard" else "utt2langs"))
        preprocess = LIDTask.build_preprocess_fn(args, train=False)
        chosen = [
            next(
                uid
                for uid, value in labels.items()
                if len(value.split()) == 1 and "-" not in value
            ),
            next(
                uid
                for uid, value in labels.items()
                if "ara-eng" in value or len(value.split()) == 2
            ),
        ]
        batch = []
        for uid in chosen:
            speech, rate = sf.read(wavs[uid], dtype="float32")
            assert rate == 16000 and len(speech) <= 18000
            example = preprocess(uid, dict(speech=speech, lid_labels=labels[uid]))
            target = example["lid_labels"]
            if tag == "hard":
                assert target.size == 1
            else:
                count = len(labels[uid].split())
                assert np.count_nonzero(target) == count
                np.testing.assert_allclose(target.sum(), 1 if tag == "kl" else count)
            batch.append((uid, example))
        _, batch = LIDTask.build_collate_fn(args, train=False)(batch)
        with torch.inference_mode():
            loss, _, _ = model(**batch)
        assert torch.isfinite(loss), tag
        self.report["cases"][tag] = dict(
            optimizer_steps=1,
            checkpoint_reload=True,
            reload_loss=float(loss),
            sampler=saved["batch_type"],
            iterator=saved["iterator_type"],
            formatted_audio=True,
            collected_stats=True,
            model_file=str(exp / "1epoch.pth"),
            train_data_dir=str(source),
            valid_data_dir=str(training_dirs["valid"]),
            evaluation_data_dir=str(data_root),
            duration_membership=membership,
        )
        return exp, source, data_root

    def direct_training_diagnostic(self, tag, config, train_set):
        """Reach the task despite a wrapper failure, which still fails the test."""
        valid_set = train_set.replace("train_", "valid_", 1)
        data_root = self.recipe / "dump" / ("raw_copy" if tag == "kl" else "raw")
        for name in [train_set, valid_set, *TEST_SETS]:
            source = self.recipe / "data" / name
            dest = data_root / name
            self.command(
                f"{tag}_copy_{name}",
                [
                    "bash",
                    "utils/copy_data_dir.sh",
                    "--validate_opts",
                    "--non-print",
                    source,
                    dest,
                ],
            )
            for filename in ("utt2lang", "utt2langs", "lang2utt", "utt2category"):
                # A failed wrapper may leave a read-only output copied from the
                # protected source. Replace only this diagnostic destination.
                (dest / filename).unlink(missing_ok=True)
                shutil.copyfile(source / filename, dest / filename)
            (dest / "category2utt").unlink(missing_ok=True)
            shutil.copyfile(source / "lang2utt", dest / "category2utt")
            if tag == "kl":
                for filename in ("utt2dur", "utt2num_samples"):
                    (dest / filename).unlink(missing_ok=True)
                    shutil.copyfile(source / filename, dest / filename)
                continue
            self.command(
                f"{tag}_format_{name}",
                [
                    "bash",
                    "scripts/audio/format_wav_scp.sh",
                    "--nj",
                    "1",
                    "--cmd",
                    "utils/run.pl",
                    "--audio-format",
                    "wav",
                    "--fs",
                    "16k",
                    source / "wav.scp",
                    dest,
                ],
            )
        training_root = data_root
        if tag != "kl":
            training_root = self.recipe / "dump/old_duration/raw"
            for name in (train_set, valid_set):
                self.command(
                    f"{tag}_duration_{name}",
                    [
                        sys.executable,
                        "../local/finalize_duration_view.py",
                        "--input-dir",
                        data_root / name,
                        "--source-data-dir",
                        self.recipe / "data" / name,
                        "--mode",
                        "raw",
                        "--output-dir",
                        training_root / name,
                        "--reuse-existing",
                    ],
                )
        train = training_root / train_set
        valid = training_root / valid_set
        stats = self.recipe / "exp" / ("stats_" + tag)
        common = options(
            config=config,
            ngpu=0,
            lang2utt=train / "lang2utt",
            lang_num=len(read_kv(train / "lang2utt")),
        )
        self.command(
            tag + "_collect_stats_direct",
            [
                sys.executable,
                "-m",
                "espnet2.bin.lid_train",
                *common,
                *options(
                    collect_stats=True,
                    use_preprocessor=False,
                    train_data_path_and_name_and_type=f"{train}/wav.scp,speech,sound",
                    valid_data_path_and_name_and_type=f"{valid}/wav.scp,speech,sound",
                    train_shape_file=train / "wav.scp",
                    valid_shape_file=valid / "wav.scp",
                    output_dir=stats / "logdir/stats.1",
                ),
            ],
        )
        self.command(
            tag + "_aggregate_stats_direct",
            [
                sys.executable,
                "-m",
                "espnet2.bin.aggregate_stats_dirs",
                *options(input_dir=stats / "logdir/stats.1", output_dir=stats),
                "--skip_sum_stats",
            ],
        )
        label = "utt2lang" if tag == "hard" else "utt2langs"
        self.command(
            tag + "_train_direct",
            [
                sys.executable,
                "-m",
                "espnet2.bin.lid_train",
                *common,
                *options(
                    use_preprocessor=True,
                    train_data_path_and_name_and_type=f"{train}/wav.scp,speech,sound",
                    valid_data_path_and_name_and_type=f"{valid}/wav.scp,speech,sound",
                    train_shape_file=stats / "train/speech_shape",
                    valid_shape_file=stats / "valid/speech_shape",
                    output_dir=self.recipe / "exp" / ("lid_smoke_" + tag),
                ),
                *options(
                    train_data_path_and_name_and_type=f"{train}/{label},lid_labels,text",
                    valid_data_path_and_name_and_type=f"{valid}/{label},lid_labels,text",
                ),
            ],
        )

    def verify_single_pair_class(self):
        import soundfile as sf
        import torch
        from espnet2.samplers.category_balanced_sampler import CategoryBalancedSampler
        from espnet2.tasks.lid import LIDTask

        inventory = self.work / "fixtures/single_pair.lang2utt"
        inventory.write_text("ara-eng pair0 pair1 pair2 pair3\n")
        args = LIDTask.get_parser().parse_args(
            [
                "--config",
                str(self.recipe / "conf/smoke_cpu_hard.yaml"),
                "--lang2utt",
                str(inventory),
                "--lang_num",
                "1",
                "--use_preprocessor",
                "true",
            ]
        )
        # A single class has no negatives. Preserve production k_top=5 in all
        # three real training runs; disable it only for this explicit edge case.
        args.loss_conf["k_top"] = 0
        model = LIDTask.build_model(args)
        preprocessor = LIDTask.build_preprocess_fn(args, train=True)
        sampler = CategoryBalancedSampler(
            batch_size=2, category2utt_file=str(inventory)
        )
        assert len(sampler) == 2
        examples = []
        for index, uid in enumerate(next(iter(sampler))):
            speech, _ = sf.read(
                self.fleurs / f"audio/train-ar_eg-{index}.wav", dtype="float32"
            )
            example = preprocessor(uid, dict(speech=speech, lid_labels="ara-eng"))
            assert example["lid_labels"].tolist() == [0]
            examples.append((uid, example))
        _, batch = LIDTask.build_collate_fn(args, train=True)(examples)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss, _, _ = model(**batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            _, prediction = model(
                speech=batch["speech"][:1],
                speech_lengths=batch["speech_lengths"][:1],
                extract_embd=True,
            )
        assert prediction.tolist() == [0]
        self.report["single_pair_class"] = dict(
            label="ara-eng",
            category_batches=2,
            loss=float(loss.detach()),
            k_top_override=0,
            official_task_forward_backward=True,
        )

    def evaluate(self, tag, exp, train_data, evaluation_data):
        import numpy as np

        for name in TEST_SETS:
            assert set(read_kv(evaluation_data / name / "wav.scp")) == set(
                read_kv(self.recipe / "data" / name / "wav.scp")
            ), (tag, name, "evaluation membership must stay unfiltered")
        out = self.recipe / "exp" / ("eval_" + tag)
        argv = [
            "bash",
            "local/evaluate.sh",
            "--model_file",
            exp / ("checkpoint.pth" if tag == "aam_bce" else "1epoch.pth"),
            "--config_file",
            exp / "config.yaml",
            "--train_data_dir",
            train_data,
            "--data_dir",
            evaluation_data,
            "--test_sets",
            *TEST_SETS,
            "--output_dir",
            out,
            "--ngpu",
            "0",
            "--batch_size",
            "2",
            "--num_workers",
            "0",
            "--thresholds",
            "0",
            "0.5",
            "1",
            "--logit_thresholds",
            "0",
            "15",
            "30",
            "--no_plots",
        ]
        if tag == "aam_bce":
            # This exact trainer checkpoint was created by this test invocation.
            argv.append("--trusted_checkpoint")
        self.command(tag + "_evaluate", argv)
        scores = out / "scores.npz"
        digest = hashlib.sha256(scores.read_bytes()).hexdigest()
        before = scores.stat().st_mtime_ns
        with np.load(scores, allow_pickle=False) as archive:
            logits = archive["logits"]
            assert logits.shape == (len(archive["utt_ids"]), len(archive["labels"]))
            assert np.isfinite(logits).all()
            assert len(set(archive["utt_ids"].tolist())) == len(logits)
            if tag in ("hard", "kl"):
                np.testing.assert_allclose(archive["probabilities"].sum(axis=1), 1)
        self.command(tag + "_score_only", [*argv, "--score_only"])
        assert hashlib.sha256(scores.read_bytes()).hexdigest() == digest
        assert scores.stat().st_mtime_ns == before
        evaluation = json.loads((out / "evaluation.json").read_text())
        assert set(TEST_SETS + ["test_cs_all"]) == set(evaluation["sets"])
        for name in TEST_SETS:
            details = list(
                csv.DictReader((out / (name + ".details.tsv")).open(), delimiter="\t")
            )
            assert len(details) == evaluation["sets"][name]["n"]
            if tag != "hard":
                assert all(
                    len(row["pred"].split()) == int(row["reference_cardinality"])
                    for row in details
                )
                with (out / (name + ".threshold_sweep.tsv")).open() as handle:
                    sweep = list(csv.DictReader(handle, delimiter="\t"))
                domains = {row["score_type"] for row in sweep}
                assert domains == ({"softmax", "logit"} if tag == "kl" else {"sigmoid"})
                assert all(0 <= float(row["accuracy"]) <= 1 for row in sweep)
        self.report["cases"][tag].update(saved_logits=True, score_only_reuse=True)

    def run(self):
        previous_cwd = Path.cwd()
        previous_env = dict(os.environ)
        try:
            self.setup()
            os.chdir(self.recipe)
            preparation = [
                "bash",
                "run.sh",
                "--profile",
                "mixed",
                "--stage",
                "1",
                "--stop_stage",
                "1",
                "--ngpu",
                "0",
                "--auto_training_budget",
                "false",
                "--enforce_training_policy",
                "false",
                "--require_cuda_visible_devices",
                "false",
                "--skip_fleurs_download",
                "true",
                "--fleurs_tsv_root",
                self.fleurs,
                "--cs_root",
                self.cs,
                "--test_sets",
                " ".join(TEST_SETS[:-1]),
            ]
            workflow_errors = []
            try:
                self.command("prepare_fleurs_xtts", preparation)
            except AssertionError as error:
                # Continue useful integration coverage, but never turn a failed
                # top-level wrapper into a successful smoke-test result.
                workflow_errors.append(str(error))
                self.report["preparation_wrapper_error"] = str(error)
                self.command(
                    "prepare_fleurs_xtts_direct",
                    [
                        "bash",
                        "local/data.sh",
                        "--skip_fleurs_download",
                        "true",
                        "--fleurs_tsv_root",
                        self.fleurs,
                        "--cs_root",
                        self.cs,
                        "--duration_missing_policy",
                        "error",
                    ],
                )
            self.prepare_yodas_fixture()
            self.verify_data()
            self.protect_prepared_data()
            for tag, filename, preprocessor, loss in CONFIGS:
                smoke = self.recipe / "conf" / ("smoke_cpu_" + tag + ".yaml")
                config = smoke_config(self.recipe / "conf" / filename, smoke)
                assert (config["preprocessor"], config["loss"]) == (preprocessor, loss)
                expected = (
                    ("sequence", "sorted") if tag == "kl" else ("category", "catbel")
                )
                assert (config["iterator_type"], config["batch_type"]) == expected
                train_set = "train_lid_pair_cs" if tag == "hard" else "train_lidseq"
                stages = [
                    "bash",
                    "run.sh",
                    "--profile",
                    "mixed",
                    "--stage",
                    "3",
                    "--stop_stage",
                    "5",
                    "--lid_config",
                    smoke,
                    "--lid_tag",
                    "smoke_" + tag,
                    "--lid_stats_dir",
                    "exp/stats_" + tag,
                    "--auto_training_budget",
                    "false",
                    "--enforce_training_policy",
                    "false",
                    "--train_batch_size",
                    "2",
                    "--accum_grad",
                    "1",
                    "--effective_batch_size",
                    "2",
                    "--ngpu",
                    "0",
                    "--require_cuda_visible_devices",
                    "false",
                    "--gpu_inference",
                    "false",
                    "--nj",
                    "1",
                    "--skip_fleurs_download",
                    "true",
                    "--fleurs_tsv_root",
                    self.fleurs,
                    "--cs_root",
                    self.cs,
                    "--cs_dev_ratio",
                    "0.02",
                    "--test_sets",
                    " ".join(TEST_SETS),
                ]
                try:
                    self.command(tag + "_stages", stages)
                except AssertionError as error:
                    workflow_errors.append(f"{tag}: {error}")
                    self.report.setdefault("training_wrapper_errors", {})[tag] = str(
                        error
                    )
                    self.direct_training_diagnostic(tag, smoke, train_set)
                exp, train_data, evaluation_data = self.verify_checkpoint(
                    tag, config, train_set
                )
                try:
                    self.evaluate(tag, exp, train_data, evaluation_data)
                except AssertionError as error:
                    workflow_errors.append(f"{tag}: {error}")
                    self.report["cases"][tag]["evaluation_error"] = str(error)
                self.verify_prepared_data_unchanged()
                self.save_report()
            self.verify_single_pair_class()
            assert not workflow_errors, "\n".join(workflow_errors)
            self.report["status"] = "passed"
        except Exception as error:
            self.report["status"] = "failed"
            self.report["error"] = str(error)
            raise
        finally:
            if hasattr(self, "source_snapshot"):
                self.report["prepared_data_unchanged"] = (
                    self.snapshot_prepared_data() == self.source_snapshot
                )
            for path, mode in getattr(self, "source_modes", {}).items():
                path.chmod(mode)
            os.chdir(previous_cwd)
            os.environ.clear()
            os.environ.update(previous_env)
            self.save_report()


@unittest.skipUnless(
    os.environ.get("CS_LID_CPU_SMOKE") == "1", "set CS_LID_CPU_SMOKE=1"
)
class TestCPUWorkflow(unittest.TestCase):
    def test_fresh_workspace(self):
        with tempfile.TemporaryDirectory(prefix="cs-lid-cpu-test-") as parent:
            Workflow(Path(parent) / "workspace").run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    Workflow(args.work_dir).run()
    print(f"PASS: {args.work_dir / 'report.json'}")
