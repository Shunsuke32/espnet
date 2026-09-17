"""CPU-only coverage of score extraction, cache reuse, and evaluation contracts."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml


RECIPE = Path(__file__).resolve().parents[3] / "egs2" / "fleurs_cs_lid"
LOCAL = RECIPE / "lid1" / "local"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.syspath_prepend(str(LOCAL))
    return load_module("test_evaluate_lid", LOCAL / "evaluate_lid.py")


def manifest(directory, refs, wavs=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "utt2langs").write_text(
        "".join(f"{utt} {ref}\n" for utt, ref in refs.items())
    )
    if wavs is not None:
        (directory / "wav.scp").write_text(
            "".join(f"{utt} {wav}\n" for utt, wav in wavs.items())
        )


def setup_run(tmp_path, evaluator, loss="aamsoftmax_sc_topk_softtarget"):
    train = tmp_path / "train"
    manifest(train, {"t1": "eng", "t2": "eng ara"})
    (train / "lang2utt").write_text("ara t2\neng t1 t2\njpn t3\n")
    model = tmp_path / "explicit-model.pth"
    model.write_bytes(b"dummy model; inference is injected")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            dict(loss=loss, loss_conf=dict(scale=30, K=3, k_top=0), lang_num=3)
        )
    )
    data = tmp_path / "data"
    manifest(data / "test_fleurs_lid", {"f1": "eng"}, {"f1": "/audio/f1.wav"})
    for i, name in enumerate(evaluator.CS_SUBSETS):
        manifest(data / name, {f"c{i}": "eng ara"}, {f"c{i}": f"/audio/c{i}.wav"})
    manifest(
        data / "test_cs_all",
        {f"c{i}": "ara eng" for i in range(4)},
        {f"c{i}": f"/audio/c{i}.wav" for i in range(4)},
    )
    return evaluator.get_parser().parse_args(
        [
            "--model_file",
            str(model),
            "--config_file",
            str(config),
            "--train_data_dir",
            str(train),
            "--data_dir",
            str(data),
            "--test_sets",
            "test_fleurs_lid test_cs_all",
            "--output_dir",
            str(tmp_path / "output"),
            "--thresholds",
            "0",
            "0.3",
            "0.5",
            "1",
            "--logit_thresholds",
            "0",
            "15",
            "30",
            "--no_plots",
        ]
    )


def fake_inference(args, labels, head, wav_scp):
    ids = [line.split()[0] for line in wav_scp.read_text().splitlines()]
    assert len(ids) == len(set(ids)) == 5
    # Deliberately return a different order from both reference and input files.
    ids.reverse()
    logits = np.asarray(
        [[0, 4, -5] if utt == "f1" else [4, 3, -5] for utt in ids], dtype=np.float32
    )
    return ids, logits, "injected_cpu_model"


def test_end_to_end_cached_cli_and_cs_union(tmp_path, evaluator):
    args = setup_run(tmp_path, evaluator)
    results = evaluator.run(args, fake_inference)
    assert len(results) == 6
    assert all(r["accuracy"] == 1 for r in results)
    aggregate = next(r for r in results if r["set"] == "test_cs_all")
    assert aggregate["num_ref"] == 4
    seen = next(r for r in aggregate["groups"] if r["group_type"] == "pair_seen_status")
    assert seen["group"] == "seen" and seen["n"] == 4
    assert (args.output_dir / "inputs" / "lang2utt").read_bytes() == (
        args.train_data_dir / "lang2utt"
    ).read_bytes()
    with np.load(args.output_dir / "scores.npz", allow_pickle=False) as archive:
        assert archive["logits"].shape == (5, 3)
        assert archive["probabilities"].dtype == np.float64
        metadata = json.loads(str(archive["metadata"]))
        assert metadata["provenance"]["head"]["scale"] == 30
        assert metadata["provenance"]["head"]["activation"] == "softmax"
    subprocess.run(
        [
            "bash",
            str(LOCAL / "evaluate.sh"),
            "--model_file",
            str(args.model_file),
            "--config_file",
            str(args.config_file),
            "--train_data_dir",
            str(args.train_data_dir),
            "--data_dir",
            str(args.data_dir),
            "--test_sets",
            "test_fleurs_lid",
            "test_cs_all",
            "--output_dir",
            str(args.output_dir),
            "--score_only",
            "--thresholds",
            "0.1",
            "0.3",
            "--logit_thresholds",
            "0",
            "30",
            "--no_plots",
        ],
        check=True,
    )


def test_sigmoid_workflow(tmp_path, evaluator):
    args = setup_run(
        tmp_path, evaluator, "arc_margin_subcenter_intertopk_multilabel_bce"
    )
    results = evaluator.run(args, fake_inference)
    assert all(r["head"]["activation"] == "sigmoid" for r in results)
    assert all(r["accuracy"] == 1 for r in results)
    assert {row["score_type"] for row in results[0]["threshold_sweep"]} == {"sigmoid"}


def test_plain_bce_rejected_before_inference(tmp_path, evaluator):
    args = setup_run(tmp_path, evaluator, "multilabel_bce")
    with pytest.raises(ValueError, match="^unsupported LID loss: multilabel_bce$"):
        evaluator.run(args, lambda *_: pytest.fail("unexpected inference"))
    assert not args.output_dir.exists()


def test_atomic_pairs_always_top1_expanded_and_monolingual_top2(tmp_path, evaluator):
    metrics = sys.modules["lid_metrics"]
    labels = ["ara-eng", "eng", "jpn"]
    head = evaluator.head_spec(dict(loss="aamsoftmax_sc_topk", lang_num=3), labels)
    logits = np.asarray([[5, 4, 1], [5, 4, 1]], dtype=np.float32)
    result = metrics.evaluate_set(
        tmp_path,
        "atomic",
        labels,
        ["pair", "single"],
        logits,
        metrics.probabilities(logits, "softmax"),
        {"pair": ("ara", "eng"), "single": ("eng",)},
        set(),
        head,
    )
    assert result["accuracy"] == 0.5
    assert result["metric"] == "atomic_top1_expanded"
    assert "threshold_sweep" not in result
    labels = ["ara", "eng", "jpn"]
    head = evaluator.head_spec(dict(loss="aamsoftmax_sc_topk", lang_num=3), labels)
    result = metrics.evaluate_set(
        tmp_path,
        "mono",
        labels,
        ["pair"],
        logits[:1],
        metrics.probabilities(logits[:1], "softmax"),
        {"pair": ("ara", "eng")},
        set(),
        head,
    )
    assert result["accuracy"] == 1


def test_actual_pair_training_manifest_end_to_end(tmp_path, evaluator):
    prep = load_module(
        "atomic_inventory_data_prep", LOCAL / "prepare_fleurs_cs_lid_data.py"
    )
    args = setup_run(tmp_path, evaluator, "aamsoftmax_sc_topk")
    examples = [
        prep.Example(
            uttid=utt,
            wav=f"/audio/{utt}.wav",
            labels=labels,
            speaker=utt,
            source="cs_fleurs" if len(labels) == 2 else "fleurs",
            subset="train",
        )
        for utt, labels in [("t1", ("eng",)), ("t2", ("eng", "ara")), ("t3", ("jpn",))]
    ]
    prep.write_data_dir(
        "train_lid_pair_cs",
        prep.to_pair_class_examples(examples),
        args.data_dir,
        "angle",
    )
    args.train_data_dir = args.data_dir / "train_lid_pair_cs"
    atomic_refs = args.train_data_dir / "utt2langs"
    assert "t2 ara-eng\n" in atomic_refs.read_text()
    labels, _, _, seen_pairs = evaluator.training_inventory(
        args, yaml.safe_load(args.config_file.read_text())
    )
    assert labels == ["ara-eng", "eng", "jpn"]
    assert seen_pairs == {("ara", "eng")}
    # Individual languages being trained does not make their novel pair seen.
    (args.data_dir / evaluator.CS_SUBSETS[-1] / "utt2langs").write_text("c3 eng jpn\n")
    (args.data_dir / "test_cs_all" / "utt2langs").write_text(
        "c0 ara eng\nc1 eng ara\nc2 ara eng\nc3 eng jpn\n"
    )
    results = evaluator.run(args, fake_inference)
    aggregate = next(row for row in results if row["set"] == "test_cs_all")
    groups = {
        row["group"]: row
        for row in aggregate["groups"]
        if row["group_type"] == "pair_seen_status"
    }
    assert groups["seen"]["n"] == 3 and groups["seen"]["correct"] == 3
    assert groups["unseen"]["n"] == 1 and groups["unseen"]["correct"] == 0
    assert aggregate["metric"] == "atomic_top1_expanded"
    assert (
        args.output_dir / "inputs" / "train.utt2langs"
    ).read_bytes() == atomic_refs.read_bytes()
    args.score_only = True
    assert evaluator.run(args) == results
    # The identical atomic manifest is not allowed as ordinary test references.
    with pytest.raises(ValueError, match="canonical language labels"):
        evaluator.read_refs(atomic_refs)


@pytest.mark.parametrize(
    "reference", ["ara-ara", "ara-eng-jpn", "ara-eng jpn", "ara--eng", "en_us-eng"]
)
def test_malformed_atomic_training_references_fail(tmp_path, evaluator, reference):
    args = setup_run(tmp_path, evaluator)
    (args.train_data_dir / "utt2langs").write_text(f"t1 {reference}\n")
    with pytest.raises(ValueError):
        evaluator.training_inventory(args, yaml.safe_load(args.config_file.read_text()))


def test_saturated_sigmoid_and_ties_rank_logits(tmp_path, evaluator):
    metrics = sys.modules["lid_metrics"]
    labels = ["ara", "eng", "jpn"]
    head = evaluator.head_spec(
        dict(
            loss="arc_margin_subcenter_intertopk_multilabel_bce",
            loss_conf=dict(scale=50, K=3, k_top=0),
            lang_num=3,
        ),
        labels,
    )
    logits = np.asarray([[40, 42, 41], [2, 2, 2]], dtype=np.float32)
    probs = metrics.probabilities(logits, "sigmoid")
    assert np.all(probs[0] == 1)
    result = metrics.evaluate_set(
        tmp_path,
        "saturation",
        labels,
        ["u1", "u2"],
        logits,
        probs,
        {"u1": ("eng", "jpn"), "u2": ("ara",)},
        set(),
        head,
        [0.5, 1],
    )
    assert result["accuracy"] == 1


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "duplicate",
        "extra",
        "duplicate_across_sets",
        "aggregate_labels",
    ],
)
def test_bad_manifests_fatal_before_inference(tmp_path, evaluator, damage):
    args = setup_run(tmp_path, evaluator)
    directory = args.data_dir / "test_fleurs_lid"
    if damage == "missing":
        (directory / "wav.scp").write_text("")
    elif damage == "duplicate":
        (directory / "utt2langs").write_text("f1 eng\nf1 eng\n")
    elif damage == "extra":
        (directory / "wav.scp").write_text("f1 /audio/f1.wav\nextra /audio/extra.wav\n")
    elif damage == "duplicate_across_sets":
        manifest(
            args.data_dir / evaluator.CS_SUBSETS[0],
            {"f1": "eng"},
            {"f1": "/audio/f1.wav"},
        )
    elif damage == "aggregate_labels":
        (args.data_dir / "test_cs_all" / "utt2langs").write_text(
            "".join(f"c{i} eng jpn\n" for i in range(4))
        )

    def forbidden(*_):
        pytest.fail("inference should not run")

    with pytest.raises(ValueError):
        evaluator.run(args, forbidden)


def test_explicit_exclusion_removes_reference_and_audio(tmp_path, evaluator):
    args = setup_run(tmp_path, evaluator)
    directory = args.data_dir / "test_fleurs_lid"
    (directory / "utt2langs").write_text("f1 eng\nexcluded ara\nexcluded ara\n")
    args.exclusions = tmp_path / "excluded.tsv"
    args.exclusions.write_text("utt\treason\nexcluded\tmissing source audio\n")
    evaluator.run(args, fake_inference)
    report = json.loads((args.output_dir / "evaluation.json").read_text())
    assert report["exclusions"] == {"excluded": "missing source audio"}


@pytest.mark.parametrize(
    "damage", ["model", "labels", "logits", "probabilities", "ids"]
)
def test_cache_tampering_or_inventory_changes_fail(tmp_path, evaluator, damage):
    args = setup_run(tmp_path, evaluator)
    evaluator.run(args, fake_inference)
    if damage == "model":
        args.model_file.write_bytes(b"different model")
    elif damage == "labels":
        (args.train_data_dir / "lang2utt").write_text("eng t1\nara t2\njpn t3\n")
    else:
        path = args.output_dir / "scores.npz"
        with np.load(path, allow_pickle=False) as archive:
            values = dict(archive)
        if damage == "logits":
            values["logits"][0, 0] = np.nan
        elif damage == "probabilities":
            values["probabilities"][0, 0] = 0.123
        else:
            values["utt_ids"][0] = values["utt_ids"][1]
        np.savez_compressed(path, **values)
    args.score_only = True
    with pytest.raises(ValueError):
        evaluator.run(args)


@pytest.mark.parametrize("name", ["hard", "kl", "aam_bce"])
def test_real_cpu_head_logits(evaluator, name):
    from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
        ArcMarginProduct_intertopk_subcenter as Hard,
        ArcMarginProduct_intertopk_subcenter_softtarget as KL,
        ArcMarginProduct_intertopk_subcenter_multilabel_bce as BCE,
    )

    classes = dict(hard=Hard, kl=KL, aam_bce=BCE)
    names = dict(
        hard="aamsoftmax_sc_topk",
        kl="aamsoftmax_sc_topk_softtarget",
        aam_bce="arc_margin_subcenter_intertopk_multilabel_bce",
    )
    conf = dict(scale=30, K=3, k_top=0)
    loss = classes[name](nout=4, nclasses=3, **conf)
    embeddings = torch.tensor([[1.0, 2, -3, 4], [-2.0, 3, 1, 4]])
    model = types.SimpleNamespace(
        loss=loss,
        extract_feats=lambda speech, lengths: (speech, lengths),
        encode_frame=lambda x: x,
        pooling=lambda x, feat_lengths: x,
        project_lang_embd=lambda x: x,
    )
    head = evaluator.head_spec(
        dict(loss=names[name], loss_conf=conf, lang_num=3), ["ara", "eng", "jpn"]
    )
    actual = evaluator.model_logits(model, embeddings, torch.tensor([4, 4]), head, 3)
    expected = (
        30
        * F.linear(F.normalize(embeddings), F.normalize(loss.weight))
        .reshape(2, 3, 3)
        .max(dim=2)
        .values
    )
    torch.testing.assert_close(actual, expected)


def test_checkpoint_model_extraction_and_strict_load(tmp_path, evaluator, monkeypatch):
    original = torch.nn.Linear(3, 2)
    model_file = tmp_path / "checkpoint.pth"
    torch.save(
        dict(model=original.state_dict(), reporter={}, optimizers=[]), model_file
    )
    fake_task = types.ModuleType("espnet2.tasks.lid")
    fake_task.LIDTask = types.SimpleNamespace(
        build_model_from_file=lambda *args: (torch.nn.Linear(3, 2), None)
    )
    monkeypatch.setitem(sys.modules, "espnet2.tasks.lid", fake_task)
    loaded, _, kind = evaluator.load_model(tmp_path / "config.yaml", model_file, "cpu")
    assert kind == "trainer_checkpoint.model"
    torch.testing.assert_close(loaded.weight, original.weight)
    torch.testing.assert_close(loaded.bias, original.bias)
    torch.save(dict(model={"weight": original.weight}), model_file)
    with pytest.raises(RuntimeError, match="Missing key"):
        evaluator.load_model(tmp_path / "config.yaml", model_file, "cpu")


def test_plot_uses_fleurs_top1_baseline(tmp_path, evaluator, monkeypatch):
    metrics = sys.modules["lid_metrics"]
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    observed = []
    original = Axes.axhline

    def capture(self, y=0, *args, **kwargs):
        observed.append(y)
        return original(self, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "axhline", capture)
    args = setup_run(tmp_path, evaluator)
    results = evaluator.run(args, fake_inference)
    metrics.plot_results(args.output_dir, results)
    assert observed and all(y == 100 for y in observed)
    from PIL import Image

    with Image.open(args.output_dir / "softmax_threshold_sweep.png") as image:
        pixels = np.asarray(image.convert("RGB"))
        assert pixels.shape[0] > 100 and pixels.std() > 5


def test_unseen_pair_is_not_seen_from_individual_languages(tmp_path, evaluator):
    metrics = sys.modules["lid_metrics"]
    rows = metrics.summarize({"u": 1}, {"u": ("eng", "jpn")}, {("ara", "eng")})
    status = next(row for row in rows if row["group_type"] == "pair_seen_status")
    assert status["group"] == "unseen"


def test_cli_help_is_model_import_free():
    result = subprocess.run(
        [sys.executable, str(LOCAL / "evaluate_lid.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--train_data_dir" in result.stdout and "--score_only" in result.stdout


def test_score_only_never_infers_without_cache(tmp_path, evaluator):
    args = setup_run(tmp_path, evaluator)
    args.score_only = True
    with pytest.raises(ValueError, match="no saved scores"):
        evaluator.run(args, lambda *_: pytest.fail("unexpected inference"))


def test_actual_inference_loop_cpu_and_alignment(tmp_path, evaluator, monkeypatch):
    from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
        ArcMarginProduct_intertopk_subcenter_multilabel_bce as BCE,
    )
    import lid_topk_softmax

    args = setup_run(
        tmp_path, evaluator, "arc_margin_subcenter_intertopk_multilabel_bce"
    )
    head = BCE(nout=3, nclasses=3, scale=30, K=3, k_top=0)
    with torch.no_grad():
        head.weight.copy_(torch.eye(3).repeat_interleave(3, dim=0))
    model = types.SimpleNamespace(
        loss=head,
        extract_feats=lambda x, lengths: (x, lengths),
        encode_frame=lambda x: x,
        pooling=lambda x, feat_lengths: x,
        project_lang_embd=lambda x: x,
    )
    monkeypatch.setattr(
        evaluator,
        "load_model",
        lambda *_, **__: (model, types.SimpleNamespace(), "tiny_cpu_head"),
    )

    def iterator(iterator_args, _):
        ids = [
            line.split()[0] for line in iterator_args.wav_scp.read_text().splitlines()
        ]
        for start in range(0, len(ids), 2):
            batch_ids = ids[start : start + 2]
            speech = torch.tensor(
                [[0.0, 4, -5] if utt == "f1" else [4.0, 3, -5] for utt in batch_ids]
            )
            yield (
                batch_ids,
                dict(speech=speech, speech_lengths=torch.tensor([3] * len(batch_ids))),
            )

    monkeypatch.setattr(lid_topk_softmax, "build_iterator", iterator)
    results = evaluator.run(args)
    assert all(result["accuracy"] == 1 for result in results)


def test_frozen_experiment_inventory_order_is_authoritative(tmp_path, evaluator):
    args = setup_run(tmp_path, evaluator)
    (args.config_file.parent / "lang2utt").write_text("eng t1\nara t2\njpn t3\n")
    with pytest.raises(ValueError, match="class order differs"):
        evaluator.run(args)


def test_aggregate_is_a_view_not_a_second_formatted_audio_input(tmp_path, evaluator):
    args = setup_run(tmp_path, evaluator)
    (args.data_dir / "test_cs_all" / "wav.scp").write_text(
        "".join(f"c{i} /separately-formatted-aggregate/{i}.flac\n" for i in range(4))
    )
    results = evaluator.run(args, fake_inference)
    assert next(r for r in results if r["set"] == "test_cs_all")["num_ref"] == 4


@pytest.mark.parametrize(
    "labels", [["eng", "eng"], ["eng-ara", "ara-eng"], ["eng", "en_us"]]
)
def test_invalid_head_inventory(evaluator, labels):
    with pytest.raises(ValueError):
        evaluator.head_spec(dict(lang_num=len(labels)), labels)


def test_threshold_sweep_keeps_empty_and_overprediction_errors(tmp_path, evaluator):
    metrics = sys.modules["lid_metrics"]
    head = evaluator.head_spec(
        dict(
            loss="arc_margin_subcenter_intertopk_multilabel_bce",
            loss_conf=dict(K=3, k_top=0),
            lang_num=3,
        ),
        ["ara", "eng", "jpn"],
    )
    logits = np.asarray([[1.0, 2.0, -10.0]])
    result = metrics.evaluate_set(
        tmp_path,
        "threshold",
        ["ara", "eng", "jpn"],
        ["u"],
        logits,
        metrics.probabilities(logits, "sigmoid"),
        {"u": ("ara", "eng")},
        set(),
        head,
        [0, 0.5, 1],
    )
    rows = [row for row in result["threshold_sweep"] if row["group_type"] == "overall"]
    assert [row["accuracy"] for row in rows] == [0, 1, 0]


def test_summary_rejects_stale_reference_labels(tmp_path):
    module = load_module(
        "test_summary_alignment", RECIPE / "evaluation" / "summarize_details.py"
    )
    path = tmp_path / "details.tsv"
    path.write_text("utt\tref\texact\nu\tara eng\t1\n")
    with pytest.raises(ValueError, match="label mismatch"):
        module.read_correctness(path, None, {"u": ("eng", "jpn")})


def test_paired_set_alignment_retains_ordered_contract(tmp_path):
    module = load_module(
        "test_paired_alignment", RECIPE / "evaluation" / "paired_significance.py"
    )
    path = tmp_path / "details.tsv"
    path.write_text("utt\tref\tset_exact\tseq_exact\nu\teng ara\t1\t0\n")
    assert module.read_details(path, "set_exact")[1]["u"] == "ara eng"
    assert module.read_details(path, "seq_exact")[1]["u"] == "eng ara"
