from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

import espnet3.systems.lid.collect_stats as stats_module
import espnet3.systems.lid.system as system_module
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.lid.system import LIDSystem


class DummyDataset:
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        return {
            "speech": np.zeros(self.lengths[index], dtype=np.float32),
            "lid_labels": "eng",
        }


def test_collect_speech_shapes(tmp_path, monkeypatch):
    organizer = SimpleNamespace(
        train=DummyDataset([3, 5]),
        valid=DummyDataset([4]),
    )
    seen = {}

    def fake_instantiate(config):
        seen["preprocessor"] = config.preprocessor
        return organizer

    monkeypatch.setattr(stats_module, "instantiate", fake_instantiate)
    config = OmegaConf.create(
        {
            "stats_dir": str(tmp_path),
            "dataset": {"preprocessor": {"_target_": "unused"}},
            "dataloader": {
                "train": {"iter_factory": {"num_workers": 0}},
                "valid": {"iter_factory": {"num_workers": 0}},
            },
        }
    )

    stats_module.collect_speech_shapes(config)

    assert seen["preprocessor"] is None
    assert config.dataset.preprocessor._target_ == "unused"
    assert (tmp_path / "train/speech_shape").read_text() == "0 3\n1 5\n"
    assert (tmp_path / "valid/speech_shape").read_text() == "0 4\n"
    for mode in ("train", "valid"):
        assert (tmp_path / mode / "stats_keys").read_text() == "\n"
        assert (tmp_path / mode / "batch_keys").read_text() == "speech\n"
        assert not (tmp_path / mode / "lid_labels_shape").exists()
        assert not list((tmp_path / mode).glob("*_stats.npz"))


def test_lid_system_uses_lid_stats_when_feature_extraction_is_disabled(
    tmp_path, monkeypatch
):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "stats_dir": str(tmp_path / "stats"),
            "model": {
                "model_conf": {"extract_feats_in_collect_stats": False},
            },
        }
    )
    calls = []
    monkeypatch.setattr(
        system_module,
        "collect_speech_shapes",
        lambda cfg: calls.append(cfg),
    )

    system = LIDSystem(training_config=config)
    system.collect_stats()

    assert calls == [config]


def test_lid_system_preserves_base_stats_path(tmp_path, monkeypatch):
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "stats_dir": str(tmp_path / "stats"),
            "model": {"model_conf": {}},
        }
    )
    monkeypatch.setattr(BaseSystem, "collect_stats", lambda self: "base")

    system = LIDSystem(training_config=config)

    assert system.collect_stats() == "base"
