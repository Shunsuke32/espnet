import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("stats_dir", [None, "exp/train_config_stats"])
@pytest.mark.parametrize("speed", ["", "0.9 1.0 1.1"])
def test_lid_template_stats_directory(tmp_path, stats_dir, speed):
    root = Path(__file__).resolve().parents[3]
    template = root / "egs2/TEMPLATE/lid1"
    for name in ("utils", "scripts"):
        (tmp_path / name).symlink_to(template / name, target_is_directory=True)
    for name in ("path.sh", "cmd.sh"):
        (tmp_path / name).write_text("")
    options = [
        "--stage",
        "100",
        "--stop_stage",
        "0",
        "--ngpu",
        "0",
        "--expdir",
        "custom_exp",
        "--fs",
        "8k",
        "--train_set",
        "train",
        "--valid_set",
        "valid",
        "--test_sets",
        "test",
        "--speed_perturb_factors",
        speed,
    ]
    if stats_dir is not None:
        options += ["--lid_stats_dir", stats_dir]
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" "${@:2}"\nprintf "STATS=%s\\n" "$lid_stats_dir"',
            "test",
            str(template / "lid.sh"),
            *options,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    expected = stats_dir or "custom_exp/lid_stats_8k"
    if speed:
        expected += "_sp"
    assert f"STATS={expected}\n" in result.stdout
