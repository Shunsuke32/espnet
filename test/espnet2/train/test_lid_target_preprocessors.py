import numpy as np
import pytest

from espnet2.train.preprocessor import (
    LIDMultiLabelPreprocessor,
    LIDSoftLabelPreprocessor,
)


@pytest.fixture()
def lang2utt(tmp_path):
    path = tmp_path / "lang2utt"
    path.write_text("ara u1\neng u2\njpn u3\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("eng", [0.0, 1.0, 0.0]),
        ("<ara> <eng>", [0.5, 0.5, 0.0]),
        (np.asarray("ara eng"), [0.5, 0.5, 0.0]),
    ],
)
def test_lid_soft_label_preprocessor(lang2utt, raw, expected):
    preprocessor = LIDSoftLabelPreprocessor(
        train=True,
        lang2utt=str(lang2utt),
        fix_duration=False,
        noise_apply_prob=0.0,
        rir_apply_prob=0.0,
    )
    data = preprocessor._text_process({"lid_labels": raw})
    np.testing.assert_allclose(data["lid_labels"], expected)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("eng", [0.0, 1.0, 0.0]),
        ("<ara> <eng>", [1.0, 1.0, 0.0]),
        (np.asarray("ara eng"), [1.0, 1.0, 0.0]),
    ],
)
def test_lid_multi_label_preprocessor(lang2utt, raw, expected):
    preprocessor = LIDMultiLabelPreprocessor(
        train=True,
        lang2utt=str(lang2utt),
        fix_duration=False,
        noise_apply_prob=0.0,
        rir_apply_prob=0.0,
    )
    data = preprocessor._text_process({"lid_labels": raw})
    np.testing.assert_allclose(data["lid_labels"], expected)


@pytest.mark.parametrize(
    "preprocessor_class", [LIDSoftLabelPreprocessor, LIDMultiLabelPreprocessor]
)
def test_lid_target_preprocessor_rejects_duplicate_labels(lang2utt, preprocessor_class):
    preprocessor = preprocessor_class(
        train=True,
        lang2utt=str(lang2utt),
        fix_duration=False,
        noise_apply_prob=0.0,
        rir_apply_prob=0.0,
    )
    with pytest.raises(ValueError, match="duplicate"):
        preprocessor._text_process({"lid_labels": "eng eng"})
