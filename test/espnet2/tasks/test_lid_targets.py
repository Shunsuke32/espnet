import numpy as np
import pytest
import torch

from espnet2.tasks.lid import LIDTask


@pytest.mark.parametrize(
    "preprocessor,loss_name",
    [
        ("lid", "aamsoftmax"),
        ("lid", "aamsoftmax_sc_topk"),
        ("lid", "softmax"),
        ("lid_softlabel", "aamsoftmax_sc_topk_softtarget"),
        ("lid_multilabel", "arc_margin_subcenter_intertopk_multilabel_bce"),
    ],
)
def test_lid_target_registration_dimensions_and_collation(
    tmp_path, preprocessor, loss_name
):
    lang2utt = tmp_path / "lang2utt"
    lang2utt.write_text("ara u1\neng u2\njpn u3\n")
    args = LIDTask.get_parser().parse_args([])
    args.lang2utt = str(lang2utt)
    args.lang_num = 3
    args.preprocessor = preprocessor
    args.preprocessor_conf = dict(
        fix_duration=False, noise_apply_prob=0.0, rir_apply_prob=0.0
    )
    args.frontend = None
    args.input_size = 7
    args.encoder = "identity"
    args.pooling = "mean"
    args.projector = "rawnet3"
    args.projector_conf = dict(output_size=4)
    args.loss = loss_name
    args.loss_conf = dict(k_top=0) if "topk" in loss_name else {}
    preprocess = LIDTask.build_preprocess_fn(args, train=False)
    data = [
        ("u1", dict(speech=np.ones((5, 7), dtype=np.float32), lid_labels="ara")),
        (
            "u2",
            dict(
                speech=np.ones((5, 7), dtype=np.float32),
                lid_labels="eng" if preprocessor == "lid" else "eng jpn",
            ),
        ),
    ]
    _, batch = LIDTask.build_collate_fn(args, train=False)(
        [(uid, preprocess(uid, item)) for uid, item in data]
    )
    labels = batch["lid_labels"]
    assert labels.shape == (2, 1 if preprocessor == "lid" else 3)
    assert "lid_labels_lengths" not in batch
    if preprocessor == "lid_softlabel":
        torch.testing.assert_close(labels.sum(1), torch.ones(2))
    elif preprocessor == "lid_multilabel":
        torch.testing.assert_close(labels.sum(1), torch.tensor([1.0, 2.0]))
    else:
        assert labels.dtype == torch.int64
    model = LIDTask.build_model(args).eval()
    assert model.encoder.output_size() == 7
    assert model.pooling.output_size() == 7
    assert model.projector.output_size() == 4
    loss, stats, weight = model(**batch)
    loss.backward()
    assert torch.isfinite(loss)
    torch.testing.assert_close(stats["loss"], loss.detach())
    assert weight.item() == 2
    assert all(
        torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.grad is not None
    )


def test_excluded_plain_bce_is_not_registered():
    with pytest.raises(SystemExit):
        LIDTask.get_parser().parse_args(["--loss", "multilabel_bce"])
