import torch
import torch.nn.functional as F

from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter_multilabel_bce,
    ArcMarginProduct_intertopk_subcenter_softtarget,
)
from espnet2.spk.loss.multilabel_bce import MultiLabelBCE


def test_multilabel_bce_matches_pytorch_mean_reduction():
    loss_module = MultiLabelBCE(nout=3, nclasses=4, pos_weight=5.0)
    inputs = torch.randn(2, 3)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    loss, accuracy, logits = loss_module(inputs, targets)
    expected = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=torch.full((4,), 5.0),
        reduction="mean",
    )

    torch.testing.assert_close(loss, expected)
    assert accuracy.ndim == 0
    assert logits.shape == targets.shape
    assert torch.equal(loss_module.compute_logits(inputs), logits)


def test_softtarget_loss_uses_batchmean_kl():
    loss_module = ArcMarginProduct_intertopk_subcenter_softtarget(
        nout=3,
        nclasses=4,
        scale=1.0,
        margin=0.0,
        K=1,
        k_top=0,
    )
    inputs = torch.randn(2, 3)
    targets = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    loss, accuracy, _ = loss_module(inputs, targets)
    cosine = F.linear(F.normalize(inputs), F.normalize(loss_module.weight))
    expected = F.kl_div(F.log_softmax(cosine, dim=1), targets, reduction="batchmean")

    torch.testing.assert_close(loss, expected)
    assert accuracy.ndim == 0


def test_arc_margin_multilabel_inference_logits_are_label_independent():
    loss_module = ArcMarginProduct_intertopk_subcenter_multilabel_bce(
        nout=4,
        nclasses=8,
        scale=30.0,
        margin=0.5,
        K=3,
        mp=0.06,
        k_top=5,
        pos_weight=50.0,
    )
    inputs = torch.randn(2, 4)
    targets = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    loss, accuracy, training_logits = loss_module(inputs, targets)
    inference_logits = loss_module.compute_logits(inputs)
    no_label_loss, no_label_accuracy, forward_inference_logits = loss_module(inputs)

    assert torch.isfinite(loss)
    assert accuracy.ndim == 0
    assert training_logits.shape == targets.shape
    assert not torch.equal(training_logits, inference_logits)
    assert no_label_loss is None
    assert no_label_accuracy is None
    torch.testing.assert_close(forward_inference_logits, inference_logits)
