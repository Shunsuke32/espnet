import pytest
import torch
import torch.nn.functional as F

from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
    ArcMarginProduct_intertopk_subcenter_multilabel_bce,
    ArcMarginProduct_intertopk_subcenter_softtarget,
)


def test_arc_multilabel_bce_matches_pytorch_mean_reduction():
    loss_module = ArcMarginProduct_intertopk_subcenter_multilabel_bce(
        nout=3, nclasses=4, K=3, k_top=0, pos_weight=5.0
    )
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
    torch.testing.assert_close(loss_module.compute_logits(inputs, targets), logits)


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


@pytest.mark.parametrize(
    "loss_class",
    [
        ArcMarginProduct_intertopk_subcenter_softtarget,
        ArcMarginProduct_intertopk_subcenter_multilabel_bce,
    ],
)
@pytest.mark.parametrize("dtype", [None, torch.bfloat16, torch.float16])
def test_arc_targets_have_finite_gradients_at_cosine_boundaries(loss_class, dtype):
    module = loss_class(nout=4, nclasses=3, K=1, k_top=0)
    with torch.no_grad():
        module.weight.copy_(
            torch.tensor([[1.0, 0, 0, 0], [-1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        )
    inputs = torch.tensor([[1.0, 0, 0, 0]], requires_grad=True)
    target = torch.tensor([[1.0, 0, 0]])
    with torch.autocast("cpu", enabled=dtype is not None, dtype=dtype):
        loss, _, _ = module(inputs, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(module.weight.grad).all()


@pytest.mark.parametrize("nclasses", [2, 3, 6])
def test_arc_multilabel_topk_penalizes_only_available_negatives(nclasses):
    module = ArcMarginProduct_intertopk_subcenter_multilabel_bce(
        nout=4, nclasses=nclasses, K=1, k_top=5, scale=1.0
    )
    module.update(0.2)
    inputs = torch.randn(2, 4)
    targets = torch.zeros(2, nclasses)
    targets[:, :2] = 1.0
    cosine = module._cosine(inputs)
    sine = (1.0 - cosine.square()).clamp_min(torch.finfo(cosine.dtype).eps).sqrt()
    phi = cosine * module.cos_m - sine * module.sin_m
    phi = torch.where(cosine > module.th, phi, cosine - module.mmm)
    phi_mp = cosine * module.cos_mp + sine * module.sin_mp
    expected = torch.where(targets.bool(), phi, phi_mp)
    torch.testing.assert_close(module.compute_logits(inputs, targets), expected)


def test_softtarget_hard_labels_match_original_loss_and_gradients():
    torch.manual_seed(7)
    hard = ArcMarginProduct_intertopk_subcenter(nout=4, nclasses=8)
    soft = ArcMarginProduct_intertopk_subcenter_softtarget(nout=4, nclasses=8)
    soft.load_state_dict(hard.state_dict(), strict=True)
    hard.update(0.3)
    soft.update(0.3)
    inputs = torch.randn(3, 4, requires_grad=True)
    soft_inputs = inputs.detach().clone().requires_grad_()
    labels = torch.tensor([[1], [4], [6]])
    loss, acc, pred = hard(inputs, labels)
    soft_loss, soft_acc, soft_pred = soft(soft_inputs, labels)
    loss.backward()
    soft_loss.backward()
    torch.testing.assert_close(soft_loss, loss)
    torch.testing.assert_close(soft_acc, acc)
    torch.testing.assert_close(soft_pred, pred)
    torch.testing.assert_close(soft_inputs.grad, inputs.grad)
    torch.testing.assert_close(soft.weight.grad, hard.weight.grad)


@pytest.mark.parametrize("nclasses", [2, 3, 6])
def test_softtarget_topk_uses_available_negatives(nclasses):
    soft = ArcMarginProduct_intertopk_subcenter_softtarget(
        nout=4, nclasses=nclasses, K=1, k_top=5, scale=1.0
    )
    reference = ArcMarginProduct_intertopk_subcenter_multilabel_bce(
        nout=4, nclasses=nclasses, K=1, k_top=5, scale=1.0
    )
    reference.load_state_dict(soft.state_dict(), strict=True)
    soft.update(0.2)
    reference.update(0.2)
    inputs = torch.randn(2, 4)
    targets = torch.zeros(2, nclasses)
    targets[:, :2] = 0.5
    loss, _, _ = soft(inputs, targets)
    expected = F.kl_div(
        F.log_softmax(reference.compute_logits(inputs, targets), dim=1),
        targets,
        reduction="batchmean",
    )
    torch.testing.assert_close(loss, expected)
