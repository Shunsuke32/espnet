import logging

import pytest
import torch
import yaml

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.legacy.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet2.legacy.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,
)


class SwappedDecoder(AbsDecoder):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens):
        return self.output, None


def test_order_insensitive_lidseq_loss_takes_per_utterance_minimum():
    model = ESPnetASRModel.__new__(ESPnetASRModel)
    torch.nn.Module.__init__(model)
    model.ignore_id = -1
    model.sos = 4
    model.eos = 4
    model.criterion_att = LabelSmoothingLoss(
        size=5,
        padding_idx=model.ignore_id,
        smoothing=0.1,
        normalize_length=False,
    )

    ys_pad = torch.tensor([[1, 2], [3, -1]])
    ys_pad_lens = torch.tensor([2, 1])
    _, ys_out_pad = add_sos_eos(ys_pad, model.sos, model.eos, model.ignore_id)
    original_output = torch.randn(2, 3, 5)

    swapped_ys_pad = ys_pad.clone()
    swapped_ys_pad[0, 0], swapped_ys_pad[0, 1] = ys_pad[0, 1], ys_pad[0, 0]
    _, swapped_ys_out_pad = add_sos_eos(
        swapped_ys_pad, model.sos, model.eos, model.ignore_id
    )
    swapped_output = torch.randn(2, 3, 5)
    model.decoder = SwappedDecoder(swapped_output)

    original = model._calc_label_smoothing_loss_per_sample(original_output, ys_out_pad)
    swapped = model._calc_label_smoothing_loss_per_sample(
        swapped_output, swapped_ys_out_pad
    )
    expected = torch.stack((torch.minimum(original[0], swapped[0]), original[1])).mean()

    actual = model._calc_order_insensitive_lidseq_att_loss(
        torch.randn(2, 4, 3),
        torch.tensor([4, 4]),
        ys_pad,
        ys_pad_lens,
        original_output,
        ys_out_pad,
    )
    torch.testing.assert_close(actual, expected)


class TinyDecoder(AbsDecoder):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(5, 4)
        self.output = torch.nn.Linear(4, 5)
        self.calls = 0

    def forward(self, encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens):
        self.calls += 1
        hidden = self.embed(ys_in_pad) + encoder_out.mean(dim=1, keepdim=True)
        return self.output(hidden), None


def build_model(**options):
    args = dict(
        vocab_size=5,
        token_list=["<blank>", "<eng>", "<ara>", "<jpn>", "<sos/eos>"],
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=None,
        postencoder=None,
        decoder=TinyDecoder(),
        ctc=CTC(odim=5, encoder_output_size=4),
        joint_network=None,
        ctc_weight=0.0,
        lsm_weight=0.1,
        report_cer=False,
        report_wer=False,
    )
    args.update(options)
    return ESPnetASRModel(**args)


@pytest.mark.parametrize(
    "options,enabled",
    [
        ({}, False),
        ({"pit_loss": None}, False),
        ({"pit_loss": False}, False),
        ({"pit_loss": True}, True),
        ({"lidseq_order_insensitive_loss": False}, False),
        ({"lidseq_order_insensitive_loss": True}, True),
        ({"pit_loss": False, "lidseq_order_insensitive_loss": False}, False),
        ({"pit_loss": True, "lidseq_order_insensitive_loss": True}, True),
        ({"pit_loss": None, "lidseq_order_insensitive_loss": True}, True),
        ({"pit_loss": True, "lidseq_order_insensitive_loss": None}, True),
    ],
)
def test_pit_option_aliases_and_default(options, enabled, caplog):
    with caplog.at_level(logging.WARNING):
        model = build_model(**options)
    assert model.pit_loss is enabled
    assert model.lidseq_order_insensitive_loss is enabled
    assert model.pit_loss_reduction == "min"
    assert model.lidseq_order_insensitive_reduction == "min"
    deprecated = options.get("lidseq_order_insensitive_loss") is not None
    assert ("is deprecated; use pit_loss instead" in caplog.text) == deprecated


@pytest.mark.parametrize(
    "options",
    [
        {"pit_loss_reduction": "min"},
        {"lidseq_order_insensitive_reduction": "min"},
        {"pit_loss_reduction": "min", "lidseq_order_insensitive_reduction": "min"},
        {"pit_loss_reduction": None, "lidseq_order_insensitive_reduction": "min"},
    ],
)
def test_pit_reduction_aliases(options, caplog):
    with caplog.at_level(logging.WARNING):
        model = build_model(pit_loss=True, **options)
    assert model.pit_loss_reduction == "min"
    deprecated = options.get("lidseq_order_insensitive_reduction") is not None
    assert (
        "use pit_loss_reduction instead" in caplog.text
    ) == deprecated


@pytest.mark.parametrize("canonical,legacy", [(True, False), (False, True)])
def test_pit_boolean_conflicts_are_rejected(canonical, legacy):
    with pytest.raises(ValueError, match="Conflicting pit_loss="):
        build_model(
            pit_loss=canonical,
            lidseq_order_insensitive_loss=legacy,
        )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("canonical,legacy", [("min", "mean"), ("mean", "min")])
def test_pit_reduction_conflicts_are_rejected(enabled, canonical, legacy):
    with pytest.raises(ValueError, match="Conflicting pit_loss_reduction="):
        build_model(
            pit_loss=enabled,
            pit_loss_reduction=canonical,
            lidseq_order_insensitive_reduction=legacy,
        )


@pytest.mark.parametrize("legacy", [False, True])
def test_pit_disabled_keeps_legacy_unused_reduction_behavior(legacy):
    options = (
        dict(
            lidseq_order_insensitive_loss=False,
            lidseq_order_insensitive_reduction="mean",
        )
        if legacy
        else dict(pit_loss=False, pit_loss_reduction="mean")
    )
    model = build_model(ctc_weight=0.5, **options)
    assert model.pit_loss is False
    assert model.pit_loss_reduction == "mean"


@pytest.mark.parametrize("legacy", [False, True])
def test_pit_enabled_still_rejects_unsupported_reduction(legacy):
    options = (
        dict(
            lidseq_order_insensitive_loss=True,
            lidseq_order_insensitive_reduction="mean",
        )
        if legacy
        else dict(pit_loss=True, pit_loss_reduction="mean")
    )
    with pytest.raises(ValueError, match="Unsupported pit_loss_reduction"):
        build_model(**options)


@pytest.mark.parametrize("legacy", [False, True])
def test_pit_enabled_still_rejects_ctc_mixture(legacy):
    option = "lidseq_order_insensitive_loss" if legacy else "pit_loss"
    with pytest.raises(ValueError, match="attention-only"):
        build_model(ctc_weight=0.5, **{option: True})


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("normalize_length", [False, True])
@pytest.mark.parametrize("two_tokens", [False, True])
def test_pit_aliases_preserve_loss_gradients_and_checkpoint_keys(
    enabled, normalize_length, two_tokens
):
    legacy_config = yaml.safe_load(
        "model_conf:\n"
        f"  lidseq_order_insensitive_loss: {str(enabled).lower()}\n"
        "  lidseq_order_insensitive_reduction: min\n"
    )["model_conf"]
    old = build_model(length_normalized_loss=normalize_length, **legacy_config)
    new = build_model(
        pit_loss=enabled,
        pit_loss_reduction="min",
        length_normalized_loss=normalize_length,
    )
    new.load_state_dict(old.state_dict(), strict=True)
    assert new.state_dict().keys() == old.state_dict().keys()
    inputs = torch.randn(2, 3, 4)
    targets = torch.tensor([[1, 2], [3, -1]] if two_tokens else [[1], [3]])
    lengths = torch.tensor([2, 1] if two_tokens else [1, 1])
    results = []
    for model in (old, new):
        encoder_out = inputs.clone().requires_grad_()
        loss, accuracy, _, _ = model._calc_att_loss(
            encoder_out, torch.tensor([3, 3]), targets, lengths
        )
        loss.backward()
        results.append(
            (
                loss.detach(),
                accuracy,
                encoder_out.grad,
                [p.grad for p in model.decoder.parameters()],
            )
        )
        assert model.decoder.calls == (2 if enabled and two_tokens else 1)
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


@pytest.mark.parametrize("options", [{}, {"pit_loss": False}])
def test_pit_default_and_off_use_original_ordered_criterion(options):
    model = build_model(**options)
    encoder_out = torch.randn(2, 3, 4)
    targets = torch.tensor([[1, 2], [3, -1]])
    lengths = torch.tensor([2, 1])
    loss, _, _, _ = model._calc_att_loss(
        encoder_out, torch.tensor([3, 3]), targets, lengths
    )
    assert model.decoder.calls == 1
    inputs, outputs = add_sos_eos(targets, model.sos, model.eos, model.ignore_id)
    logits, _ = model.decoder(encoder_out, torch.tensor([3, 3]), inputs, lengths + 1)
    torch.testing.assert_close(loss, model.criterion_att(logits, outputs), rtol=0, atol=0)


def test_pit_does_not_change_accuracy_to_permutation_invariant():
    logits = torch.full((1, 3, 5), -10.0)
    logits[0, torch.arange(3), torch.tensor([2, 1, 4])] = 10.0
    model = build_model(pit_loss=True, lsm_weight=0.0, decoder=SwappedDecoder(logits))
    loss, accuracy, _, _ = model._calc_att_loss(
        torch.zeros(1, 2, 4), torch.tensor([2]), torch.tensor([[1, 2]]), torch.tensor([2])
    )
    assert loss.item() == 0.0
    assert accuracy == pytest.approx(1 / 3)


def test_pit_still_rejects_more_than_two_tokens():
    model = build_model(pit_loss=True)
    with pytest.raises(RuntimeError, match="only supports 1- or 2-token"):
        model._calc_att_loss(
            torch.randn(1, 3, 4),
            torch.tensor([3]),
            torch.tensor([[1, 2, 3]]),
            torch.tensor([3]),
        )


def test_pit_still_rejects_language_prefix():
    model = build_model(pit_loss=True, lang_token_id=1)
    with pytest.raises(RuntimeError, match="does not support lang_token_id"):
        model._calc_att_loss(
            torch.randn(1, 3, 4), torch.tensor([3]), torch.tensor([[1]]), torch.tensor([1])
        )
