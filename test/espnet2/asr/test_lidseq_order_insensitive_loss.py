import torch

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.legacy.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet2.legacy.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,
)


class SwappedDecoder(torch.nn.Module):
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
