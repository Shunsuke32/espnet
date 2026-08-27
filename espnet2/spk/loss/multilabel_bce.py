from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from espnet2.spk.loss.abs_loss import AbsLoss


class MultiLabelBCE(AbsLoss):
    """Multi-label classification head with BCEWithLogitsLoss."""

    def __init__(
        self,
        nout: int,
        nclasses: int,
        threshold: float = 0.5,
        pos_weight: Optional[Union[float, Sequence[float]]] = None,
    ):
        super().__init__(nout)
        self.nclasses = int(nclasses)
        self.threshold = float(threshold)
        self.classifier = nn.Linear(nout, nclasses)

        if pos_weight is None:
            pos_weight_tensor = None
        elif isinstance(pos_weight, (float, int)):
            pos_weight_tensor = torch.full((nclasses,), float(pos_weight))
        else:
            if len(pos_weight) != nclasses:
                raise ValueError(
                    f"pos_weight must have {nclasses} values, got {len(pos_weight)}"
                )
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    def compute_logits(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() != 2:
            raise ValueError(f"expected input shape (batch, dim), got {input.shape}")
        return self.classifier(input)

    def forward(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        logits = self.compute_logits(input)
        if label is None:
            return None, None, logits

        target = label.to(dtype=logits.dtype)
        if target.shape != logits.shape:
            raise ValueError(
                f"target shape {target.shape} does not match logits {logits.shape}"
            )

        loss = self.criterion(logits, target)
        pred = torch.sigmoid(logits) >= self.threshold
        gold = target >= 0.5
        accuracy = pred.eq(gold).all(dim=1).float().mean()
        return loss, accuracy, logits
