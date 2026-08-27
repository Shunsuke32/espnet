# Code from WeSpeaker: https://github.com/wenet-e2e/wespeaker/blob/
# c9ec537b53fe1e04525be74b2550ee95bed3a891/wespeaker/models/projections.py#L243

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.spk.loss.abs_loss import AbsLoss


class ArcMarginProduct_intertopk_subcenter(AbsLoss):
    """ArcFace loss (AAMSoftmax loss) with Inter-TopK penalty and Sub-center.

    This loss function combines three techniques:
    1. ArcFace: Additive angular margin loss for better feature discrimination
    2. Sub-center: Multiple prototypes per class to handle intra-class variation
    3. Inter-TopK: Additional penalty on hardest negative samples

    Reference:
        Multi-Query Multi-Head Attention Pooling and Inter-TopK Penalty
        for Speaker Verification
        https://arxiv.org/pdf/2110.05042.pdf
        Sub-Center ArcFace: Boosting Face Recognition by
        Large-Scale Noisy Web Faces.
        https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf

    Args:
        nout: Dimension of input features (embedding size)
        nclasses: Number of output classes
        scale: Feature scaling factor
        margin: Angular margin for positive samples
        easy_margin: Whether to use easy margin variant
        K: Number of sub-centers per class
        mp: Margin penalty for hard negative samples
        k_top: Number of hardest negative samples to penalize
        do_lm: Whether to enable Large Margin Fine-tuning mode
    """

    def __init__(
        self,
        nout: int,
        nclasses: int,
        scale: float = 32.0,
        margin: float = 0.2,
        easy_margin: bool = False,
        K: int = 3,
        mp: float = 0.06,
        k_top: int = 5,
        do_lm: bool = False,
    ):
        super().__init__(nout)
        self.in_features = nout
        self.out_features = nclasses
        self.scale = scale
        self.margin = margin
        self.do_lm = do_lm

        # Sub-center and Inter-TopK configuration
        self.K = K  # Number of sub-centers per class
        if do_lm:  # Large Margin Fine-tuning mode: disable hard sample penalty
            self.mp = 0.0
            self.k_top = 0
        else:
            self.mp = mp  # Margin penalty for hard negatives
            self.k_top = k_top  # Number of hardest negatives to penalize

        # Weight with K sub-centers for each of the nclasses classes
        self.weight = nn.Parameter(torch.FloatTensor(self.K * nclasses, nout))
        nn.init.xavier_uniform_(self.weight)

        # Pre-computed trigonometric values for angular margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # Threshold for easy_margin
        self.mm = math.sin(math.pi - margin) * margin  # For numerical stability
        self.mmm = 1.0 + math.cos(math.pi - margin)  # Enhanced continuity term

        # Store margin for updates
        self.m = self.margin

        # Pre-computed values for hard sample penalty (initially 0)
        self.cos_mp = math.cos(0.0)
        self.sin_mp = math.sin(0.0)

        self.ce = nn.CrossEntropyLoss()

    def update(self, margin: float = 0.2):
        r"""Update margin and related trigonometric values during training."""

        self.margin = margin

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

        # Adaptive hard sample margin: scales with main margin
        if margin > 0.001:
            mp = self.mp * (margin / 0.2)
        else:
            mp = 0.0  # Disable penalty for very small margins
        self.cos_mp = math.cos(mp)
        self.sin_mp = math.sin(mp)

    def forward(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        r"""Forward pass of ArcFace (AAMSoftmax) with sub-center and inter-topk penalty.

        Args:
            input: Input embeddings, shape (batch_size, embedding_dim)
            label: Ground truth labels, shape (batch_size,)

        Returns:
            loss: Cross-entropy loss with angular margins
            accuracy: Classification accuracy
            pred_lids: Predicted class indices
        """

        cosine = F.linear(
            F.normalize(input), F.normalize(self.weight)
        )  # Output: (batch_size, K*nclasses)

        # Reshape to separate sub-centers: (batch_size, nclasses, K)
        cosine = torch.reshape(cosine, (-1, self.out_features, self.K))

        # Sub-center max pooling: select the best sub-center for each class
        # This handles intra-class variation by choosing the closest sub-center
        cosine, _ = torch.max(cosine, 2)  # Output: (batch_size, nclasses)

        preds = torch.argmax(cosine, dim=1)  # Output: (batch_size,)

        if label is not None:
            if len(label.size()) == 2:
                label = label.squeeze(1)
            accuracy = (preds == label).float().mean()
        else:
            # Inference mode: return predictions only
            loss = None
            accuracy = None
            return loss, accuracy, preds

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Apply angular margin to positive samples:
        # cos(θ + m) = cosθ·cos(m) - sinθ·sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Apply margin penalty to hard negative samples:
        # cos(θ - mp) = cosθ·cos(mp) + sinθ·sin(mp)
        # This increases the penalty for hardest negative samples
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp

        # Handle numerical stability for angular margin
        if self.easy_margin:
            # Easy margin: only apply margin when cosine > 0
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard margin with numerical stability
            # Use enhanced continuity term (mmm) for better gradient flow
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        if self.k_top > 0:
            # Find top-k hardest negative samples (excluding ground truth class)
            _, top_k_index = torch.topk(cosine - 2 * one_hot, self.k_top)
            top_k_one_hot = input.new_zeros(cosine.size()).scatter_(1, top_k_index, 1)

            output = (
                (one_hot * phi)  # Ground truth: apply positive margin
                + (top_k_one_hot * phi_mp)  # Hard negatives: apply penalty margin
                + (
                    (1.0 - one_hot - top_k_one_hot) * cosine
                )  # Other negatives: no margin
            )
        else:
            # Without Inter-TopK: only apply margin to ground truth
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale logits for stable training
        output *= self.scale

        loss = self.ce(output, label)
        return loss, accuracy, preds


class ArcMarginProduct_intertopk_subcenter_softtarget(
    ArcMarginProduct_intertopk_subcenter
):
    """ArcFace loss with support for soft class targets.

    This variant keeps the same projection/margin logic as
    ``ArcMarginProduct_intertopk_subcenter`` but accepts either hard integer
    labels or a dense target distribution of shape ``(batch, nclasses)``.
    Multi-label LID can therefore use targets such as 0.5/0.5 for two
    languages while preserving the existing MMS+ECAPA classifier shape.  Dense
    targets are optimized with KL(target || prediction).
    """

    def forward(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
        cosine, _ = torch.max(cosine, 2)

        preds = torch.argmax(cosine, dim=1)

        if label is None:
            return None, None, preds

        if label.dim() == 2 and label.size(1) == self.out_features:
            target = label.to(dtype=cosine.dtype)
            target_sum = target.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
            target = target / target_sum
            positive_mask = target > 0
        else:
            if len(label.size()) == 2:
                label = label.squeeze(1)
            label = label.long()
            target = torch.zeros_like(cosine)
            target.scatter_(1, label.view(-1, 1), 1.0)
            positive_mask = target > 0

        accuracy = positive_mask.gather(1, preds.view(-1, 1)).float().mean()

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)

        positive_one_hot = positive_mask.to(dtype=cosine.dtype)

        if self.k_top > 0:
            _, top_k_index = torch.topk(cosine - 2 * positive_one_hot, self.k_top)
            top_k_one_hot = input.new_zeros(cosine.size()).scatter_(1, top_k_index, 1)
            output = (
                positive_one_hot * phi
                + top_k_one_hot * phi_mp
                + (1.0 - positive_one_hot - top_k_one_hot) * cosine
            )
        else:
            output = positive_one_hot * phi + (1.0 - positive_one_hot) * cosine

        output *= self.scale
        loss = F.kl_div(F.log_softmax(output, dim=1), target, reduction="batchmean")
        return loss, accuracy, preds


class ArcMarginProduct_intertopk_subcenter_multilabel_bce(
    ArcMarginProduct_intertopk_subcenter
):
    """Sub-center cosine classifier optimized with multi-label BCE.

    The training logits apply the ArcFace margin to every positive label and
    the Inter-TopK penalty to the hardest negatives. Inference logits do not
    depend on labels and are scaled cosine similarities.
    """

    def __init__(
        self,
        nout: int,
        nclasses: int,
        scale: float = 32.0,
        margin: float = 0.2,
        easy_margin: bool = False,
        K: int = 3,
        mp: float = 0.06,
        k_top: int = 5,
        do_lm: bool = False,
        threshold: float = 0.5,
        pos_weight: Optional[Union[float, Sequence[float]]] = None,
        apply_margin: bool = True,
    ):
        super().__init__(
            nout=nout,
            nclasses=nclasses,
            scale=scale,
            margin=margin,
            easy_margin=easy_margin,
            K=K,
            mp=mp,
            k_top=k_top,
            do_lm=do_lm,
        )
        self.nclasses = int(nclasses)
        self.threshold = float(threshold)
        self.apply_margin = bool(apply_margin)

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

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    def _cosine(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() != 2:
            raise ValueError(f"expected input shape (batch, dim), got {input.shape}")
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
        cosine, _ = torch.max(cosine, 2)
        return cosine

    def _target_mask(
        self, label: torch.Tensor, ref: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if label.dim() == 2 and label.size(1) == self.out_features:
            target = label.to(dtype=ref.dtype)
            positive_mask = target > 0
        else:
            if len(label.size()) == 2:
                label = label.squeeze(1)
            label = label.long()
            target = torch.zeros_like(ref)
            target.scatter_(1, label.view(-1, 1), 1.0)
            positive_mask = target > 0
        if target.shape != ref.shape:
            raise ValueError(
                f"target shape {target.shape} does not match logits {ref.shape}"
            )
        return target, positive_mask

    def compute_logits(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        cosine = self._cosine(input)
        if label is None or not self.apply_margin:
            return cosine * self.scale

        _, positive_mask = self._target_mask(label, cosine)
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)

        positive_one_hot = positive_mask.to(dtype=cosine.dtype)
        if self.k_top > 0:
            _, top_k_index = torch.topk(cosine - 2 * positive_one_hot, self.k_top)
            top_k_one_hot = input.new_zeros(cosine.size()).scatter_(1, top_k_index, 1)
            output = (
                positive_one_hot * phi
                + top_k_one_hot * phi_mp
                + (1.0 - positive_one_hot - top_k_one_hot) * cosine
            )
        else:
            output = positive_one_hot * phi + (1.0 - positive_one_hot) * cosine
        return output * self.scale

    def forward(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        if label is None:
            return None, None, self.compute_logits(input)

        logits = self.compute_logits(input, label)
        target, _ = self._target_mask(label, logits)
        loss = self.bce(logits, target)
        pred = torch.sigmoid(logits) >= self.threshold
        gold = target >= 0.5
        accuracy = pred.eq(gold).all(dim=1).float().mean()
        return loss, accuracy, logits
