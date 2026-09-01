"""
Class-weighted loss for LesionBridge Stage 1 segmentation.

Fixes the MA (microaneurysm) collapse observed in the first training run:
MA scored 0.0 Dice/Recall on both held-out test sets despite 0.193 dice
during cross-validation - the model learned to always predict "no MA"
since standard BCE+Dice, averaged equally across 4 classes, lets it get
away with sacrificing the rarest class.

Measured pixel-level positive ratios across the 437 pooled training images
(Refined_IDRiD train + DDR lesion_segmentation train):
    MA: 0.000252  (rarest - 12.6x rarer than HE, 2.4x rarer than SE)
    HE: 0.003168
    EX: 0.001954
    SE: 0.000601

Three changes, combined:
  1. Per-class BCE pos_weight, using sqrt-dampened inverse frequency
     (raw inverse frequency for MA is ~3963x, which would destabilize
     training - sqrt dampening keeps it strong but trainable, capped at 50).
  2. Tversky loss (alpha=0.3, beta=0.7) instead of plain Dice - penalizes
     false negatives more than false positives, directly targeting MA's
     zero-recall problem.
  3. An explicit extra per-class loss weight so MA's contribution to the
     total gradient can't be washed out by the three easier classes.
"""

import torch
import torch.nn as nn

CLASSES = ["MA", "HE", "EX", "SE"]

# sqrt(1/positive_ratio), capped at 50 to avoid destabilizing training
POS_WEIGHT = {
    "MA": 50.0,   # sqrt(3962.5) = 62.95, capped
    "HE": 17.8,
    "EX": 22.6,
    "SE": 40.8,
}

# extra multiplier on top of pos_weight, so MA's loss term can't be
# outweighed by the three easier/larger-lesion classes during averaging
CLASS_LOSS_WEIGHT = {
    "MA": 3.0,
    "HE": 1.0,
    "EX": 1.0,
    "SE": 1.0,
}

TVERSKY_ALPHA = 0.3   # false-positive penalty weight
TVERSKY_BETA = 0.7    # false-negative penalty weight (higher = favor recall)


def tversky_loss_per_channel(logits, targets, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, eps=1e-6):
    """
    logits, targets: (B, C, H, W). Returns a (C,) tensor - Tversky loss
    per class, NOT reduced to a scalar, so the caller can apply
    per-class weights before summing.
    """
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)  # reduce over batch + spatial, keep channel dim
    tp = (probs * targets).sum(dim=dims)
    fp = (probs * (1 - targets)).sum(dim=dims)
    fn = ((1 - probs) * targets).sum(dim=dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - tversky  # shape (C,)


class WeightedMultiLesionLoss(nn.Module):
    """
    Combined per-class weighted BCE + Tversky loss.
    classes: list of class names, in the same channel order as the model's
             output (must match CLASSES order used everywhere else).
    """

    def __init__(self, classes=CLASSES, device="cpu"):
        super().__init__()
        self.classes = classes
        pos_weight_tensor = torch.tensor(
            [POS_WEIGHT[c] for c in classes], dtype=torch.float32
        ).view(1, -1, 1, 1).to(device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")
        self.class_weight = torch.tensor(
            [CLASS_LOSS_WEIGHT[c] for c in classes], dtype=torch.float32
        ).to(device)

    def forward(self, logits, targets):
        # per-pixel weighted BCE, then mean per channel -> (C,)
        bce_per_pixel = self.bce(logits, targets)
        bce_per_channel = bce_per_pixel.mean(dim=(0, 2, 3))

        tversky_per_channel = tversky_loss_per_channel(logits, targets)

        per_channel_loss = bce_per_channel + tversky_per_channel  # (C,)
        weighted = per_channel_loss * self.class_weight
        return weighted.sum() / self.class_weight.sum()
