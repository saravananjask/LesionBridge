"""
Lesion-attention pooling for LesionBridge Stage 3.

RETFound's ViT-L/16 @ 224 produces a 14x14 grid of patch tokens (196 total,
16x16 pixels each in input space). Stage 2's pseudo-lesion-masks are
384x384, 4-channel (MA/HE/EX/SE) probability maps. This module downsamples
a mask to match the 14x14 patch grid and turns it into attention weights
over the patch tokens, so the classifier pools RETFound's features toward
image regions the segmenter flagged as lesion-containing, rather than
naive uniform/CLS-only pooling.
"""

import numpy as np
import torch
import torch.nn.functional as F

PATCH_GRID = 14  # 224 / 16


def mask_to_patch_attention(mask_4ch_uint8):
    """
    mask_4ch_uint8: (4, 384, 384) uint8 array (as saved by generate_pseudo_masks.py,
                     probability x 255 per class channel: MA, HE, EX, SE)
    Returns: (196,) float32 array, softmax-normalized attention weight per patch,
             computed as the max lesion probability (across the 4 classes) in
             that patch region - "how lesion-suspicious is this patch."
    """
    mask = mask_4ch_uint8.astype(np.float32) / 255.0  # (4, 384, 384), in [0,1]
    combined = mask.max(axis=0)  # (384, 384) - max across lesion classes

    tensor = torch.from_numpy(combined).unsqueeze(0).unsqueeze(0)  # (1,1,384,384)
    pooled = F.adaptive_avg_pool2d(tensor, (PATCH_GRID, PATCH_GRID))  # (1,1,14,14)
    weights = pooled.flatten().numpy()  # (196,)

    # softmax normalize so weights sum to 1 (a valid attention distribution);
    # add a small uniform floor first so patches with zero lesion signal
    # still get *some* weight rather than being fully zeroed out
    weights = weights + 1e-3
    exp_w = np.exp(weights - weights.max())
    attn = exp_w / exp_w.sum()
    return attn.astype(np.float32)


def attention_pool_patches(patch_tokens, attn_weights):
    """
    patch_tokens: (196, 1024) float array
    attn_weights: (196,) float array, sums to 1
    Returns: (1024,) weighted-average feature vector
    """
    return (patch_tokens * attn_weights[:, None]).sum(axis=0)


def uniform_pool_patches(patch_tokens):
    """Fallback for modalities with no pseudo-mask (OCT): plain mean pooling."""
    return patch_tokens.mean(axis=0)
