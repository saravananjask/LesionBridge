"""
LesionBridge - XAI: lesion-conditioned attention vs. vanilla RETFound
self-attention.

Motivation / what this actually demonstrates (important for the paper -
read before reporting numbers): our Stage 3 classifier pools RETFound's
patch tokens using attention DERIVED FROM Stage 1's pseudo-lesion-masks
(see lesion_attention.py). Because that attention is built directly from
the lesion mask, it aligns with lesion regions by construction - reporting
"our attention overlaps with the lesion mask" would be circular, not a
finding. The genuinely informative comparison is the other direction: does
RETFound's OWN frozen self-attention (with no lesion guidance at all)
already look at lesion regions on its own, or does it need our explicit
conditioning? We answer that by extracting RETFound's vanilla CLS-to-patch
attention via attention rollout (Abnar & Zuidema, 2020 - standard,
citable technique: average attention heads per layer, add the residual
identity term, renormalize, chain-multiply across layers, read off the
CLS row) and measuring how much it overlaps with lesion regions, with NO
lesion information involved in computing it.

Produces:
  1. Qualitative figure per modality (CFP/UWF only - OCT has no lesion
     mask/attention mechanism, see generate_pseudo_masks.py): a few sample
     test images, each shown as (original | vanilla ViT attention overlay |
     our lesion-conditioned attention overlay).
  2. A quantitative table: mean IoU between the top-20%-attended patches
     under vanilla ViT attention vs. the top-20% highest-lesion-probability
     patches (from Stage 1's pseudo-mask), averaged over a sample of test
     images per modality. Low overlap here is the expected, useful result -
     it is the empirical justification for why LesionBridge's explicit
     lesion-conditioning step is needed rather than assuming a frozen
     foundation model already attends to lesions unprompted.

Run: python xai_attention_comparison.py
"""

import os
import csv
import random
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoImageProcessor

from train_segmentation import BASE, CLASSES
from train_classification import MMRDR_CSV, load_csv_rows
from extract_retfound_features import RETFOUND_REPOS, IMG_DIRS
from lesion_attention import PATCH_GRID
import torch.nn.functional as F

MODALITIES = ["CFP", "UWF"]  # OCT excluded: no pseudo-mask / lesion-attention exists for it
PSEUDO_MASK_DIR = os.path.join(BASE, "MMRDR_pseudo_masks")
FIG_DIR = os.path.join(BASE, "figures")

N_QUALITATIVE = 4        # sample images per modality for the visual figure
N_QUANT_SAMPLE = 150      # sample images per modality for the alignment score
TOP_FRACTION = 0.20       # "top-20% attended/lesion patches" definition
RANDOM_SEED = 42


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_rows(modality):
    all_rows = load_csv_rows(MMRDR_CSV[modality])
    return [r for r in all_rows if r["image"].split("/")[-1].startswith("ts")]


def raw_lesion_signal(mask_4ch_uint8):
    """
    Same pooling lesion_attention.mask_to_patch_attention() does, but
    returns the pre-softmax pooled signal (196,) instead of an attention
    distribution - needed here to independently rank patches by lesion
    probability for the alignment metric.
    """
    mask = mask_4ch_uint8.astype(np.float32) / 255.0
    combined = mask.max(axis=0)
    tensor = torch.from_numpy(combined).unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(tensor, (PATCH_GRID, PATCH_GRID))
    return pooled.flatten().numpy()


def attention_rollout(attentions):
    """
    attentions: tuple of per-layer tensors, each (1, num_heads, tokens, tokens),
    as returned by a HF ViTModel(..., output_attentions=True).
    Standard rollout (Abnar & Zuidema 2020): fuse heads by mean, add the
    residual-connection identity term, renormalize each row to sum to 1,
    then chain-multiply across layers. Returns the CLS token's attention
    to all 196 patch tokens (CLS-to-CLS excluded), as a (196,) array.
    """
    num_tokens = attentions[0].shape[-1]
    result = torch.eye(num_tokens)
    for attn in attentions:
        fused = attn.mean(dim=1)[0]  # (tokens, tokens), batch=1
        fused = fused + torch.eye(num_tokens)
        fused = fused / fused.sum(dim=-1, keepdim=True)
        result = fused @ result
    cls_to_patches = result[0, 1:]  # drop CLS-to-CLS
    return cls_to_patches.numpy()


def top_k_indices(values, frac):
    k = max(1, round(len(values) * frac))
    return set(np.argsort(values)[-k:].tolist())


def iou(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def overlay_heatmap(ax, img_rgb, attn_14, title):
    grid = attn_14.reshape(PATCH_GRID, PATCH_GRID)
    grid_norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    heat = cv2.resize(grid_norm, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    ax.imshow(img_rgb)
    ax.imshow(heat, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def run_modality(modality, device):
    print(f"\n=== {modality} ===")
    repo = RETFOUND_REPOS[modality]
    img_dir = IMG_DIRS[modality]
    mask_dir = os.path.join(PSEUDO_MASK_DIR, modality)

    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo, attn_implementation="eager").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    test_rows = load_test_rows(modality)
    random.seed(RANDOM_SEED)
    quant_sample = random.sample(test_rows, min(N_QUANT_SAMPLE, len(test_rows)))

    # --- quantitative alignment score (vanilla ViT attention vs lesion regions) ---
    ious = []
    with torch.no_grad():
        for r in quant_sample:
            fname = r["image"].split("/")[-1]
            stem = os.path.splitext(fname)[0]
            img_bgr = cv2.imread(os.path.join(img_dir, fname))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            inputs = processor(images=img_rgb, return_tensors="pt").to(device)
            out = model(**inputs, output_attentions=True)
            vanilla_attn = attention_rollout([a.cpu() for a in out.attentions])

            mask_channels = [
                cv2.imread(os.path.join(mask_dir, c, f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
                for c in CLASSES
            ]
            mask_4ch = np.stack(mask_channels, axis=0)
            lesion_signal = raw_lesion_signal(mask_4ch)

            vanilla_top = top_k_indices(vanilla_attn, TOP_FRACTION)
            lesion_top = top_k_indices(lesion_signal, TOP_FRACTION)
            ious.append(iou(vanilla_top, lesion_top))

    mean_iou = float(np.mean(ious))
    print(f"  vanilla-ViT-attention <-> lesion-region IoU (top-{int(TOP_FRACTION*100)}% patches, "
          f"n={len(quant_sample)}): mean={mean_iou:.4f} +/- {np.std(ious):.4f}")

    # --- qualitative comparison figure ---
    os.makedirs(FIG_DIR, exist_ok=True)
    qual_sample = random.sample(test_rows, min(N_QUALITATIVE, len(test_rows)))
    fig, axes = plt.subplots(len(qual_sample), 3, figsize=(9, 3 * len(qual_sample)))
    if len(qual_sample) == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for row_idx, r in enumerate(qual_sample):
            fname = r["image"].split("/")[-1]
            stem = os.path.splitext(fname)[0]
            img_bgr = cv2.imread(os.path.join(img_dir, fname))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            inputs = processor(images=img_rgb, return_tensors="pt").to(device)
            out = model(**inputs, output_attentions=True)
            vanilla_attn = attention_rollout([a.cpu() for a in out.attentions])

            mask_channels = [
                cv2.imread(os.path.join(mask_dir, c, f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
                for c in CLASSES
            ]
            mask_4ch = np.stack(mask_channels, axis=0)
            from lesion_attention import mask_to_patch_attention
            our_attn = mask_to_patch_attention(mask_4ch)

            axes[row_idx, 0].imshow(img_rgb)
            axes[row_idx, 0].set_title(f"{fname} (grade {r['grade']})", fontsize=9)
            axes[row_idx, 0].axis("off")
            overlay_heatmap(axes[row_idx, 1], img_rgb, vanilla_attn, "vanilla ViT attention")
            overlay_heatmap(axes[row_idx, 2], img_rgb, our_attn, "our lesion-conditioned attention")

    plt.suptitle(f"{modality}: vanilla vs. lesion-conditioned attention", fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, f"xai_attention_comparison_{modality.lower()}.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  qualitative figure -> {fig_path}")

    del model
    torch.cuda.empty_cache()
    return mean_iou, float(np.std(ious)), len(quant_sample)


def main():
    device = get_device()
    print(f"Device: {device}")

    summary = []
    for modality in MODALITIES:
        mean_iou, std_iou, n = run_modality(modality, device)
        summary.append({"modality": modality, "n": n, "mean_iou": mean_iou, "std_iou": std_iou})

    out_csv = os.path.join(BASE, "results_xai_attention_alignment.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["modality", "n", "mean_iou", "std_iou"])
        w.writeheader()
        w.writerows(summary)
    print(f"\nAlignment summary written -> {out_csv}")
    print("Note: this measures vanilla RETFound attention's overlap with lesion regions,")
    print("NOT our method's overlap (which is circular by construction - see module docstring).")


if __name__ == "__main__":
    main()
