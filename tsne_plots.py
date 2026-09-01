"""
LesionBridge - t-SNE visualizations of Stage 3's fused feature space.

No retraining, no new model inference - projects the already-computed
fused feature vectors ([CLS embedding ; lesion-attention-pooled patch
feature], the exact 2048-dim input each classifier head receives) with
scikit-learn's t-SNE.

Produces two kinds of figure:

  1. Class-separability, per modality (CFP/UWF/OCT): t-SNE of the official
     test set's fused vectors, colored by true DR/DME grade. Shows whether
     the frozen RETFound + lesion-attention representation already
     separates severity before the classifier head acts on it.

  2. Domain-gap (UWF only, the one modality with external validation):
     t-SNE of MMRDR-UWF (official train+test pooled) and the independent
     UWF_fundus_dataset plotted together, colored by SOURCE dataset rather
     than grade. Visual evidence for why external-validation QWK (0.6220)
     is lower than in-domain test QWK (0.7326) - a real distribution
     shift between cohorts, not a modeling artifact.

Run: python tsne_plots.py
"""

import os
import random
import numpy as np
import cv2
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_classification import (
    BASE, MMRDR_CSV, NUM_CLASSES, load_csv_rows, MMRDRFeatureDataset,
)
from lesion_attention import mask_to_patch_attention, attention_pool_patches
from external_validation import (
    load_all_rows as load_external_rows, FEATURE_DIR as EXT_FEATURE_DIR,
    PSEUDO_MASK_DIR as EXT_MASK_DIR,
)
from train_segmentation import CLASSES

FIG_DIR = os.path.join(BASE, "figures")
FOLDS_DIR = os.path.join(BASE, "folds")

RANDOM_SEED = 42
MAX_POINTS_PER_MODALITY = 2600   # official test sets are already <= this
MAX_POINTS_PER_DOMAIN_GROUP = 1200


def fused_vectors_from_dataset(rows, modality):
    ds = MMRDRFeatureDataset(rows, modality)
    X, y = [], []
    for i in range(len(ds)):
        x, grade = ds[i]
        X.append(x.numpy())
        y.append(grade)
    return np.stack(X), np.array(y)


def fused_vectors_external_uwf(rows):
    """Same fused-vector construction as external_validation.py's Phase 3,
    but just returns the vectors (no classifier prediction needed here)."""
    X = []
    kept_rows = []
    for r in rows:
        stem = os.path.splitext(r["filename"])[0]
        feat_path = os.path.join(EXT_FEATURE_DIR, f"{stem}.npz")
        mask_paths = [os.path.join(EXT_MASK_DIR, c, f"{stem}.png") for c in CLASSES]
        if not os.path.isfile(feat_path) or not all(os.path.isfile(p) for p in mask_paths):
            continue
        feat = np.load(feat_path)
        cls_emb = feat["cls"].astype(np.float32)
        patch_tokens = feat["patches"].astype(np.float32)
        mask_channels = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_paths]
        mask_4ch = np.stack(mask_channels, axis=0)
        attn = mask_to_patch_attention(mask_4ch)
        pooled = attention_pool_patches(patch_tokens, attn)
        X.append(np.concatenate([cls_emb, pooled]))
        kept_rows.append(r)
    return np.stack(X), kept_rows


def subsample(X, y, max_n, seed=RANDOM_SEED):
    if len(X) <= max_n:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx], (y[idx] if isinstance(y, np.ndarray) else [y[i] for i in idx])


def run_class_separability(modality):
    print(f"\n=== t-SNE class-separability: {modality} ===")
    all_rows = load_csv_rows(MMRDR_CSV[modality])
    test_rows = [r for r in all_rows if r["image"].split("/")[-1].startswith("ts")]
    X, y = fused_vectors_from_dataset(test_rows, modality)
    X, y = subsample(X, y, MAX_POINTS_PER_MODALITY)
    print(f"  n={len(X)} points, feature dim={X.shape[1]}")

    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, init="pca")
    emb = tsne.fit_transform(X)

    num_classes = NUM_CLASSES[modality]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    cmap = plt.cm.viridis(np.linspace(0, 1, num_classes))
    for g in range(num_classes):
        mask = np.array(y) == g
        ax.scatter(emb[mask, 0], emb[mask, 1], s=10, color=cmap[g], label=f"grade {g}", alpha=0.7)
    ax.set_title(f"{modality} test-set fused features (t-SNE), colored by true grade")
    ax.legend(fontsize=8, markerscale=1.5)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"tsne_{modality.lower()}_class_separability.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  -> {out_path}")


def run_domain_gap_uwf():
    print("\n=== t-SNE domain gap: MMRDR-UWF vs external UWF_fundus_dataset ===")

    fold_rows = load_csv_rows(os.path.join(FOLDS_DIR, "mmrdr_uwf_folds.csv"))
    all_rows = load_csv_rows(MMRDR_CSV["UWF"])
    test_rows = [r for r in all_rows if r["image"].split("/")[-1].startswith("ts")]
    mmrdr_rows = fold_rows + test_rows
    X_mmrdr, _ = fused_vectors_from_dataset(mmrdr_rows, "UWF")
    X_mmrdr, _ = subsample(X_mmrdr, np.zeros(len(X_mmrdr)), MAX_POINTS_PER_DOMAIN_GROUP)
    print(f"  MMRDR-UWF sampled: {len(X_mmrdr)} points")

    ext_rows = load_external_rows()
    ext_rows_sample = ext_rows
    if len(ext_rows_sample) > MAX_POINTS_PER_DOMAIN_GROUP:
        random.seed(RANDOM_SEED)
        ext_rows_sample = random.sample(ext_rows_sample, MAX_POINTS_PER_DOMAIN_GROUP)
    X_ext, _ = fused_vectors_external_uwf(ext_rows_sample)
    print(f"  External UWF_fundus_dataset sampled: {len(X_ext)} points")

    X = np.concatenate([X_mmrdr, X_ext], axis=0)
    source = np.array(["MMRDR-UWF"] * len(X_mmrdr) + ["External (UWF_fundus_dataset)"] * len(X_ext))

    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, init="pca")
    emb = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color in [("MMRDR-UWF", "tab:blue"), ("External (UWF_fundus_dataset)", "tab:red")]:
        mask = source == label
        ax.scatter(emb[mask, 0], emb[mask, 1], s=10, color=color, label=label, alpha=0.6)
    ax.set_title("UWF feature space: MMRDR (in-domain) vs.\nexternal cohort (t-SNE, colored by source)")
    ax.legend(fontsize=9)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "tsne_uwf_domain_gap.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  -> {out_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    for modality in ["CFP", "UWF", "OCT"]:
        run_class_separability(modality)
    run_domain_gap_uwf()
    print("\nAll t-SNE figures generated.")


if __name__ == "__main__":
    main()
