"""
LesionBridge - remaining paper figures (everything not already produced by
generate_pseudo_masks.py's review grid, plot_roc_calibration.py, or
xai_attention_comparison.py).

MUST be run AFTER reconstruct_stage1_cv_metrics.py (needs
results_stage1_cv_per_fold_metrics.csv) and after Stage 3's test-probability
export (needs results_stage3_{cfp,uwf,oct}_test_probs.csv) and the external
validation run (needs results_external_validation_uwf_predictions.csv).

Produces (all under figures/):
  fig1_dataset_overview.png       - sample images per modality + DR grade distribution
  fig3b_qualitative_segmentation.png - prediction vs ground-truth overlay, Stage 1
  fig3c_stage1_cv_violin.png      - violin plot, 5-fold CV Dice per lesion class
  fig5_stage3_training_curves.png - Stage 3 classifier training loss (CFP/UWF/OCT)
  fig6_stage3_cv_qwk.png          - CFP violin (true per-fold) + UWF/OCT mean+-std bars
  fig7_stage3_confusion_matrices.png - CFP/UWF/OCT official-test confusion matrices
  fig10_generalization_summary.png   - external validation confusion matrix,
                                        CV/test/external QWK comparison, and
                                        raw-5-class vs collapsed-3-class distribution

NOTE on fig5/fig6 data provenance: Stage 3's CV/final training runs only
printed loss/QWK to the console (not logged to file) when they were
originally run. The numeric values used below for CFP's 5 per-fold QWKs and
for all three modalities' final-model training-loss checkpoints (epochs
10/20/30/40/50) are the actual values that were printed in those runs and
reported back in this conversation - transcribed here, not fabricated or
estimated. UWF and OCT's per-fold QWK breakdowns were never printed
individually (only the final mean+-std summary line was), so those two are
shown as mean+-std bars rather than true violins; the module docstring in
train_classification.py can be extended to log per-fold CSVs if a true
violin is wanted for the paper, but that would require re-running their
5-fold CV.

Run: python generate_remaining_figures.py
"""

import os
import csv
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score

from train_segmentation import (
    BASE, IMG_SIZE, CLASSES, RIDRID_LABEL_IDS, get_device, get_transforms, build_model,
)
from train_classification import MMRDR_CSV, load_csv_rows, NUM_CLASSES
from evaluate_test_sets import RefinedIDRiDTestDataset, DDRTestDataset
from extract_retfound_features import IMG_DIRS as MMRDR_IMG_DIRS

import torch

FIG_DIR = os.path.join(BASE, "figures")
FOLDS_DIR = os.path.join(BASE, "folds")
SEG_CKPT_PATH = os.path.join(BASE, "checkpoints", "segmentation", "final_model.pt")

# --- real, previously-printed data, transcribed from this conversation (see docstring) ---
CFP_CV_FOLD_QWKS = [0.8501, 0.8300, 0.8417, 0.8370, 0.8341]
UWF_CV_MEAN_STD = (0.7304, 0.0050)
OCT_CV_MEAN_STD = (0.7630, 0.0156)

STAGE3_FINAL_TRAIN_LOSS = {
    "CFP": {10: 0.6437, 20: 0.5906, 30: 0.5445, 40: 0.5100, 50: 0.4785},
    "UWF": {10: 1.1634, 20: 1.1110, 30: 1.0789, 40: 1.0707, 50: 1.0501},
    "OCT": {10: 0.4332, 20: 0.3938, 30: 0.3533, 40: 0.3155, 50: 0.2763},
}


# ---------------------------------------------------------------------------
# Fig 1: dataset overview
# ---------------------------------------------------------------------------

def fig1_dataset_overview():
    print("\n=== Fig 1: dataset overview ===")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    # (a) top row: one sample image per source
    samples = [
        ("MMRDR - CFP", os.path.join(MMRDR_IMG_DIRS["CFP"], sorted(os.listdir(MMRDR_IMG_DIRS["CFP"]))[0])),
        ("MMRDR - UWF", os.path.join(MMRDR_IMG_DIRS["UWF"], sorted(os.listdir(MMRDR_IMG_DIRS["UWF"]))[0])),
        ("MMRDR - OCT", os.path.join(MMRDR_IMG_DIRS["OCT"], sorted(os.listdir(MMRDR_IMG_DIRS["OCT"]))[0])),
    ]
    for ax, (title, path) in zip(axes[0], samples):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # (b) bottom-left/mid: Refined_IDRiD / DDR sample with lesion annotation overlay
    ridrid_img_dir = os.path.join(BASE, "Refined_IDRiD", "Train", "Images")
    ridrid_label_dir = os.path.join(BASE, "Refined_IDRiD", "Train", "Labels")
    ridrid_files = sorted(os.listdir(ridrid_img_dir))
    fname = ridrid_files[0]
    stem = os.path.splitext(fname)[0]
    img = cv2.cvtColor(cv2.imread(os.path.join(ridrid_img_dir, fname)), cv2.COLOR_BGR2RGB)
    unified = cv2.imread(os.path.join(ridrid_label_dir, f"{stem}_vessel.png"), cv2.IMREAD_GRAYSCALE)
    overlay = img.copy()
    colors = {"MA": (255, 0, 0), "HE": (0, 255, 0), "EX": (0, 0, 255), "SE": (255, 255, 0)}
    for cls in CLASSES:
        m = (unified == RIDRID_LABEL_IDS[cls])
        overlay[m] = colors[cls]
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    axes[1, 0].imshow(blended)
    axes[1, 0].set_title("Refined_IDRiD (lesion annotations)", fontsize=10)
    axes[1, 0].axis("off")

    # (c) bottom-mid: DR grade distribution across MMRDR modalities
    grade_counts = {}
    for modality in ["CFP", "UWF", "OCT"]:
        fold_rows = load_csv_rows(os.path.join(FOLDS_DIR, f"mmrdr_{modality.lower()}_folds.csv"))
        all_test_rows = load_csv_rows(MMRDR_CSV[modality])
        test_rows = [r for r in all_test_rows if r["image"].split("/")[-1].startswith("ts")]
        grades = [int(r["grade"]) for r in fold_rows] + [int(r["grade"]) for r in test_rows]
        counts = np.bincount(grades, minlength=NUM_CLASSES[modality])
        grade_counts[modality] = counts

    ax = axes[1, 1]
    max_classes = max(len(v) for v in grade_counts.values())
    width = 0.25
    x = np.arange(max_classes)
    for i, (modality, counts) in enumerate(grade_counts.items()):
        padded = np.pad(counts, (0, max_classes - len(counts)))
        ax.bar(x + i * width, padded, width=width, label=modality)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"grade {g}" for g in range(max_classes)])
    ax.set_ylabel("image count")
    ax.set_title("DR grade distribution (MMRDR, all official rows)", fontsize=10)
    ax.legend(fontsize=8)

    axes[1, 2].axis("off")
    axes[1, 2].text(0.05, 0.5,
                     "Dataset summary:\n"
                     "- MMRDR: CFP/UWF/OCT, weak image-level labels\n"
                     "- Refined_IDRiD + DDR: pixel-level lesion masks\n"
                     "  (MA/HE/EX/SE), used for Stage 1 training\n"
                     "- UWF_fundus_dataset: fully independent cohort,\n"
                     "  used only for external validation",
                     fontsize=9, va="center", transform=axes[1, 2].transAxes)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig1_dataset_overview.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Fig 3b: qualitative segmentation overlay (final model, official test sets)
# ---------------------------------------------------------------------------

def fig3b_qualitative_segmentation(device, n_per_source=3):
    print("\n=== Fig 3b: qualitative segmentation overlay ===")
    ckpt = torch.load(SEG_CKPT_PATH, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = get_transforms(train=False)
    ridrid_test = RefinedIDRiDTestDataset(transform=transform)
    ddr_test = DDRTestDataset(transform=transform)

    random.seed(42)
    sources = [("Refined_IDRiD test", ridrid_test), ("DDR test", ddr_test)]
    n_rows = n_per_source * len(sources)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))

    colors = {"MA": (1, 0, 0), "HE": (0, 1, 0), "EX": (0, 0, 1), "SE": (1, 1, 0)}
    row = 0
    with torch.no_grad():
        for name, ds in sources:
            idxs = random.sample(range(len(ds)), min(n_per_source, len(ds)))
            for idx in idxs:
                image, mask = ds[idx]
                logits = model(image.unsqueeze(0).to(device))
                probs = torch.sigmoid(logits).cpu().numpy()[0]  # (4, H, W)

                # de-normalize just for display (ImageNet stats used in get_transforms)
                img_disp = image.permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_disp = np.clip(img_disp * std + mean, 0, 1)

                gt_overlay = img_disp.copy()
                pred_overlay = img_disp.copy()
                for i, cls in enumerate(CLASSES):
                    gt_overlay[mask[i].numpy() > 0.5] = colors[cls]
                    pred_overlay[probs[i] > 0.5] = colors[cls]
                gt_blend = 0.6 * img_disp + 0.4 * gt_overlay
                pred_blend = 0.6 * img_disp + 0.4 * pred_overlay

                axes[row, 0].imshow(img_disp)
                axes[row, 0].set_title(f"{name} - original", fontsize=8)
                axes[row, 1].imshow(gt_blend)
                axes[row, 1].set_title("ground truth", fontsize=8)
                axes[row, 2].imshow(pred_blend)
                axes[row, 2].set_title("prediction", fontsize=8)
                for c in range(3):
                    axes[row, c].axis("off")
                row += 1

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig3b_qualitative_segmentation.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  -> {out_path}")
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Fig 3c: Stage 1 CV violin plot (per-class Dice across 5 folds)
# ---------------------------------------------------------------------------

def fig3c_stage1_cv_violin():
    print("\n=== Fig 3c: Stage 1 CV violin plot ===")
    csv_path = os.path.join(BASE, "results_stage1_cv_per_fold_metrics.csv")
    rows = load_csv_rows(csv_path)

    data = {cls: [] for cls in CLASSES}
    for r in rows:
        data[r["class"]].append(float(r["Dice"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = list(range(1, len(CLASSES) + 1))
    parts = ax.violinplot([data[c] for c in CLASSES], positions=positions, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Dice score")
    ax.set_title("Stage 1: 5-fold CV Dice per lesion class")
    for i, cls in enumerate(CLASSES):
        vals = data[cls]
        ax.scatter([positions[i]] * len(vals), vals, color="black", s=15, zorder=3, alpha=0.7)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig3c_stage1_cv_violin.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Fig 5: Stage 3 training curves
# ---------------------------------------------------------------------------

def fig5_stage3_training_curves():
    print("\n=== Fig 5: Stage 3 training curves ===")
    fig, ax = plt.subplots(figsize=(7, 5))
    for modality, losses in STAGE3_FINAL_TRAIN_LOSS.items():
        epochs = sorted(losses.keys())
        vals = [losses[e] for e in epochs]
        ax.plot(epochs, vals, marker="o", label=modality)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (cross-entropy)")
    ax.set_title("Stage 3: final classifier training loss")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig5_stage3_training_curves.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  -> {out_path}  (logged at epochs 10/20/30/40/50 only, as originally printed)")


# ---------------------------------------------------------------------------
# Fig 6: Stage 3 CV QWK - CFP true violin, UWF/OCT mean+-std bars
# ---------------------------------------------------------------------------

def fig6_stage3_cv_qwk():
    print("\n=== Fig 6: Stage 3 CV QWK summary ===")
    fig, ax = plt.subplots(figsize=(7, 5))

    parts = ax.violinplot([CFP_CV_FOLD_QWKS], positions=[1], showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.scatter([1] * len(CFP_CV_FOLD_QWKS), CFP_CV_FOLD_QWKS, color="black", s=15, zorder=3)

    for pos, (mean, std) in zip([2, 3], [UWF_CV_MEAN_STD, OCT_CV_MEAN_STD]):
        ax.bar(pos, mean, yerr=std, width=0.5, capsize=6, alpha=0.6)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["CFP\n(true per-fold violin)", "UWF\n(mean +/- std)", "OCT\n(mean +/- std)"])
    ax.set_ylabel("QWK")
    ax.set_title("Stage 3: 5-fold CV QWK per modality")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig6_stage3_cv_qwk.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Fig 7: Stage 3 confusion matrices (official test sets)
# ---------------------------------------------------------------------------

def load_test_probs(modality):
    path = os.path.join(BASE, f"results_stage3_{modality.lower()}_test_probs.csv")
    y_true, y_pred = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        prob_cols = [c for c in reader.fieldnames if c.startswith("prob_")]
        for row in reader:
            y_true.append(int(row["true_label"]))
            probs = [float(row[c]) for c in prob_cols]
            y_pred.append(int(np.argmax(probs)))
    return np.array(y_true), np.array(y_pred)


def fig7_stage3_confusion_matrices():
    print("\n=== Fig 7: Stage 3 confusion matrices ===")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, modality in zip(axes, ["CFP", "UWF", "OCT"]):
        y_true, y_pred = load_test_probs(modality)
        num_classes = NUM_CLASSES[modality]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{modality} official test", fontsize=10)
        ax.set_xlabel("predicted grade")
        ax.set_ylabel("true grade")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig7_stage3_confusion_matrices.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Fig 10: generalization summary (external validation)
# ---------------------------------------------------------------------------

def fig10_generalization_summary():
    print("\n=== Fig 10: generalization summary ===")
    preds_path = os.path.join(BASE, "results_external_validation_uwf_predictions.csv")
    rows = load_csv_rows(preds_path)
    true3 = np.array([int(r["true_label3"]) for r in rows])
    pred3 = np.array([int(r["pred_label3_mapped"]) for r in rows])
    pred5_raw = np.array([int(r["pred_label5_raw"]) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) confusion matrix
    cm = confusion_matrix(true3, pred3, labels=[0, 1, 2])
    ax = axes[0]
    ax.imshow(cm, cmap="Oranges")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Normal", "NPDR", "PDR"])
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Normal", "NPDR", "PDR"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("External validation confusion matrix\n(UWF_fundus_dataset, zero-shot)", fontsize=9)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)

    # (b) QWK comparison: CV vs official test vs external, per modality
    qwk_cv = {"CFP": np.mean(CFP_CV_FOLD_QWKS), "UWF": UWF_CV_MEAN_STD[0], "OCT": OCT_CV_MEAN_STD[0]}
    qwk_test = {"CFP": 0.8146, "UWF": 0.7326, "OCT": 0.8137}
    qwk_external = {"UWF": cohen_kappa_score(true3, pred3, weights="quadratic", labels=[0, 1, 2])}

    ax = axes[1]
    modalities = ["CFP", "UWF", "OCT"]
    x = np.arange(len(modalities))
    width = 0.25
    ax.bar(x - width, [qwk_cv[m] for m in modalities], width, label="CV mean")
    ax.bar(x, [qwk_test[m] for m in modalities], width, label="official test")
    ext_vals = [qwk_external.get(m, 0) for m in modalities]
    ax.bar(x + width, ext_vals, width, label="external (UWF only)")
    ax.set_xticks(x); ax.set_xticklabels(modalities)
    ax.set_ylabel("QWK")
    ax.set_ylim(0, 1)
    ax.set_title("QWK: CV vs. official test vs. external validation", fontsize=9)
    ax.legend(fontsize=8)

    # (c) raw 5-class vs collapsed 3-class prediction distribution (external)
    ax = axes[2]
    from collections import Counter
    dist5 = Counter(pred5_raw.tolist())
    grades5 = sorted(dist5.keys())
    ax.bar([str(g) for g in grades5], [dist5[g] for g in grades5], color="steelblue")
    ax.set_xlabel("raw 5-class predicted grade")
    ax.set_ylabel("count")
    ax.set_title("External validation: raw 5-class\nprediction distribution", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig10_generalization_summary.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  -> {out_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    fig1_dataset_overview()
    fig3b_qualitative_segmentation(device)
    fig3c_stage1_cv_violin()
    fig5_stage3_training_curves()
    fig6_stage3_cv_qwk()
    fig7_stage3_confusion_matrices()
    fig10_generalization_summary()

    print("\nAll remaining figures generated under:", FIG_DIR)


if __name__ == "__main__":
    main()
