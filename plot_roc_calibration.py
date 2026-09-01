"""
LesionBridge - ROC/AUC curves + calibration (reliability diagram / ECE)
for Stage 3, per modality (CFP/UWF/OCT).

Reads the per-image class-probability CSVs written by
compute_test_probabilities.py (no GPU, no model needed here - pure
metrics/plotting on already-computed probabilities).

Produces, per modality:
  - Figure: one-vs-rest ROC curves (one per class) + micro/macro-average AUC
  - Figure: reliability diagram (confidence vs empirical accuracy, 10 bins)
    with Expected Calibration Error (ECE) annotated
And an overall table CSV summarizing AUC (macro/micro) and ECE per modality.

Run: python plot_roc_calibration.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from train_classification import BASE, NUM_CLASSES

MODALITIES = ["CFP", "UWF", "OCT"]
FIG_DIR = os.path.join(BASE, "figures")
N_BINS = 10


def load_probs(modality):
    path = os.path.join(BASE, f"results_stage3_{modality.lower()}_test_probs.csv")
    true_labels, probs = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        prob_cols = [c for c in reader.fieldnames if c.startswith("prob_")]
        for row in reader:
            true_labels.append(int(row["true_label"]))
            probs.append([float(row[c]) for c in prob_cols])
    return np.array(true_labels), np.array(probs)


def plot_roc(modality, y_true, y_prob, num_classes):
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2:  # label_binarize collapses to 1 column for 2 classes - not our case, but guard anyway
        y_bin = np.hstack([1 - y_bin, y_bin])

    fpr, tpr, roc_auc = {}, {}, {}
    for c in range(num_classes):
        fpr[c], tpr[c], _ = roc_curve(y_bin[:, c], y_prob[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])

    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro-average
    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(6, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for c in range(num_classes):
        plt.plot(fpr[c], tpr[c], color=colors[c], lw=1.5,
                  label=f"grade {c} (AUC={roc_auc[c]:.3f})")
    plt.plot(fpr["micro"], tpr["micro"], color="deeppink", linestyle=":", lw=2.5,
              label=f"micro-avg (AUC={roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], color="navy", linestyle=":", lw=2.5,
              label=f"macro-avg (AUC={roc_auc['macro']:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{modality} - one-vs-rest ROC (official test set)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"roc_{modality.lower()}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  ROC figure -> {out_path}")

    return roc_auc["macro"], roc_auc["micro"]


def compute_ece(confidences, correctness, n_bins=N_BINS):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = in_bin.sum()
        if count == 0:
            bin_stats.append((lo, hi, 0, np.nan, np.nan))
            continue
        bin_acc = correctness[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (count / n) * abs(bin_acc - bin_conf)
        bin_stats.append((lo, hi, count, bin_acc, bin_conf))
    return ece, bin_stats


def plot_calibration(modality, y_true, y_prob):
    y_pred = y_prob.argmax(axis=1)
    confidences = y_prob.max(axis=1)
    correctness = (y_pred == y_true).astype(np.float32)

    ece, bin_stats = compute_ece(confidences, correctness)

    bin_centers = [(lo + hi) / 2 for lo, hi, *_ in bin_stats]
    bin_accs = [s[3] for s in bin_stats]

    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="perfect calibration")
    valid = [(c, a) for c, a in zip(bin_centers, bin_accs) if not np.isnan(a)]
    if valid:
        xs, ys = zip(*valid)
        plt.bar(xs, ys, width=1.0 / N_BINS * 0.9, alpha=0.7, edgecolor="black",
                label="empirical accuracy")
    plt.xlabel("Confidence (max predicted probability)")
    plt.ylabel("Empirical accuracy")
    plt.title(f"{modality} - reliability diagram (ECE={ece:.4f})")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"calibration_{modality.lower()}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Calibration figure -> {out_path} (ECE={ece:.4f})")

    return ece


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    summary_rows = []

    for modality in MODALITIES:
        print(f"\n=== {modality} ===")
        y_true, y_prob = load_probs(modality)
        num_classes = NUM_CLASSES[modality]

        macro_auc, micro_auc = plot_roc(modality, y_true, y_prob, num_classes)
        ece = plot_calibration(modality, y_true, y_prob)

        summary_rows.append({
            "modality": modality, "n_test": len(y_true),
            "macro_auc": macro_auc, "micro_auc": micro_auc, "ece": ece,
        })

    out_csv = os.path.join(BASE, "results_stage3_roc_calibration_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["modality", "n_test", "macro_auc", "micro_auc", "ece"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary written -> {out_csv}")
    for r in summary_rows:
        print(f"  {r['modality']}: macro_AUC={r['macro_auc']:.4f} "
              f"micro_AUC={r['micro_auc']:.4f} ECE={r['ece']:.4f}")


if __name__ == "__main__":
    main()
