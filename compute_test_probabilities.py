"""
LesionBridge - probability export for ROC/AUC + calibration analysis.

train_classification.py's --stage test only reports argmax predictions
(accuracy/macro-F1/QWK). ROC/AUC and calibration (ECE, reliability diagram)
both need the FULL softmax probability vector per test image, not just the
predicted class. This script re-runs inference (frozen classifier, cheap)
on each modality's official test set and dumps per-image class
probabilities to CSV - no retraining, just a different readout of the same
trained models used in Stage 3.

Run: python compute_test_probabilities.py
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F

from train_classification import (
    BASE, MMRDR_CSV, NUM_CLASSES, CKPT_DIR, load_csv_rows,
    MMRDRFeatureDataset, ClassifierHead,
)

MODALITIES = ["CFP", "UWF", "OCT"]
OUT_DIR = BASE  # write CSVs alongside the other results_* files


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_modality(modality, device):
    num_classes = NUM_CLASSES[modality]
    all_rows = load_csv_rows(MMRDR_CSV[modality])
    test_rows = [r for r in all_rows if r["image"].split("/")[-1].startswith("ts")]
    print(f"\n=== {modality}: exporting test-set probabilities (n={len(test_rows)}) ===")

    ckpt_path = os.path.join(CKPT_DIR, f"{modality.lower()}_classifier_final.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ClassifierHead(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = MMRDRFeatureDataset(test_rows, modality)

    out_path = os.path.join(OUT_DIR, f"results_stage3_{modality.lower()}_test_probs.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_label"] + [f"prob_{c}" for c in range(num_classes)])
        with torch.no_grad():
            for idx, row in enumerate(test_rows):
                x, y = dataset[idx]
                x = x.unsqueeze(0).to(device)
                logits = model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                fname = row["image"].split("/")[-1]
                w.writerow([fname, y] + [f"{p:.6f}" for p in probs])

    print(f"  written -> {out_path}")


def main():
    device = get_device()
    print(f"Device: {device}")
    for modality in MODALITIES:
        run_modality(modality, device)
    print("\nAll modalities done. These CSVs feed plot_roc_calibration.py next.")


if __name__ == "__main__":
    main()
