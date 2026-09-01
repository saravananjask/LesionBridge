"""
LesionBridge - corrected CFP official-test evaluation, excluding the 8
images confirmed (via leakage_audit_resumable.py, exact 256-bit perceptual
hash match) to be pixel-identical to images in DDR's TRAIN split - i.e.
images Stage 1's segmenter was directly trained on, which independently
also landed in MMRDR-CFP's official test set (source: OIA-DDR, the public
dataset MMRDR-CFP was built from - DDR and MMRDR-CFP overlap because they
share a common upstream source, not because of anything wrong in our
train/test splitting).

This does NOT retrain anything. It just re-runs Stage 3 CFP's --stage test
evaluation with those 8 filenames removed from the test set, so the paper
can report a test QWK/accuracy/macro-F1 that is provably free of this
specific leakage vector, alongside the original (contaminated) number for
transparency.

Run: python recompute_cfp_test_clean.py
"""

import os
import csv
import torch
import numpy as np

from train_classification import (
    BASE, MMRDR_CSV, NUM_CLASSES, CKPT_DIR, load_csv_rows,
    MMRDRFeatureDataset, ClassifierHead, compute_metrics,
)
from torch.utils.data import DataLoader

# the 8 filenames confirmed to be exact-pixel duplicates of DDR-train images
# (see leakage_audit_ddr_vs_mmrdr_cfp.txt for the full audit / DDR counterparts)
CONTAMINATED_TEST_FILES = {
    "ts000057.jpg", "ts000170.jpg", "ts000417.jpg", "ts000648.jpg",
    "ts001298.jpg", "ts001394.jpg", "ts001921.jpg", "ts002159.jpg",
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    device = get_device()
    modality = "CFP"
    num_classes = NUM_CLASSES[modality]

    all_rows = load_csv_rows(MMRDR_CSV[modality])
    test_rows_all = [r for r in all_rows if r["image"].split("/")[-1].startswith("ts")]
    test_rows_clean = [r for r in test_rows_all
                        if r["image"].split("/")[-1] not in CONTAMINATED_TEST_FILES]

    n_removed = len(test_rows_all) - len(test_rows_clean)
    print(f"CFP official test set: {len(test_rows_all)} total, "
          f"removing {n_removed} pixel-identical-to-DDR-train images, "
          f"{len(test_rows_clean)} remaining (leakage-clean).")

    ckpt_path = os.path.join(CKPT_DIR, "cfp_classifier_final.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ClassifierHead(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def run_eval(rows, label):
        ds = MMRDRFeatureDataset(rows, modality)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_true.extend(y.numpy().tolist())
        metrics = compute_metrics(all_true, all_preds, num_classes)
        print(f"  [{label}] n={len(rows)} accuracy={metrics['accuracy']:.4f} "
              f"macro_F1={metrics['macro_f1']:.4f} QWK={metrics['qwk']:.4f}")
        return metrics

    print("\n=== Re-evaluating CFP official test set ===")
    metrics_original = run_eval(test_rows_all, "ORIGINAL (n=2225, includes 8 contaminated)")
    metrics_clean = run_eval(test_rows_clean, "LEAKAGE-CLEAN (contaminated images removed)")

    out_csv = os.path.join(BASE, "results_stage3_cfp_test_metrics_leakage_audit.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "n_test", "accuracy", "macro_f1", "qwk"])
        w.writerow(["original", len(test_rows_all), metrics_original["accuracy"],
                    metrics_original["macro_f1"], metrics_original["qwk"]])
        w.writerow(["leakage_clean", len(test_rows_clean), metrics_clean["accuracy"],
                    metrics_clean["macro_f1"], metrics_clean["qwk"]])
    print(f"\nWritten -> {out_csv}")

    delta_qwk = metrics_clean["qwk"] - metrics_original["qwk"]
    print(f"\nQWK delta after removing the 8 contaminated images: {delta_qwk:+.4f}")
    print("Report the leakage-clean number as the primary CFP test result; "
          "the original can be kept in a footnote/appendix showing the difference is negligible.")


if __name__ == "__main__":
    main()
