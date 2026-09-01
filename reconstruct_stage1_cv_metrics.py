"""
LesionBridge - reconstruct per-fold, per-class Stage 1 CV metrics.

train_segmentation.py's 5-fold CV only ever printed per-epoch dice to the
console and saved each fold's BEST checkpoint (checkpoints/segmentation/
fold{0..4}_best.pt) - it never wrote the full 10-metric breakdown to a
file. Those 5 checkpoints already exist on disk, so this script re-runs
INFERENCE ONLY (no retraining) for each fold's best checkpoint against its
own held-out validation rows, using the same 10-metric suite
(segmentation_metrics.py) already used for the official test-set
evaluation. This is what feeds the Stage 1 CV violin plot - real per-fold,
per-class Dice (and the other 9 metrics), not reconstructed/approximated
from the printed summary numbers.

Run: python reconstruct_stage1_cv_metrics.py
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_segmentation import (
    BASE, FOLDS_CSV, CLASSES, CKPT_DIR, get_device,
    load_fold_rows, LesionSegDataset, get_transforms, build_model,
)
from segmentation_metrics import compute_all_metrics_for_class

N_FOLDS = 5
EVAL_BATCH_SIZE = 4


def evaluate_fold(fold_idx, device):
    rows = load_fold_rows(FOLDS_CSV)
    val_rows = [r for r in rows if int(r["fold"]) == fold_idx]
    print(f"\n=== Fold {fold_idx}: re-evaluating held-out val set (n={len(val_rows)}) ===")

    ckpt_path = os.path.join(CKPT_DIR, f"fold{fold_idx}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_ds = LesionSegDataset(val_rows, transform=get_transforms(train=False))
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=2)

    # collect per-image probability maps and targets, per class, across the whole fold
    per_class_probs = {c: [] for c in CLASSES}
    per_class_targets = {c: [] for c in CLASSES}

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 4, H, W)
            masks_np = masks.numpy()                      # (B, 4, H, W)
            for b in range(probs.shape[0]):
                for i, cls in enumerate(CLASSES):
                    per_class_probs[cls].append(probs[b, i])
                    per_class_targets[cls].append(masks_np[b, i])

    fold_results = []
    for cls in CLASSES:
        metrics = compute_all_metrics_for_class(per_class_probs[cls], per_class_targets[cls])
        metrics["fold"] = fold_idx
        metrics["class"] = cls
        fold_results.append(metrics)
        print(f"  {cls}: Dice={metrics['Dice']:.4f} IoU={metrics['IoU']:.4f} "
              f"MCC={metrics['MCC']:.4f}")

    del model
    torch.cuda.empty_cache()
    return fold_results


def main():
    device = get_device()
    print(f"Device: {device}")

    all_results = []
    for fold_idx in range(N_FOLDS):
        all_results.extend(evaluate_fold(fold_idx, device))

    out_csv = os.path.join(BASE, "results_stage1_cv_per_fold_metrics.csv")
    fieldnames = ["fold", "class", "Dice", "IoU", "Precision", "Recall_Sensitivity",
                  "Specificity", "HD95_pixels", "MCC", "Lesion_level_Sensitivity",
                  "Volumetric_Similarity", "ECE"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\nWritten -> {out_csv}")

    print("\nPer-class Dice mean +/- std across the 5 folds:")
    for cls in CLASSES:
        vals = [r["Dice"] for r in all_results if r["class"] == cls]
        print(f"  {cls}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  (folds: {[round(v,4) for v in vals]})")


if __name__ == "__main__":
    main()
