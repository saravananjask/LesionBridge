"""
LesionBridge - Stage 1 FINAL, one-shot test evaluation.

Evaluates the final segmentation model (trained on all 437 pooled training
images, see train_final_segmentation.py) on the two official test sets that
have NEVER been touched during training or cross-validation:
  - Refined_IDRiD/Test  (27 images)
  - DDR lesion_segmentation/test (225 images)

These are reported separately (not pooled) since they come from different
institutions/protocols - pooling them would hide whether the model
generalizes evenly or is only strong on one source.

Run: python evaluate_test_sets.py
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from train_seg_smoke import (
    BASE, IMG_SIZE, CLASSES, RIDRID_LABEL_IDS, get_device, get_transforms,
    build_model, dice_score,
)
from segmentation_metrics import compute_all_metrics_for_class

CKPT_PATH = os.path.join(BASE, "checkpoints", "segmentation", "final_model.pt")


class RefinedIDRiDTestDataset(Dataset):
    def __init__(self, transform=None):
        self.img_dir = os.path.join(BASE, "Refined_IDRiD", "Test", "Images")
        self.label_dir = os.path.join(BASE, "Refined_IDRiD", "Test", "Labels")
        self.files = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        stem = os.path.splitext(fname)[0]
        image = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, fname)), cv2.COLOR_BGR2RGB)

        # Test labels are named like "IDRiD_55f055.png" matching image stem exactly
        label_path = os.path.join(self.label_dir, f"{stem}.png")
        unified = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        mask = np.zeros((*unified.shape, len(CLASSES)), dtype=np.float32)
        for i, cls in enumerate(CLASSES):
            mask[..., i] = (unified == RIDRID_LABEL_IDS[cls]).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            mask = mask.permute(2, 0, 1).float()
        return image, mask


class DDRTestDataset(Dataset):
    def __init__(self, transform=None):
        self.img_dir = os.path.join(BASE, "DDR", "DDR-dataset", "lesion_segmentation", "test", "image")
        self.label_dir = os.path.join(BASE, "DDR", "DDR-dataset", "lesion_segmentation", "test", "label")
        self.files = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        stem = os.path.splitext(fname)[0]
        image = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, fname)), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        mask = np.zeros((h, w, len(CLASSES)), dtype=np.float32)
        for i, cls in enumerate(CLASSES):
            mask_path = os.path.join(self.label_dir, cls, f"{stem}.tif")
            if not os.path.isfile(mask_path):
                alt = os.path.join(self.label_dir, cls, f"{stem}.png")
                mask_path = alt if os.path.isfile(alt) else mask_path
            if os.path.isfile(mask_path):
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask[..., i] = (m > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            mask = mask.permute(2, 0, 1).float()
        return image, mask


def evaluate(model, dataset, name, device):
    """
    Runs the model over every image in `dataset` and computes the full
    10-metric suite per class (see segmentation_metrics.py), aggregated
    correctly across the whole test set (pooled confusion counts for the
    pixel metrics, per-image HD95 averaged over lesion-present images,
    pooled instance counts for lesion-level sensitivity, pooled pixels
    for ECE) rather than naively averaging per-batch numbers.
    """
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    model.eval()

    # collect per-class probability maps and targets for every image first,
    # then compute metrics - needed because several metrics (HD95, ECE,
    # lesion-level sensitivity) are not batch-separable averages
    per_class_probs = {c: [] for c in CLASSES}
    per_class_targets = {c: [] for c in CLASSES}

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            masks = masks.numpy()
            for b in range(probs.shape[0]):
                for i, cls in enumerate(CLASSES):
                    per_class_probs[cls].append(probs[b, i])
                    per_class_targets[cls].append(masks[b, i])

    print(f"\n=== {name} (n={len(dataset)} images) ===")
    all_class_metrics = {}
    for cls in CLASSES:
        metrics = compute_all_metrics_for_class(per_class_probs[cls], per_class_targets[cls])
        all_class_metrics[cls] = metrics
        print(f"  [{cls}] "
              f"Dice={metrics['Dice']:.4f} IoU={metrics['IoU']:.4f} "
              f"Prec={metrics['Precision']:.4f} Rec={metrics['Recall_Sensitivity']:.4f} "
              f"Spec={metrics['Specificity']:.4f}")
        hd95_str = f"{metrics['HD95_pixels']:.2f}px" if metrics['HD95_pixels'] is not None else "N/A"
        lesion_sens_str = (f"{metrics['Lesion_level_Sensitivity']:.4f}"
                            if metrics['Lesion_level_Sensitivity'] is not None else "N/A")
        print(f"        HD95={hd95_str} MCC={metrics['MCC']:.4f} "
              f"LesionSens={lesion_sens_str} VS={metrics['Volumetric_Similarity']:.4f} "
              f"ECE={metrics['ECE']:.4f}")

    mean_dice = np.mean([all_class_metrics[c]['Dice'] for c in CLASSES])
    print(f"  mean Dice across classes: {mean_dice:.4f}")
    return mean_dice, all_class_metrics


def main():
    device = get_device()
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded final model from {CKPT_PATH}")

    transform = get_transforms(train=False)

    ridrid_test = RefinedIDRiDTestDataset(transform=transform)
    ddr_test = DDRTestDataset(transform=transform)

    ridrid_result = evaluate(model, ridrid_test, "Refined_IDRiD Test (held-out, never touched)", device)
    ddr_result = evaluate(model, ddr_test, "DDR lesion_segmentation Test (held-out, never touched)", device)

    print("\n=== Summary for paper (mean Dice) ===")
    print(f"  Refined_IDRiD test: {ridrid_result[0]:.4f}")
    print(f"  DDR test:           {ddr_result[0]:.4f}")
    print("  (report both separately - do not pool, they're different institutions/protocols)")

    # write all 10 metrics x 4 classes x 2 test sets to a CSV for the paper's tables
    out_csv = os.path.join(BASE, "results_stage1_test_metrics.csv")
    metric_names = ["Dice", "IoU", "Precision", "Recall_Sensitivity", "Specificity",
                     "HD95_pixels", "MCC", "Lesion_level_Sensitivity",
                     "Volumetric_Similarity", "ECE"]
    with open(out_csv, "w", newline="") as f:
        import csv as csv_module
        w = csv_module.writer(f)
        w.writerow(["test_set", "class"] + metric_names)
        for test_name, result in [("Refined_IDRiD", ridrid_result), ("DDR", ddr_result)]:
            for cls in CLASSES:
                m = result[1][cls]
                w.writerow([test_name, cls] + [m[mn] for mn in metric_names])
    print(f"\nFull 10-metric x 4-class results written -> {out_csv}")


if __name__ == "__main__":
    main()
