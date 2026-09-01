"""
LesionBridge - Stage 1 FINAL model training.

The 5-fold CV (mean dice 0.479 +/- 0.020) already validated the approach
and hyperparameters. This script trains one final model on ALL 437 pooled
training images (Refined_IDRiD train + DDR lesion_segmentation train),
with no held-out fold, using the same settings. This final model is what
gets evaluated on the official test sets afterwards (evaluate_test_sets.py)
and is what feeds Stage 2 (pseudo-label transfer onto MMRDR).

Run: python train_final_segmentation.py
"""

import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from train_segmentation import (
    BASE, FOLDS_CSV, IMG_SIZE, CLASSES, BATCH_SIZE, NUM_EPOCHS, LR,
    ENCODER_NAME, ENCODER_WEIGHTS, get_device, load_fold_rows,
    LesionSegDataset, get_transforms, build_model,
)
from weighted_loss import WeightedMultiLesionLoss

CKPT_DIR = os.path.join(BASE, "checkpoints", "segmentation")
FINAL_CKPT_PATH = os.path.join(CKPT_DIR, "final_model.pt")


def main():
    device = get_device()
    print(f"Training FINAL segmentation model on full training set | device: {device}")

    rows = load_fold_rows(FOLDS_CSV)  # ignore the 'fold' column entirely - use all 437
    print(f"total training images: {len(rows)}")

    train_ds = LesionSegDataset(rows, transform=get_transforms(train=True))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    loss_fn = WeightedMultiLesionLoss(classes=CLASSES, device=device)

    os.makedirs(CKPT_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)
        scheduler.step()
        print(f"  epoch {epoch+1}/{NUM_EPOCHS} | train_loss {train_loss:.4f}")

    torch.save({"model_state": model.state_dict(),
                "classes": CLASSES,
                "trained_on": "full_437_images_no_holdout"},
               FINAL_CKPT_PATH)
    print(f"\nFinal model saved -> {FINAL_CKPT_PATH}")
    print("Next: run evaluate_test_sets.py to get the one-shot test-set numbers.")


if __name__ == "__main__":
    main()
