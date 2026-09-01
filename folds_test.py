"""
Generate leakage-aware 5-fold cross-validation splits for LesionBridge.

Design (matches the experimental plan we agreed on):

1. Segmentation stage (Refined_IDRiD Train + DDR lesion_segmentation train):
   - Official Test partitions of both datasets are NEVER touched here -
     they are held out entirely for final reporting.
   - Only the official "train" portions are pooled and split into 5 folds
     for cross-validation / hyperparameter tuning.
   - DDR filenames were checked as a possible patient-grouping proxy
     (e.g. "007-0004-000.jpg") but the leading number turned out to be a
     batch/source code, not a patient ID: 382 of 383 DDR segmentation-train
     images share the single prefix "007". Grouping by it would collapse
     almost the entire dataset into one fold, so it is NOT used. DDR has no
     public patient identifier, so folds are plain random (image-level).
     Refined_IDRiD has one image per patient/eye, so no grouping is needed
     there either.
   - This lack of patient-level grouping for DDR must be disclosed as a
     limitation in the paper.

2. Fusion/classification stage (MMRDR):
   - MMRDR's own official train/test split (tr*/ts* filenames) is respected
     and never altered. Only the "train" (tr*) rows are split into 5 folds.
   - No patient ID is published in the MMRDR csvs (anonymized per the paper),
     so folds here are plain random K-fold, stratified by DR grade.
     NOTE: for the CFP modality specifically, the source OIA-DDR dataset
     lacks patient IDs even for the official train/test split, so some
     patient overlap between train and test may already exist upstream.
     This is a known, disclosed limitation of MMRDR itself, not something
     this script can fix.

3. External validation (UWF_fundus_dataset):
   - Not split at all. Used once, at the very end, as a frozen test set.

Run: python make_folds.py
Outputs CSV fold-assignment files next to this script.
"""

import os
import csv
import random
from collections import defaultdict

BASE = "/sessions/compassionate-fervent-dijkstra/mnt/Dataset"
OUT_DIR = os.path.join(BASE, "folds")
N_FOLDS = 5
SEED = 42

random.seed(SEED)


def group_kfold(items, groups, n_folds):
    """
    items: list of identifiers
    groups: list of group keys, same length as items (items sharing a group
            key always land in the same fold)
    returns: list of fold index (0..n_folds-1) per item
    """
    group_to_items = defaultdict(list)
    for idx, g in enumerate(groups):
        group_to_items[g].append(idx)

    unique_groups = list(group_to_items.keys())
    random.shuffle(unique_groups)

    fold_of_group = {}
    for i, g in enumerate(unique_groups):
        fold_of_group[g] = i % n_folds

    fold_assignment = [None] * len(items)
    for g, idxs in group_to_items.items():
        for idx in idxs:
            fold_assignment[idx] = fold_of_group[g]
    return fold_assignment


def stratified_kfold(items, labels, n_folds):
    """Simple stratified k-fold: distribute each class round-robin across folds."""
    label_to_idxs = defaultdict(list)
    for idx, lab in enumerate(labels):
        label_to_idxs[lab].append(idx)

    fold_assignment = [None] * len(items)
    for lab, idxs in label_to_idxs.items():
        random.shuffle(idxs)
        for i, idx in enumerate(idxs):
            fold_assignment[idx] = i % n_folds
    return fold_assignment


def make_segmentation_folds():
    print("=== Segmentation folds (Refined_IDRiD train + DDR lesion_segmentation train) ===")
    items = []   # (source, path_id, group_key)
    groups = []

    # Refined_IDRiD train images - one per patient/eye, unique group per image
    ridrid_img_dir = os.path.join(BASE, "Refined_IDRiD", "Train", "Images")
    for f in sorted(os.listdir(ridrid_img_dir)):
        items.append(("Refined_IDRiD", f, None))
        groups.append(f"ridrid_{f}")

    # DDR lesion_segmentation train images - no usable patient ID exists.
    # (Checked: the filename's leading number is a batch/source code, not a
    # patient ID - 382/383 images share the single value "007". Using it as
    # a grouping key would wrongly collapse almost the whole dataset into
    # one fold, so each image gets its own group, i.e. plain random folding.)
    ddr_img_dir = os.path.join(BASE, "DDR", "DDR-dataset", "lesion_segmentation", "train", "image")
    for f in sorted(os.listdir(ddr_img_dir)):
        items.append(("DDR", f, None))
        groups.append(f"ddr_{f}")

    fold_assignment = group_kfold(items, groups, N_FOLDS)

    out_path = os.path.join(OUT_DIR, "segmentation_folds.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "filename", "group_key", "fold"])
        for (source, fname, gkey), fold in zip(items, fold_assignment):
            w.writerow([source, fname, gkey, fold])

    counts = defaultdict(int)
    for fold in fold_assignment:
        counts[fold] += 1
    print(f"  total images: {len(items)}")
    print(f"  fold sizes: {dict(sorted(counts.items()))}")
    print(f"  written -> {out_path}")
    print("  NOTE: DDR grouped by site-prefix proxy (no public patient ID). "
          "Refined_IDRiD grouped 1 image = 1 group.")


def make_mmrdr_folds():
    print("\n=== MMRDR fusion/classification folds (train split only, per modality) ===")
    for modality, csv_name, img_subdir in [
        ("CFP", "FP.csv", "MMRDR-CFP"),
        ("UWF", "UWF.csv", "MMRDR-UWF"),
        ("OCT", "OCT.csv", "MMRDR-OCT"),
    ]:
        csv_path = os.path.join(BASE, "MMRDR", img_subdir, csv_name)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        train_rows = [r for r in rows if r["image"].split("/")[-1].startswith("tr")]
        test_rows = [r for r in rows if r["image"].split("/")[-1].startswith("ts")]

        labels = [r["grade"] for r in train_rows]
        fold_assignment = stratified_kfold(train_rows, labels, N_FOLDS)

        out_path = os.path.join(OUT_DIR, f"mmrdr_{modality.lower()}_folds.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "grade", "lesion", "lr", "fold"])
            for r, fold in zip(train_rows, fold_assignment):
                w.writerow([r["image"], r["grade"], r["lesion"], r["lr"], fold])

        counts = defaultdict(int)
        for fold in fold_assignment:
            counts[fold] += 1
        print(f"  [{modality}] train rows: {len(train_rows)} (folded, stratified by grade) "
              f"| held-out official test rows: {len(test_rows)} (untouched)")
        print(f"    fold sizes: {dict(sorted(counts.items()))}")
        print(f"    written -> {out_path}")

    print("  NOTE: no patient ID published in MMRDR csvs -> folds are stratified random, "
          "not patient-grouped. CFP train/test split itself may already contain patient "
          "overlap upstream (disclosed limitation of MMRDR, inherited here).")


def note_uwf_external():
    print("\n=== UWF_fundus_dataset ===")
    print("  Held out entirely. No folds generated. Use once, at the end, "
          "as a frozen external validation / generalization test.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    make_segmentation_folds()
    make_mmrdr_folds()
    note_uwf_external()
    print(f"\nAll fold files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
