"""
Verify integrity of the four DR datasets before training.

Checks:
 - MMRDR: every CSV row's image file exists; every image file has a CSV row;
          sample a subset of images to confirm they open correctly.
 - DDR:   DR_grading train/valid/test.txt counts match actual image files;
          lesion_segmentation image/label folders have matching filenames.
 - Refined_IDRiD: Train/Test Images vs Labels counts match, filenames pair up.
 - UWF_fundus_dataset: per-class folder counts, sample images open correctly.

Run: python verify_datasets.py
"""

import os
import csv
import random
from PIL import Image

BASE = "/sessions/compassionate-fervent-dijkstra/mnt/Dataset"
SAMPLE_CHECK_N = 200  # how many images to actually try opening, per dataset/subset


def check_image_opens(paths, n=SAMPLE_CHECK_N):
    sample = random.sample(paths, min(n, len(paths)))
    bad = []
    for p in sample:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            bad.append((p, str(e)))
    return bad, len(sample)


def verify_mmrdr():
    print("\n=== MMRDR ===")
    for modality, csv_name, img_subdir in [
        ("CFP", "FP.csv", "MMRDR-CFP"),
        ("UWF", "UWF.csv", "MMRDR-UWF"),
        ("OCT", "OCT.csv", "MMRDR-OCT"),
    ]:
        csv_path = os.path.join(BASE, "MMRDR", img_subdir, csv_name)
        img_dir = os.path.join(BASE, "MMRDR", img_subdir, "img")
        if not os.path.isfile(csv_path):
            print(f"  [{modality}] MISSING csv: {csv_path}")
            continue

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        csv_images = {row["image"].split("/")[-1] for row in rows}
        actual_images = set(os.listdir(img_dir)) if os.path.isdir(img_dir) else set()

        missing_files = csv_images - actual_images
        orphan_files = actual_images - csv_images

        print(f"  [{modality}] csv rows: {len(rows)} | files on disk: {len(actual_images)}")
        if missing_files:
            print(f"    WARNING: {len(missing_files)} images listed in csv but missing on disk "
                  f"(e.g. {list(missing_files)[:3]})")
        if orphan_files:
            print(f"    WARNING: {len(orphan_files)} image files on disk not listed in csv "
                  f"(e.g. {list(orphan_files)[:3]})")
        if not missing_files and not orphan_files:
            print("    OK: csv <-> files fully consistent")

        paths = [os.path.join(img_dir, f) for f in actual_images]
        bad, n_checked = check_image_opens(paths)
        print(f"    sample-opened {n_checked} images, {len(bad)} failed to open")
        for p, err in bad[:5]:
            print(f"      BAD: {p} -> {err}")


def verify_ddr():
    print("\n=== DDR ===")
    ddr_root = os.path.join(BASE, "DDR", "DDR-dataset")

    # DR grading
    grading_dir = os.path.join(ddr_root, "DR_grading")
    for split in ["train", "valid", "test"]:
        txt_path = os.path.join(grading_dir, f"{split}.txt")
        img_dir = os.path.join(grading_dir, split)
        if not os.path.isfile(txt_path):
            print(f"  [DR_grading/{split}] MISSING list file")
            continue
        with open(txt_path) as f:
            listed = [line.split()[0] for line in f if line.strip()]
        actual = set(os.listdir(img_dir)) if os.path.isdir(img_dir) else set()
        missing = set(listed) - actual
        print(f"  [DR_grading/{split}] listed: {len(listed)} | on disk: {len(actual)} "
              f"| missing: {len(missing)}")

    # lesion segmentation
    seg_dir = os.path.join(ddr_root, "lesion_segmentation")
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(seg_dir, split, "image")
        # DDR's original release names this folder inconsistently:
        # "label" for train/test, "segmentation label" for valid.
        label_dir = os.path.join(seg_dir, split, "label")
        if not os.path.isdir(label_dir):
            alt = os.path.join(seg_dir, split, "segmentation label")
            if os.path.isdir(alt):
                label_dir = alt
        if not os.path.isdir(img_dir):
            print(f"  [lesion_segmentation/{split}] MISSING image dir")
            continue
        images = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
        lesion_types = [d for d in os.listdir(label_dir)] if os.path.isdir(label_dir) else []
        print(f"  [lesion_segmentation/{split}] images: {len(images)} | lesion label types: {lesion_types}")
        for lt in lesion_types:
            lt_dir = os.path.join(label_dir, lt)
            masks = {os.path.splitext(f)[0] for f in os.listdir(lt_dir)}
            missing = images - masks
            if missing:
                print(f"    WARNING: [{lt}] missing masks for {len(missing)} images "
                      f"(e.g. {list(missing)[:3]})")
        # sample-open images
        paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        bad, n_checked = check_image_opens(paths, n=100)
        print(f"    sample-opened {n_checked} images, {len(bad)} failed")


def verify_refined_idrid():
    print("\n=== Refined_IDRiD ===")
    root = os.path.join(BASE, "Refined_IDRiD")
    for split in ["Train", "Test"]:
        img_dir = os.path.join(root, split, "Images")
        label_dir = os.path.join(root, split, "Labels")
        if not os.path.isdir(img_dir):
            print(f"  [{split}] MISSING image dir")
            continue
        images = os.listdir(img_dir)
        labels = os.listdir(label_dir) if os.path.isdir(label_dir) else []
        print(f"  [{split}] images: {len(images)} | labels: {len(labels)}")
        if len(images) != len(labels):
            print(f"    WARNING: count mismatch (images={len(images)}, labels={len(labels)})")
        paths = [os.path.join(img_dir, f) for f in images]
        bad, n_checked = check_image_opens(paths, n=len(paths))
        print(f"    opened all {n_checked} images, {len(bad)} failed")
        for p, err in bad:
            print(f"      BAD: {p} -> {err}")


def verify_uwf():
    print("\n=== UWF_fundus_dataset ===")
    root = os.path.join(
        BASE, "UWF_fundus_dataset",
        "Ultra-wide-field (SLO) fundus image dataset for intelligent diabetic retinopathy system",
        "diabetic retinopathy",
    )
    if not os.path.isdir(root):
        print(f"  MISSING root: {root}")
        return
    total = 0
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = os.listdir(cls_dir)
        total += len(files)
        paths = [os.path.join(cls_dir, f) for f in files]
        bad, n_checked = check_image_opens(paths, n=100)
        print(f"  [{cls}] {len(files)} files | sample-opened {n_checked}, {len(bad)} failed")
    print(f"  TOTAL images: {total}")


def main():
    verify_mmrdr()
    verify_ddr()
    verify_refined_idrid()
    verify_uwf()
    print("\nDone. Review any WARNING/BAD lines above before proceeding to training.")


if __name__ == "__main__":
    main()
