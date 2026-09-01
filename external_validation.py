"""
LesionBridge - External validation on UWF_fundus_dataset (independent cohort).

This is the final generalization test in the experimental plan: everything
so far (Stage 1 segmentation, Stage 2 pseudo-masks, Stage 3 classifier) was
trained/tuned/tested entirely on MMRDR. UWF_fundus_dataset is a completely
separate, independently collected dataset (1,630 images, 809 patients) that
has NEVER been touched anywhere in this pipeline - not for CV, not for
hyperparameter choices, nothing. Running our trained UWF classifier on it,
zero-shot, tests whether LesionBridge generalizes across cohorts/devices,
not just across a random split of the same source data.

Label-space note (important, documented so it's defensible in the paper):
UWF_fundus_dataset ships 3-class labels (0=Normal, 1=NPDR, 2=PDR), while our
MMRDR-trained UWF classifier predicts the standard 5-class DR grade
(0=No DR, 1=Mild NPDR, 2=Moderate NPDR, 3=Severe NPDR, 4=PDR). We do NOT
retrain or fine-tune anything for this external set. Instead we collapse
the model's 5-class prediction down to the same 3-class clinical grouping
the external dataset uses (1/2/3 -> NPDR), which is a standard, clinically
motivated grouping (not a fitted/tuned mapping), and score against that.
Raw 5-class prediction distribution is also reported for transparency.

Pipeline (each phase resumable / skip-if-cached, in case this needs to be
re-run or interrupted):
  1. Run Stage 1's final segmentation model over every UWF_fundus_dataset
     image -> 4-class pseudo-lesion-masks (mirrors generate_pseudo_masks.py).
  2. Extract frozen RETFound features (same checkpoint used for MMRDR-UWF:
     iszt/RETFound_mae_meh - closest available match, no UWF-specific
     RETFound checkpoint exists) for every image (mirrors
     extract_retfound_features.py).
  3. Lesion-attention-pool the patch tokens using the Stage 1 pseudo-masks
     (identical mechanism to train_classification.py), load the trained
     uwf_classifier_final.pt head, predict, collapse to 3-class, score.

Run: python external_validation.py
"""

import os
import csv
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

from train_segmentation import BASE, IMG_SIZE, CLASSES, get_device, build_model
from lesion_attention import mask_to_patch_attention, attention_pool_patches
from train_classification import ClassifierHead, NUM_CLASSES, CKPT_DIR

EXT_ROOT = os.path.join(
    BASE, "UWF_fundus_dataset",
    "Ultra-wide-field (SLO) fundus image dataset for intelligent diabetic retinopathy system",
    "diabetic retinopathy",
)
CSV_FILES = ["train.csv", "val.csv", "test.csv"]  # combined: we use ALL of it, zero-shot only
# confirmed by inspection: label <-> folder correspondence is exact
# (Normal512=496 rows w/ label 0, NPDR512=634 rows w/ label 1, PDR512=500 rows w/ label 2)
FOLDER_BY_LABEL = {"0": "Normal512", "1": "NPDR512", "2": "PDR512"}

SEG_CKPT_PATH = os.path.join(BASE, "checkpoints", "segmentation", "final_model.pt")
CLS_CKPT_PATH = os.path.join(CKPT_DIR, "uwf_classifier_final.pt")

PSEUDO_MASK_DIR = os.path.join(BASE, "UWF_fundus_dataset_pseudo_masks")
FEATURE_DIR = os.path.join(BASE, "UWF_fundus_dataset_retfound_features")
RETFOUND_REPO = "iszt/RETFound_mae_meh"  # same checkpoint used for MMRDR-UWF

SEG_BATCH_SIZE = 16
FEAT_BATCH_SIZE = 8

# clinically standard collapse: mild/moderate/severe NPDR all map to "NPDR"
GRADE5_TO_DR3 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
DR3_NAMES = {0: "Normal", 1: "NPDR", 2: "PDR"}


def load_all_rows():
    """
    Reads train.csv+val.csv+test.csv (we use ALL of them - none of this
    dataset was used for training/tuning anywhere, so there's no internal
    split to respect here, only a single zero-shot evaluation pass).

    The raw CSVs have a small amount of real messiness (checked by hand):
    a few filenames appear more than once with conflicting labels, and at
    least one filename has a typo (single underscore in the CSV vs a double
    underscore in the actual file on disk). This affects ~4 of 1630 rows.
    We dedupe by filename (first occurrence wins) and skip any row whose
    image file doesn't actually exist on disk, logging the count so it's
    not a silent data loss.
    """
    seen_filenames = set()
    rows = []
    n_dupe = 0
    n_missing_file = 0
    for fn in CSV_FILES:
        path = os.path.join(EXT_ROOT, fn)
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                label = r["Label"].strip()
                if label not in FOLDER_BY_LABEL:
                    continue  # guards against any stray header/blank rows
                fname = r["Image Name"].strip()
                if fname in seen_filenames:
                    n_dupe += 1
                    continue
                seen_filenames.add(fname)
                folder = FOLDER_BY_LABEL[label]
                img_path = os.path.join(EXT_ROOT, folder, fname)
                if not os.path.isfile(img_path):
                    n_missing_file += 1
                    continue
                rows.append({"filename": fname, "label3": int(label), "img_path": img_path})

    if n_dupe or n_missing_file:
        print(f"  (external CSV cleanup: skipped {n_dupe} duplicate filename rows, "
              f"{n_missing_file} rows whose image file was not found on disk)")
    return rows


# ---------------------------------------------------------------------------
# Phase 1: pseudo-mask generation (Stage 1 model, resumable)
# ---------------------------------------------------------------------------

def imagenet_normalize(img_rgb_uint8):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img_rgb_uint8.astype(np.float32) / 255.0
    return (img - mean) / std


class SegInferenceDataset(Dataset):
    def __init__(self, rows, out_dir):
        self.out_dir = out_dir
        self.rows = [r for r in rows if not self._already_done(r["filename"])]
        self.skipped = len(rows) - len(self.rows)

    def _already_done(self, filename):
        stem = os.path.splitext(filename)[0]
        return all(
            os.path.isfile(os.path.join(self.out_dir, c, f"{stem}.png"))
            for c in CLASSES
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img_bgr = cv2.imread(r["img_path"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_norm = imagenet_normalize(img_resized)
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        return img_tensor, r["filename"]


def run_pseudo_mask_generation(rows, device):
    for c in CLASSES:
        os.makedirs(os.path.join(PSEUDO_MASK_DIR, c), exist_ok=True)

    dataset = SegInferenceDataset(rows, PSEUDO_MASK_DIR)
    print(f"\n=== Phase 1: pseudo-masks (external UWF set) ===")
    print(f"  total: {len(dataset) + dataset.skipped} | "
          f"already done (skipped): {dataset.skipped} | remaining: {len(dataset)}")
    if len(dataset) == 0:
        print("  nothing to do.")
        return

    ckpt = torch.load(SEG_CKPT_PATH, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(dataset, batch_size=SEG_BATCH_SIZE, shuffle=False, num_workers=2)
    processed = 0
    with torch.no_grad():
        for images, filenames in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            for b in range(probs.shape[0]):
                stem = os.path.splitext(filenames[b])[0]
                for i, cls in enumerate(CLASSES):
                    mask_uint8 = (probs[b, i] * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(PSEUDO_MASK_DIR, cls, f"{stem}.png"), mask_uint8)
            processed += images.size(0)
            if processed % 200 < SEG_BATCH_SIZE:
                print(f"  processed {processed}/{len(dataset)}")

    del model
    torch.cuda.empty_cache()
    print(f"  done: {processed} images -> {PSEUDO_MASK_DIR}")


# ---------------------------------------------------------------------------
# Phase 2: RETFound feature extraction (resumable)
# ---------------------------------------------------------------------------

class RawImageDataset(Dataset):
    def __init__(self, rows, out_dir, processor):
        self.out_dir = out_dir
        self.processor = processor
        self.rows = [r for r in rows if not self._already_done(r["filename"])]
        self.skipped = len(rows) - len(self.rows)

    def _already_done(self, filename):
        stem = os.path.splitext(filename)[0]
        return os.path.isfile(os.path.join(self.out_dir, f"{stem}.npz"))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img_bgr = cv2.imread(r["img_path"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=img_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]
        return pixel_values, r["filename"]


def run_feature_extraction(rows, device):
    os.makedirs(FEATURE_DIR, exist_ok=True)
    print(f"\n=== Phase 2: RETFound feature extraction (external UWF set) ===")
    print(f"  repo: {RETFOUND_REPO}")

    processor = AutoImageProcessor.from_pretrained(RETFOUND_REPO)
    dataset = RawImageDataset(rows, FEATURE_DIR, processor)
    print(f"  total: {len(dataset) + dataset.skipped} | "
          f"already done (skipped): {dataset.skipped} | remaining: {len(dataset)}")

    if len(dataset) == 0:
        print("  nothing to do.")
        return

    model = AutoModel.from_pretrained(RETFOUND_REPO).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    loader = DataLoader(dataset, batch_size=FEAT_BATCH_SIZE, shuffle=False, num_workers=2)
    processed = 0
    with torch.no_grad():
        for pixel_values, filenames in loader:
            pixel_values = pixel_values.to(device)
            out = model(pixel_values=pixel_values)
            hidden = out.last_hidden_state
            cls_emb = hidden[:, 0, :].cpu().numpy()
            patch_tokens = hidden[:, 1:, :].cpu().numpy()
            for b in range(cls_emb.shape[0]):
                stem = os.path.splitext(filenames[b])[0]
                np.savez_compressed(
                    os.path.join(FEATURE_DIR, f"{stem}.npz"),
                    cls=cls_emb[b].astype(np.float16),
                    patches=patch_tokens[b].astype(np.float16),
                )
            processed += pixel_values.size(0)
            if processed % 200 < FEAT_BATCH_SIZE:
                print(f"  processed {processed}/{len(dataset)}")

    del model
    torch.cuda.empty_cache()
    print(f"  done: {processed} images -> {FEATURE_DIR}")


# ---------------------------------------------------------------------------
# Phase 3: lesion-attention pooling + frozen classifier + scoring
# ---------------------------------------------------------------------------

def run_evaluation(rows, device):
    print(f"\n=== Phase 3: zero-shot evaluation on external UWF set (n={len(rows)}) ===")

    model = ClassifierHead(num_classes=NUM_CLASSES["UWF"]).to(device)
    ckpt = torch.load(CLS_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    true3, pred5_list, pred3_list, missing = [], [], [], []

    with torch.no_grad():
        for r in rows:
            stem = os.path.splitext(r["filename"])[0]
            feat_path = os.path.join(FEATURE_DIR, f"{stem}.npz")
            mask_paths = [os.path.join(PSEUDO_MASK_DIR, c, f"{stem}.png") for c in CLASSES]

            if not os.path.isfile(feat_path) or not all(os.path.isfile(p) for p in mask_paths):
                missing.append(r["filename"])
                continue

            feat = np.load(feat_path)
            cls_emb = feat["cls"].astype(np.float32)
            patch_tokens = feat["patches"].astype(np.float32)

            mask_channels = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_paths]
            mask_4ch = np.stack(mask_channels, axis=0)
            attn = mask_to_patch_attention(mask_4ch)
            pooled = attention_pool_patches(patch_tokens, attn)

            fused = np.concatenate([cls_emb, pooled])
            x = torch.from_numpy(fused).float().unsqueeze(0).to(device)
            logits = model(x)
            pred5 = int(logits.argmax(dim=1).cpu().item())
            pred3 = GRADE5_TO_DR3[pred5]

            true3.append(r["label3"])
            pred5_list.append(pred5)
            pred3_list.append(pred3)

    if missing:
        print(f"  WARNING: {len(missing)} images missing cached features/masks, skipped "
              f"(re-run this script to fill them in before trusting the numbers).")

    n = len(true3)
    acc = accuracy_score(true3, pred3_list)
    macro_f1 = f1_score(true3, pred3_list, average="macro", labels=[0, 1, 2])
    qwk = cohen_kappa_score(true3, pred3_list, weights="quadratic", labels=[0, 1, 2])
    cm = confusion_matrix(true3, pred3_list, labels=[0, 1, 2])

    print(f"\n  n={n} | accuracy={acc:.4f} macro_F1={macro_f1:.4f} QWK={qwk:.4f}")
    print("  confusion matrix (rows=true, cols=pred), order [Normal, NPDR, PDR]:")
    print(cm)

    from collections import Counter
    print(f"  raw 5-class prediction distribution: {dict(sorted(Counter(pred5_list).items()))}")

    out_csv = os.path.join(BASE, "results_external_validation_uwf_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "accuracy", "macro_f1", "qwk"])
        w.writerow([n, acc, macro_f1, qwk])
    print(f"  written -> {out_csv}")

    preds_csv = os.path.join(BASE, "results_external_validation_uwf_predictions.csv")
    with open(preds_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_label3", "pred_label5_raw", "pred_label3_mapped"])
        rows_scored = [r for r in rows if r["filename"] not in set(missing)]
        for r, p5, p3 in zip(rows_scored, pred5_list, pred3_list):
            w.writerow([r["filename"], r["label3"], p5, p3])
    print(f"  per-image predictions written -> {preds_csv}")


def main():
    device = get_device()
    print(f"Device: {device}")

    rows = load_all_rows()
    print(f"Loaded {len(rows)} external UWF_fundus_dataset rows "
          f"(combined train+val+test - none of it was used for training).")

    run_pseudo_mask_generation(rows, device)
    run_feature_extraction(rows, device)
    run_evaluation(rows, device)

    print("\nExternal validation complete.")


if __name__ == "__main__":
    main()
