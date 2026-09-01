"""
LesionBridge - Stage 2: pseudo-lesion-mask generation on MMRDR.

Runs the final Stage 1 segmentation model (trained on Refined_IDRiD + DDR,
with the corrected class-weighted loss - MA now functional, not zero) over
every CFP and UWF image in MMRDR to generate 4-class (MA/HE/EX/SE)
pseudo-lesion-masks. This is the actual novel mechanism in LesionBridge:
turning MMRDR's weak, image-level lesion tags into pixel-level structure by
transferring knowledge from the small, richly-annotated source datasets.

OCT is deliberately skipped: the segmenter was trained on surface color
fundus photos (CFP-style images). OCT is a completely different imaging
modality (cross-sectional retinal scans) with no shared lesion morphology,
so applying this segmenter to OCT would be meaningless, not just noisy.

Runs on BOTH the train (tr*) and test (ts*) portions of CFP/UWF, since the
downstream fusion/classification stage needs pseudo-masks available at
both training time and MMRDR-test-time evaluation.

Masks are saved at the segmenter's native 384x384 resolution (not
upscaled back to each image's original, variable size) as uint8 PNGs
(probability x 255), one subfolder per lesion class, mirroring DDR's own
folder convention:
    Dataset/MMRDR_pseudo_masks/CFP/MA/tr000001.png
    Dataset/MMRDR_pseudo_masks/CFP/HE/tr000001.png
    ... etc, and the same under UWF/

Resumable: skips any image whose 4 output mask files already exist, so an
interrupted run can just be restarted.

Run: python generate_pseudo_masks.py
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from train_segmentation import BASE, IMG_SIZE, CLASSES, get_device, build_model

CKPT_PATH = os.path.join(BASE, "checkpoints", "segmentation", "final_model.pt")
OUT_ROOT = os.path.join(BASE, "MMRDR_pseudo_masks")
BATCH_SIZE = 16  # inference-only, no gradients, so can go higher than training

MODALITIES = {
    "CFP": os.path.join(BASE, "MMRDR", "MMRDR-CFP", "img"),
    "UWF": os.path.join(BASE, "MMRDR", "MMRDR-UWF", "img"),
}
# OCT deliberately excluded - see module docstring


def imagenet_normalize(img_rgb_uint8):
    """Matches albumentations A.Normalize() defaults used in training."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img_rgb_uint8.astype(np.float32) / 255.0
    img = (img - mean) / std
    return img


class MMRDRImageDataset(Dataset):
    """
    Loads raw images for inference only (no masks/labels needed here).
    Skips any file whose 4 pseudo-mask outputs already exist, for resume.
    """

    def __init__(self, img_dir, out_dir):
        self.img_dir = img_dir
        self.out_dir = out_dir
        all_files = sorted(os.listdir(img_dir))
        self.files = [f for f in all_files if not self._already_done(f)]
        self.skipped = len(all_files) - len(self.files)

    def _already_done(self, filename):
        stem = os.path.splitext(filename)[0]
        return all(
            os.path.isfile(os.path.join(self.out_dir, cls, f"{stem}.png"))
            for cls in CLASSES
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_bgr = cv2.imread(os.path.join(self.img_dir, fname))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_norm = imagenet_normalize(img_resized)
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        return img_tensor, fname


def run_modality(model, device, modality_name, img_dir):
    out_dir = os.path.join(OUT_ROOT, modality_name)
    for cls in CLASSES:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    dataset = MMRDRImageDataset(img_dir, out_dir)
    print(f"\n=== {modality_name} ===")
    print(f"  total files: {len(dataset) + dataset.skipped} | "
          f"already done (skipped): {dataset.skipped} | remaining: {len(dataset)}")

    if len(dataset) == 0:
        print("  nothing to do.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    model.eval()
    processed = 0
    with torch.no_grad():
        for images, filenames in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 4, H, W)

            for b in range(probs.shape[0]):
                stem = os.path.splitext(filenames[b])[0]
                for i, cls in enumerate(CLASSES):
                    mask_uint8 = (probs[b, i] * 255).astype(np.uint8)
                    out_path = os.path.join(out_dir, cls, f"{stem}.png")
                    cv2.imwrite(out_path, mask_uint8)

            processed += images.size(0)
            if processed % 500 < BATCH_SIZE:
                print(f"  processed {processed}/{len(dataset)}")

    print(f"  done: {processed} images -> {out_dir}")


def main():
    device = get_device()
    print(f"Device: {device}")

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded final segmentation model from {CKPT_PATH}")

    for modality_name, img_dir in MODALITIES.items():
        run_modality(model, device, modality_name, img_dir)

    print("\nAll done. Pseudo-masks written under:", OUT_ROOT)
    print("OCT was skipped (different imaging modality, segmenter doesn't apply).")


if __name__ == "__main__":
    main()
