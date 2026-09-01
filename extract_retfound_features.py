"""
LesionBridge - Stage 3, step 1: RETFound feature extraction (frozen).

RETFound (ViT-Large/16, 300M params, 224x224 input) is used as a FROZEN
feature extractor, not fine-tuned - a full fine-tune of a 300M-parameter
transformer isn't practical on an 8GB laptop GPU, and frozen-backbone
"linear/attention probing" is standard, defensible practice for large
foundation models.

Two RETFound checkpoints are used, matching each modality's imaging type:
  - iszt/RETFound_mae_meh        (CFP-pretrained) -> used for CFP and UWF
    (UWF is still a fundus-style surface photo, closest available match;
    there is no UWF-specific RETFound checkpoint publicly released)
  - iszt/RETFound_mae_natureOCT  (OCT-pretrained)  -> used for OCT only

For each image, this script saves BOTH the CLS token embedding (1024-dim,
global summary) AND the 14x14 grid of patch tokens (196 x 1024) to a
compressed .npz file. The patch-token grid is what lets the downstream
classifier do lesion-conditioned attention pooling using Stage 2's
pseudo-masks (for CFP/UWF) - the CLS token alone would discard spatial
information needed for that.

Requires: `hf auth login` already done (both RETFound repos are gated -
access must be manually accepted on huggingface.co first).

Run:
    python extract_retfound_features.py --modality CFP
    python extract_retfound_features.py --modality UWF
    python extract_retfound_features.py --modality OCT
    python extract_retfound_features.py --modality all
"""

import os
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor

BASE = r"D:\Journal 2026\Aug 26\Retinopathy 30.8.26\Dataset"
FEATURE_DIR = os.path.join(BASE, "MMRDR_retfound_features")

RETFOUND_REPOS = {
    "CFP": "iszt/RETFound_mae_meh",
    "UWF": "iszt/RETFound_mae_meh",   # closest available match, no UWF-specific checkpoint exists
    "OCT": "iszt/RETFound_mae_natureOCT",
}

IMG_DIRS = {
    "CFP": os.path.join(BASE, "MMRDR", "MMRDR-CFP", "img"),
    "UWF": os.path.join(BASE, "MMRDR", "MMRDR-UWF", "img"),
    "OCT": os.path.join(BASE, "MMRDR", "MMRDR-OCT", "img"),
}

BATCH_SIZE = 8  # ViT-L is heavy; keep conservative for 8GB VRAM


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RawImageDataset(Dataset):
    """Loads raw images for a modality, resumable (skips already-extracted files)."""

    def __init__(self, img_dir, out_dir, processor):
        self.img_dir = img_dir
        self.out_dir = out_dir
        self.processor = processor
        all_files = sorted(os.listdir(img_dir))
        self.files = [f for f in all_files if not self._already_done(f)]
        self.skipped = len(all_files) - len(self.files)

    def _already_done(self, filename):
        stem = os.path.splitext(filename)[0]
        return os.path.isfile(os.path.join(self.out_dir, f"{stem}.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_bgr = cv2.imread(os.path.join(self.img_dir, fname))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # processor handles resize-to-224 + normalization matching RETFound's training
        inputs = self.processor(images=img_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]  # (3, 224, 224)
        return pixel_values, fname


def extract_modality(modality, device):
    repo = RETFOUND_REPOS[modality]
    img_dir = IMG_DIRS[modality]
    out_dir = os.path.join(FEATURE_DIR, modality)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {modality} (RETFound repo: {repo}) ===")
    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False  # frozen - no fine-tuning

    dataset = RawImageDataset(img_dir, out_dir, processor)
    print(f"  total files: {len(dataset) + dataset.skipped} | "
          f"already done (skipped): {dataset.skipped} | remaining: {len(dataset)}")
    if len(dataset) == 0:
        print("  nothing to do.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    processed = 0
    with torch.no_grad():
        for pixel_values, filenames in loader:
            pixel_values = pixel_values.to(device)
            out = model(pixel_values=pixel_values)
            hidden = out.last_hidden_state  # (B, 197, 1024): [CLS, 196 patch tokens]
            cls_emb = hidden[:, 0, :].cpu().numpy()          # (B, 1024)
            patch_tokens = hidden[:, 1:, :].cpu().numpy()    # (B, 196, 1024)

            for b in range(cls_emb.shape[0]):
                stem = os.path.splitext(filenames[b])[0]
                out_path = os.path.join(out_dir, f"{stem}.npz")
                np.savez_compressed(
                    out_path,
                    cls=cls_emb[b].astype(np.float16),
                    patches=patch_tokens[b].astype(np.float16),
                )

            processed += pixel_values.size(0)
            if processed % 200 < BATCH_SIZE:
                print(f"  processed {processed}/{len(dataset)}")

    print(f"  done: {processed} images -> {out_dir}")

    # free VRAM before loading the next modality's model (if running --modality all)
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["CFP", "UWF", "OCT", "all"], default="all")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    modalities = ["CFP", "UWF", "OCT"] if args.modality == "all" else [args.modality]
    for m in modalities:
        extract_modality(m, device)

    print("\nAll requested feature extraction done. Cached under:", FEATURE_DIR)


if __name__ == "__main__":
    main()
