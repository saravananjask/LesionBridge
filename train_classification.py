"""
LesionBridge - Stage 3, step 2: lesion-conditioned DR/DME classification.

Trains a lightweight classifier head on top of FROZEN, pre-extracted
RETFound features (see extract_retfound_features.py). For CFP and UWF,
patch-token features are pooled using lesion-attention weights derived
from Stage 2's pseudo-masks (see lesion_attention.py) - the actual novel
mechanism connecting Stage 1's segmentation model to this classification
stage. OCT has no segmenter (different modality, see generate_pseudo_masks.py
docstring for why), so it uses plain mean-pooling instead.

As established in the experimental plan: MMRDR's official train/test split
(tr*/ts* filenames) is respected as-is. The pre-generated per-modality fold
CSVs (mmrdr_cfp_folds.csv etc) are used for 5-fold CV within the official
train portion only, for model selection. The official test portion is
touched exactly once, at the end, for the reported number.

CFP/UWF predict 5-class DR severity grade (0-4).
OCT predicts 3-class DME grade (0-2).

Metrics: accuracy, macro-F1, and quadratic weighted kappa (QWK) - QWK is
the standard metric for ordinal DR-grading tasks (used in the Kaggle DR
competition and most DR-grading papers), since it penalizes a
grade-0-predicted-as-grade-4 error far more than grade-0-as-grade-1.

Run:
    python train_classification.py --modality CFP --stage cv
    python train_classification.py --modality CFP --stage final
    python train_classification.py --modality CFP --stage test
    (repeat --modality UWF / OCT)
"""

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from lesion_attention import mask_to_patch_attention, attention_pool_patches, uniform_pool_patches

BASE = r"D:\Journal 2026\Aug 26\Retinopathy 30.8.26\Dataset"
FEATURE_DIR = os.path.join(BASE, "MMRDR_retfound_features")
PSEUDO_MASK_DIR = os.path.join(BASE, "MMRDR_pseudo_masks")
FOLDS_DIR = os.path.join(BASE, "folds")
CKPT_DIR = os.path.join(BASE, "checkpoints", "classification")

MMRDR_CSV = {
    "CFP": os.path.join(BASE, "MMRDR", "MMRDR-CFP", "FP.csv"),
    "UWF": os.path.join(BASE, "MMRDR", "MMRDR-UWF", "UWF.csv"),
    "OCT": os.path.join(BASE, "MMRDR", "MMRDR-OCT", "OCT.csv"),
}
NUM_CLASSES = {"CFP": 5, "UWF": 5, "OCT": 3}
HAS_MASK = {"CFP": True, "UWF": True, "OCT": False}
LESION_CLASSES = ["MA", "HE", "EX", "SE"]

BATCH_SIZE = 32   # frozen features are cheap - can afford a larger batch
NUM_EPOCHS = 50
LR = 1e-3
N_FOLDS = 5


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_csv_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


class MMRDRFeatureDataset(Dataset):
    """
    rows: list of dicts with at least "image" and "grade" keys (from either
          the fold CSVs or the raw MMRDR csv's test rows).
    modality: "CFP" | "UWF" | "OCT"
    """

    def __init__(self, rows, modality, ablation=False, baseline=False):
        self.rows = rows
        self.modality = modality
        self.feature_dir = os.path.join(FEATURE_DIR, modality)
        self.mask_dir = os.path.join(PSEUDO_MASK_DIR, modality)
        # ablation=True forces plain mean-pooling even for CFP/UWF (which
        # normally get lesion-attention pooling) - isolates the contribution
        # of the lesion-attention mechanism vs a no-conditioning baseline.
        self.has_mask = HAS_MASK[modality] and not ablation and not baseline
        # baseline=True: CLS-token-only, no patch tokens at all (the weakest,
        # simplest use of the frozen RETFound features - global embedding only,
        # no spatial/lesion information whatsoever). Distinct from ablation,
        # which still fuses CLS + mean-pooled patches.
        self.baseline = baseline

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        fname = row["image"].split("/")[-1]
        stem = os.path.splitext(fname)[0]
        grade = int(row["grade"])

        feat = np.load(os.path.join(self.feature_dir, f"{stem}.npz"))
        cls_emb = feat["cls"].astype(np.float32)          # (1024,)

        if self.baseline:
            return torch.from_numpy(cls_emb).float(), grade

        patch_tokens = feat["patches"].astype(np.float32)  # (196, 1024)

        if self.has_mask:
            mask_channels = []
            for c in LESION_CLASSES:
                import cv2
                m = cv2.imread(os.path.join(self.mask_dir, c, f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
                mask_channels.append(m)
            mask_4ch = np.stack(mask_channels, axis=0)  # (4, 384, 384) uint8
            attn = mask_to_patch_attention(mask_4ch)
            pooled = attention_pool_patches(patch_tokens, attn)
        else:
            pooled = uniform_pool_patches(patch_tokens)

        fused = np.concatenate([cls_emb, pooled])  # (2048,)
        return torch.from_numpy(fused).float(), grade


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=256, num_classes=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def compute_metrics(y_true, y_pred, num_classes):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)))
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic",
                             labels=list(range(num_classes)))
    return {"accuracy": acc, "macro_f1": macro_f1, "qwk": qwk}


def train_one_model(train_rows, val_rows, modality, device, epochs=NUM_EPOCHS, verbose=True, ablation=False, baseline=False):
    num_classes = NUM_CLASSES[modality]
    train_ds = MMRDRFeatureDataset(train_rows, modality, ablation=ablation, baseline=baseline)
    # num_workers/persistent_workers/pin_memory only affect data-loading wall-clock
    # time, not the training math - safe to raise without breaking ablation validity
    # (unlike batch_size/epochs, which must stay identical to the full-model runs
    # for a fair full-model-vs-baseline comparison).
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, persistent_workers=True, pin_memory=True)

    in_dim = 1024 if baseline else 2048
    model = ClassifierHead(in_dim=in_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    val_loader = None
    if val_rows is not None:
        val_ds = MMRDRFeatureDataset(val_rows, modality, ablation=ablation, baseline=baseline)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, persistent_workers=True, pin_memory=True)

    best_qwk = -1.0
    best_state = None
    best_metrics = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        if val_loader is not None:
            model.eval()
            all_preds, all_true = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    logits = model(x)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_true.extend(y.numpy().tolist())
            metrics = compute_metrics(all_true, all_preds, num_classes)
            if metrics["qwk"] > best_qwk:
                best_qwk = metrics["qwk"]
                best_metrics = dict(metrics)
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1}/{epochs} | train_loss {train_loss:.4f} | "
                      f"val_acc {metrics['accuracy']:.4f} val_macroF1 {metrics['macro_f1']:.4f} "
                      f"val_QWK {metrics['qwk']:.4f}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1}/{epochs} | train_loss {train_loss:.4f}")

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        return model, best_qwk, best_metrics
    return model, None, None


def run_cv(modality, device, ablation=False, baseline=False):
    fold_csv = os.path.join(FOLDS_DIR, f"mmrdr_{modality.lower()}_folds.csv")
    rows = load_csv_rows(fold_csv)
    tag = " [ABLATION: mean-pooling baseline, no lesion-attention]" if ablation else (" [BASELINE: CLS-token-only]" if baseline else "")
    print(f"\n=== {modality} 5-fold CV (train-portion only, official test untouched){tag} ===")

    fold_qwks = []
    fold_records = []
    for fold_idx in range(N_FOLDS):
        train_rows = [r for r in rows if int(r["fold"]) != fold_idx]
        val_rows = [r for r in rows if int(r["fold"]) == fold_idx]
        print(f"  Fold {fold_idx}: train={len(train_rows)} val={len(val_rows)}")
        _, best_qwk, best_metrics = train_one_model(train_rows, val_rows, modality, device, verbose=True, ablation=ablation, baseline=baseline)
        print(f"  Fold {fold_idx} best val QWK: {best_qwk:.4f} "
              f"(acc={best_metrics['accuracy']:.4f} macroF1={best_metrics['macro_f1']:.4f}, same epoch)")
        fold_qwks.append(best_qwk)
        fold_records.append({
            "modality": modality,
            "fold": fold_idx,
            "accuracy": best_metrics["accuracy"],
            "macro_f1": best_metrics["macro_f1"],
            "qwk": best_metrics["qwk"],
        })

    mean_qwk = np.mean(fold_qwks)
    std_qwk = np.std(fold_qwks)
    print(f"\n{modality} CV summary{tag}: QWK mean={mean_qwk:.4f} +/- {std_qwk:.4f}")

    suffix = "_ablation" if ablation else ("_baseline" if baseline else "")
    out_csv = os.path.join(BASE, f"results_stage3_{modality.lower()}_cv_per_fold{suffix}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["modality", "fold", "accuracy", "macro_f1", "qwk"])
        w.writeheader()
        w.writerows(fold_records)
        w.writerow({"modality": modality, "fold": "mean", "accuracy": "", "macro_f1": "", "qwk": mean_qwk})
        w.writerow({"modality": modality, "fold": "std", "accuracy": "", "macro_f1": "", "qwk": std_qwk})
    print(f"Per-fold CV results written -> {out_csv}")


def run_final(modality, device, ablation=False, baseline=False):
    fold_csv = os.path.join(FOLDS_DIR, f"mmrdr_{modality.lower()}_folds.csv")
    rows = load_csv_rows(fold_csv)  # all official-train rows, ignore fold column
    tag = " [ABLATION]" if ablation else (" [BASELINE]" if baseline else "")
    print(f"\n=== {modality} FINAL model{tag} (all {len(rows)} official-train rows, no holdout) ===")

    model, _, _ = train_one_model(rows, None, modality, device, verbose=True, ablation=ablation, baseline=baseline)

    os.makedirs(CKPT_DIR, exist_ok=True)
    suffix = "_ablation" if ablation else ("_baseline" if baseline else "")
    ckpt_path = os.path.join(CKPT_DIR, f"{modality.lower()}_classifier_final{suffix}.pt")
    torch.save({"model_state": model.state_dict(), "modality": modality, "ablation": ablation, "baseline": baseline}, ckpt_path)
    print(f"Final classifier saved -> {ckpt_path}")


def run_test(modality, device, ablation=False, baseline=False):
    suffix = "_ablation" if ablation else ("_baseline" if baseline else "")
    ckpt_path = os.path.join(CKPT_DIR, f"{modality.lower()}_classifier_final{suffix}.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    in_dim = 1024 if baseline else 2048
    model = ClassifierHead(in_dim=in_dim, num_classes=NUM_CLASSES[modality]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_rows = load_csv_rows(MMRDR_CSV[modality])
    test_rows = [r for r in all_rows if r["image"].split("/")[-1].startswith("ts")]
    tag = " [ABLATION]" if ablation else (" [BASELINE]" if baseline else "")
    print(f"\n=== {modality} official TEST set{tag} (n={len(test_rows)}, never used in training/CV) ===")

    test_ds = MMRDRFeatureDataset(test_rows, modality, ablation=ablation, baseline=baseline)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_true.extend(y.numpy().tolist())

    metrics = compute_metrics(all_true, all_preds, NUM_CLASSES[modality])
    print(f"  accuracy={metrics['accuracy']:.4f} macro_F1={metrics['macro_f1']:.4f} "
          f"QWK={metrics['qwk']:.4f}")

    out_csv = os.path.join(BASE, f"results_stage3_{modality.lower()}_test_metrics{suffix}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modality", "accuracy", "macro_f1", "qwk", "n_test"])
        w.writerow([modality, metrics["accuracy"], metrics["macro_f1"], metrics["qwk"], len(test_rows)])
    print(f"  written -> {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["CFP", "UWF", "OCT"], required=True)
    parser.add_argument("--stage", choices=["cv", "final", "test"], required=True)
    parser.add_argument("--ablation", action="store_true",
                         help="Force plain mean-pooling instead of lesion-attention pooling "
                              "(baseline for the ablation study). No effect on OCT, which "
                              "already uses mean-pooling by default (no masks exist for it).")
    parser.add_argument("--baseline", action="store_true",
                         help="CLS-token-only baseline (1024-dim): drop patch tokens entirely, "
                              "no pooling of any kind. The simplest possible use of the frozen "
                              "RETFound features - global embedding only, no spatial/lesion info. "
                              "Mutually exclusive with --ablation.")
    args = parser.parse_args()
    if args.ablation and args.baseline:
        parser.error("--ablation and --baseline are mutually exclusive - pick one.")

    device = get_device()
    print(f"Device: {device}")

    if args.stage == "cv":
        run_cv(args.modality, device, ablation=args.ablation, baseline=args.baseline)
    elif args.stage == "final":
        run_final(args.modality, device, ablation=args.ablation, baseline=args.baseline)
    elif args.stage == "test":
        run_test(args.modality, device, ablation=args.ablation, baseline=args.baseline)


if __name__ == "__main__":
    main()
