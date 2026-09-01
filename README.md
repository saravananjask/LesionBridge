# LesionBridge

Lesion-conditioned multimodal diabetic retinopathy (DR) grading via pseudo-label transfer and attention pooling, across color fundus photography (CFP), ultra-widefield (UWF) imaging, and optical coherence tomography (OCT).

## Overview

LesionBridge is a three-stage pipeline that transfers explicit lesion structure from small, richly-annotated segmentation data into large-scale, weakly-labeled DR classification:

1. **Stage 1 — Lesion Segmentation.** A Swin-B + U-Net model trained on Refined_IDRiD and DDR localizes four lesion classes (microaneurysms, hemorrhages, hard exudates, soft exudates) at the pixel level.
2. **Stage 2 — Pseudo-Label Transfer.** The frozen Stage 1 model is applied to a large multimodal dataset (MMRDR) to generate lesion pseudo-masks for images that only carry image-level DR grade labels.
3. **Stage 3 — Lesion-Attention Pooling and Classification.** A frozen RETFound foundation-model classifier is conditioned on these pseudo-masks through a novel lesion-attention pooling mechanism, fusing global (CLS token) and lesion-weighted local (patch token) representations.

## Key Results

| Modality | Official Test QWK | 5-Fold CV QWK (mean ± std) |
|---|---|---|
| CFP | 0.8151 | 0.8386 ± 0.0069 |
| UWF | 0.7326 | 0.7298 ± 0.0054 |
| OCT | 0.8137 | 0.7678 ± 0.0129 |

Zero-shot external validation (independent UWF cohort, never used in training/CV/tuning): **QWK 0.6220**.

A controlled three-way ablation (CLS-only baseline → mean-pooling → lesion-attention pooling) shows lesion-attention pooling delivers a large, decisive gain for UWF (QWK +0.042) but only a small, mixed effect for CFP — reported honestly as a modality-dependent finding rather than a uniform improvement claim. Full details, all tables, and all figures are in `docs/LesionBridge_Manuscript_Draft.md`.

## Repository Structure

```
LesionBridge/
├── code/              # training, evaluation, and figure-generation scripts
├── results/           # all reported metrics as CSV (Tables 1-9)
├── figures/           # all manuscript figures
├── folds/             # cross-validation fold assignment CSVs
├── docs/              # manuscript draft, reference list, tables workbook
└── checkpoints/       # trained classification head weights (small, ~1-2MB each)
```

Segmentation checkpoints (~368MB each) and the raw datasets are **not included** in this repository — see Data below.

## Data

This project uses:

- **Refined_IDRiD** and **DDR** (lesion segmentation, Stage 1)
- **MMRDR** — a large multimodal retinal image dataset for DR detection, used for pseudo-label transfer and classifier training (Stages 2-3). See reference [29] in `docs/LesionBridge_References.md`.
- **UWF_fundus_dataset** — an independent cohort used exclusively for zero-shot external validation. See reference [30].

None of these datasets are redistributed in this repository. Obtain them from their original sources (cited above) and place them at the repository root following the paths referenced in `code/*.py`.

## Pipeline

```bash
# Stage 1: segmentation
python train_segmentation.py

# Stage 2: pseudo-label transfer
python generate_pseudo_masks.py

# Stage 3: classification (per modality)
python extract_retfound_features.py
python train_classification.py --modality CFP --stage cv
python train_classification.py --modality CFP --stage final
python train_classification.py --modality CFP --stage test
# repeat --modality UWF / OCT

# Ablation study
python train_classification.py --modality CFP --stage final --ablation
python train_classification.py --modality CFP --stage final --baseline
# (repeat --stage test for each, and for UWF)
```

## Citation

If you use this work, please cite the accompanying manuscript (see `docs/LesionBridge_Manuscript_Draft.md`).

## License

MIT License — see `LICENSE`.
