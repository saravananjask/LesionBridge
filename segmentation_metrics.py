"""
10-metric evaluation suite for LesionBridge Stage 1 segmentation.

5 traditional (pixel-overlap) metrics:
    Dice, IoU (Jaccard), Precision, Recall/Sensitivity, Specificity

5 metrics chosen to surface what Dice hides, each with a stated reason:
    HD95            - boundary accuracy (Dice ignores edge precision)
    MCC             - honest under extreme class imbalance (tiny lesions like MA)
    Lesion-level Sens - "was the lesion found at all", not pixel overlap
    Volumetric Similarity - does predicted lesion burden match ground truth
    ECE             - is the model's confidence trustworthy

All functions operate on a single 2D binary/probability map at a time and
are aggregated (pooled or averaged, as appropriate per metric - see
comments) across a whole test set by the caller.
"""

import numpy as np
from scipy import ndimage


def confusion_counts(pred_bin, target_bin):
    # cast to plain Python int (unbounded) rather than numpy int32/int64 -
    # MCC multiplies four of these together, which overflows fixed-width
    # integer types even at moderate image resolutions (e.g. 384x384)
    tp = int(np.sum((pred_bin == 1) & (target_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (target_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (target_bin == 1)))
    tn = int(np.sum((pred_bin == 0) & (target_bin == 0)))
    return tp, fp, fn, tn


def dice_from_counts(tp, fp, fn, eps=1e-6):
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def iou_from_counts(tp, fp, fn, eps=1e-6):
    return (tp + eps) / (tp + fp + fn + eps)


def precision_from_counts(tp, fp, eps=1e-6):
    return (tp + eps) / (tp + fp + eps)


def recall_from_counts(tp, fn, eps=1e-6):
    return (tp + eps) / (tp + fn + eps)


def specificity_from_counts(tn, fp, eps=1e-6):
    return (tn + eps) / (tn + fp + eps)


def mcc_from_counts(tp, fp, fn, tn):
    # Python ints are arbitrary-precision, so this can't overflow even
    # though (tp*tn) and the four-term product get very large for
    # high-resolution images with a majority-background class.
    numerator = (tp * tn) - (fp * fn)
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq == 0:
        # any of the four marginal sums is zero -> MCC undefined by
        # convention this is reported as 0, not NaN
        return 0.0
    denominator = np.sqrt(float(denom_sq))
    return float(numerator) / denominator


def volumetric_similarity_from_counts(tp, fp, fn, eps=1e-6):
    # VS = 1 - |FN - FP| / (2*TP + FP + FN)
    return 1 - (abs(fn - fp) + eps) / (2 * tp + fp + fn + eps)


def hausdorff_95(pred_bin, target_bin):
    """
    95th-percentile symmetric Hausdorff distance, in pixels.
    Returns None if either mask is empty (undefined boundary distance) -
    caller should exclude None values when averaging, not treat as 0.
    """
    if pred_bin.sum() == 0 or target_bin.sum() == 0:
        return None

    def surface_points(mask):
        eroded = ndimage.binary_erosion(mask)
        boundary = mask & ~eroded
        return np.argwhere(boundary)

    pred_pts = surface_points(pred_bin.astype(bool))
    target_pts = surface_points(target_bin.astype(bool))
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return None

    # Distance transform of each boundary image gives, at every pixel,
    # distance to the nearest boundary pixel of that mask - sample it at
    # the other mask's boundary points for an efficient symmetric HD95.
    target_boundary_img = np.zeros_like(target_bin, dtype=bool)
    target_boundary_img[tuple(target_pts.T)] = True
    pred_boundary_img = np.zeros_like(pred_bin, dtype=bool)
    pred_boundary_img[tuple(pred_pts.T)] = True

    dt_target = ndimage.distance_transform_edt(~target_boundary_img)
    dt_pred = ndimage.distance_transform_edt(~pred_boundary_img)

    d_pred_to_target = dt_target[tuple(pred_pts.T)]
    d_target_to_pred = dt_pred[tuple(target_pts.T)]

    hd95 = max(
        np.percentile(d_pred_to_target, 95),
        np.percentile(d_target_to_pred, 95),
    )
    return float(hd95)


def lesion_level_sensitivity(pred_bin, target_bin, min_overlap_px=1):
    """
    Connected-component (instance) level sensitivity: fraction of true
    lesion instances that overlap a predicted lesion by at least
    min_overlap_px pixels. Returns (n_matched, n_total) - caller aggregates
    across the whole test set as sum(matched)/sum(total), NOT a mean of
    per-image ratios (avoids divide-by-zero on lesion-free images and
    weights every lesion instance equally).
    """
    target_labeled, n_target = ndimage.label(target_bin)
    if n_target == 0:
        return 0, 0
    matched = 0
    for lesion_id in range(1, n_target + 1):
        lesion_mask = target_labeled == lesion_id
        overlap = np.sum(lesion_mask & (pred_bin == 1))
        if overlap >= min_overlap_px:
            matched += 1
    return matched, n_target


def expected_calibration_error(probs, targets, n_bins=10):
    """
    Pixel-level ECE, pooled over as many images/pixels as the caller passes
    in (pass flattened, concatenated arrays across the whole test set for
    a stable estimate - a single image has too few pixels per bin).
    """
    probs = probs.flatten()
    targets = targets.flatten()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        avg_confidence = probs[in_bin].mean()
        avg_accuracy = targets[in_bin].mean()  # fraction of true positives in this bin
        ece += (count / n) * abs(avg_confidence - avg_accuracy)
    return float(ece)


def compute_all_metrics_for_class(all_preds_prob, all_targets, threshold=0.5):
    """
    all_preds_prob: list of 2D probability arrays (one per test image) for ONE class
    all_targets:    list of 2D binary ground-truth arrays, same length/order

    Returns a dict of the 10 metrics for this class, aggregated across the
    whole test set using the aggregation rule appropriate to each metric
    (pooled confusion counts for the pixel metrics/MCC/VS, per-image
    average excluding undefined cases for HD95, pooled instance counts for
    lesion-level sensitivity, pooled pixels for ECE).
    """
    total_tp = total_fp = total_fn = total_tn = 0
    hd95_values = []
    total_matched = total_lesions = 0
    all_probs_flat = []
    all_targets_flat = []

    for prob, target in zip(all_preds_prob, all_targets):
        pred_bin = (prob >= threshold).astype(np.uint8)
        target_bin = target.astype(np.uint8)

        tp, fp, fn, tn = confusion_counts(pred_bin, target_bin)
        total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn

        hd = hausdorff_95(pred_bin, target_bin)
        if hd is not None:
            hd95_values.append(hd)

        matched, n_lesions = lesion_level_sensitivity(pred_bin, target_bin)
        total_matched += matched
        total_lesions += n_lesions

        all_probs_flat.append(prob.flatten())
        all_targets_flat.append(target_bin.flatten())

    all_probs_flat = np.concatenate(all_probs_flat)
    all_targets_flat = np.concatenate(all_targets_flat)

    return {
        "Dice": dice_from_counts(total_tp, total_fp, total_fn),
        "IoU": iou_from_counts(total_tp, total_fp, total_fn),
        "Precision": precision_from_counts(total_tp, total_fp),
        "Recall_Sensitivity": recall_from_counts(total_tp, total_fn),
        "Specificity": specificity_from_counts(total_tn, total_fp),
        "HD95_pixels": float(np.mean(hd95_values)) if hd95_values else None,
        "MCC": mcc_from_counts(total_tp, total_fp, total_fn, total_tn),
        "Lesion_level_Sensitivity": (total_matched / total_lesions) if total_lesions > 0 else None,
        "Volumetric_Similarity": volumetric_similarity_from_counts(total_tp, total_fp, total_fn),
        "ECE": expected_calibration_error(all_probs_flat, all_targets_flat),
        "_n_images_with_lesion_for_HD95": len(hd95_values),
        "_n_lesion_instances_total": total_lesions,
    }
