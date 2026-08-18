from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor


def dice_score(prediction: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    prediction = prediction.float()
    target = target.float()
    reduce_dims = tuple(range(1, prediction.ndim))
    intersection = (prediction * target).sum(dim=reduce_dims)
    denominator = prediction.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


def hd95(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ImportError as exc:
        raise ImportError("HD95 requires `pip install scipy`.") from exc
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float("inf")
    pred_surface = prediction ^ binary_erosion(prediction)
    target_surface = target ^ binary_erosion(target)
    target_distance = distance_transform_edt(~target_surface, sampling=spacing)
    pred_distance = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([target_distance[pred_surface], pred_distance[target_surface]])
    return float(np.percentile(distances, 95))

