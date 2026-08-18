from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
import torch.nn.functional as F


def soft_margin_mask_loss(logits: Tensor, box_mask: Tensor) -> Tensor:
    # SoftMarginLoss requires {-1,+1}; the paper writes B in {0,1} but uses its sign.
    targets = box_mask.to(logits.dtype).mul(2.0).sub(1.0)
    return F.soft_margin_loss(logits, targets)


def binary_dice_loss(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    probabilities = torch.sigmoid(logits)
    targets = targets.to(probabilities.dtype)
    reduce_dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def adcam_loss(seed_logits: Tensor, box_mask: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    soft_margin = soft_margin_mask_loss(seed_logits, box_mask)
    dice = binary_dice_loss(seed_logits, box_mask)
    return soft_margin + dice, {"soft_margin": soft_margin, "dice": dice}


def local_supervised_contrastive_loss(
    embeddings: Tensor,
    raw_image: Tensor,
    box_mask: Tensor,
    radius_tau: int,
    intensity_delta_hu: float,
    temperature: float = 0.1,
    max_positive: int = 128,
    max_negative: int = 512,
    denominator: Literal["negatives", "all"] = "negatives",
) -> Tensor:
    """Equation (8), using box-center local positives and filtered outside-box negatives."""
    losses = []
    for batch_index in range(embeddings.shape[0]):
        mask = box_mask[batch_index, 0] > 0.5
        points = torch.nonzero(mask, as_tuple=False)
        if points.numel() == 0:
            continue
        lo, hi = points.amin(dim=0), points.amax(dim=0)
        center = torch.div(lo + hi, 2, rounding_mode="floor")
        center_value = raw_image[batch_index, 0, center[0], center[1], center[2]]

        spatial_shape = mask.shape
        ranges = [
            torch.arange(
                max(0, int(center[axis]) - radius_tau),
                min(spatial_shape[axis], int(center[axis]) + radius_tau + 1),
                device=mask.device,
            )
            for axis in range(3)
        ]
        local_grid = torch.meshgrid(*ranges, indexing="ij")
        positive_coords = torch.stack([axis.reshape(-1) for axis in local_grid], dim=1)
        positive_index = tuple(positive_coords.t())
        squared_distance = ((positive_coords - center) ** 2).sum(dim=1)
        positive_difference = (
            raw_image[batch_index, 0][positive_index] - center_value
        ).abs()
        positive_selector = (
            mask[positive_index]
            & (squared_distance <= radius_tau**2)
            & (positive_difference <= intensity_delta_hu)
        )
        positive_coords = positive_coords[positive_selector]

        raw_sample = raw_image[batch_index, 0]
        negative_selector = (~mask) & ((raw_sample - center_value).abs() > intensity_delta_hu)
        negative_coords = torch.nonzero(negative_selector, as_tuple=False)
        if positive_coords.shape[0] < 2 or negative_coords.shape[0] == 0:
            continue

        positive_coords = _random_subset(positive_coords, max_positive)
        negative_coords = _random_subset(negative_coords, max_negative)
        positive = _gather_embeddings(embeddings[batch_index], positive_coords)
        negative = _gather_embeddings(embeddings[batch_index], negative_coords)
        positive_logits = positive @ positive.t() / temperature
        diagonal = torch.eye(positive.shape[0], dtype=torch.bool, device=positive.device)
        positive_logits = positive_logits.masked_fill(diagonal, float("-inf"))
        negative_logits = positive @ negative.t() / temperature
        log_numerator = torch.logsumexp(positive_logits, dim=1)
        if denominator == "negatives":
            log_denominator = torch.logsumexp(negative_logits, dim=1)
        elif denominator == "all":
            log_denominator = torch.logsumexp(
                torch.cat([positive_logits, negative_logits], dim=1), dim=1
            )
        else:
            raise ValueError(f"Unknown contrastive denominator: {denominator}")
        losses.append(-(log_numerator - log_denominator).mean())
    if not losses:
        # Preserve a valid autograd path for batches with no usable points.
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def geometric_coembedding_loss(
    similarity_logits: Tensor,
    coarse_probability: Tensor,
    edge_band: Tensor,
    max_edge_points: int = 4096,
) -> Tensor:
    """Center-edge same-class loss corresponding to equations (9)-(11)."""
    losses = []
    pseudo_same_class = (coarse_probability.detach() >= 0.5).to(similarity_logits.dtype)
    for batch_index in range(similarity_logits.shape[0]):
        selected = torch.nonzero(edge_band[batch_index, 0], as_tuple=False)
        if selected.numel() == 0:
            continue
        selected = _random_subset(selected, max_edge_points)
        index = tuple(selected.t())
        logits = similarity_logits[batch_index, 0][index]
        targets = pseudo_same_class[batch_index, 0][index]
        losses.append(F.binary_cross_entropy_with_logits(logits, targets))
    if not losses:
        return similarity_logits.sum() * 0.0
    return torch.stack(losses).mean()


def _random_subset(points: Tensor, maximum: int) -> Tensor:
    if maximum <= 0 or points.shape[0] <= maximum:
        return points
    order = torch.randperm(points.shape[0], device=points.device)[:maximum]
    return points[order]


def _gather_embeddings(embedding: Tensor, coords: Tensor) -> Tensor:
    values = embedding[:, coords[:, 0], coords[:, 1], coords[:, 2]].t()
    return F.normalize(values, dim=1, eps=1e-6)
