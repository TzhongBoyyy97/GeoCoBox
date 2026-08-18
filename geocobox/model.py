from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        for index in range(2):
            layers.extend(
                [
                    nn.Conv3d(
                        in_channels if index == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.InstanceNorm3d(out_channels, affine=True),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UNet3DBackbone(nn.Module):
    """Four-level 3D U-Net returning full-resolution decoder features."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, dropout: float = 0.0):
        super().__init__()
        channels = [base_channels * (2**i) for i in range(5)]
        self.encoders = nn.ModuleList(
            [
                ConvBlock(in_channels, channels[0], dropout),
                ConvBlock(channels[0], channels[1], dropout),
                ConvBlock(channels[1], channels[2], dropout),
                ConvBlock(channels[2], channels[3], dropout),
            ]
        )
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock(channels[3], channels[4], dropout)
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose3d(channels[4], channels[3], 2, 2),
                nn.ConvTranspose3d(channels[3], channels[2], 2, 2),
                nn.ConvTranspose3d(channels[2], channels[1], 2, 2),
                nn.ConvTranspose3d(channels[1], channels[0], 2, 2),
            ]
        )
        self.decoders = nn.ModuleList(
            [
                ConvBlock(channels[3] * 2, channels[3], dropout),
                ConvBlock(channels[2] * 2, channels[2], dropout),
                ConvBlock(channels[1] * 2, channels[1], dropout),
                ConvBlock(channels[0] * 2, channels[0], dropout),
            ]
        )
        self.out_channels = channels[0]

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))
        return x


class ContrastiveHead(nn.Module):
    """The two-layer point-wise convolution head described in the paper."""

    def __init__(self, in_channels: int, embedding_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, embedding_dim, kernel_size=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return F.normalize(self.layers(features), dim=1, eps=1e-6)


def _binary_boundary(mask: Tensor) -> Tensor:
    mask = mask.float()
    dilated = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool3d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded) > 0


def _dilate(mask: Tensor, radius: int) -> Tensor:
    result = mask.float()
    # Iterative 3x3x3 dilation avoids a very expensive (2r+1)^3 pooling kernel.
    for _ in range(max(0, radius)):
        result = F.max_pool3d(result, kernel_size=3, stride=1, padding=1)
    return result > 0


def _centers_from_box_mask(box_mask: Tensor) -> Tensor:
    centers = []
    spatial_shape = box_mask.shape[2:]
    fallback = [size // 2 for size in spatial_shape]
    for sample in box_mask[:, 0]:
        points = torch.nonzero(sample > 0.5, as_tuple=False)
        if points.numel() == 0:
            centers.append(torch.tensor(fallback, device=box_mask.device, dtype=torch.long))
        else:
            lo = points.amin(dim=0)
            hi = points.amax(dim=0)
            centers.append(torch.div(lo + hi, 2, rounding_mode="floor"))
    return torch.stack(centers)


class GeoCoBox(nn.Module):
    """3D U-Net with Anatomical-Driven CAM and Geometric Co-embedding."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        embedding_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = UNet3DBackbone(in_channels, base_channels, dropout)
        feature_channels = self.backbone.out_channels
        # FC_soft and FC_seed are implemented as voxel-wise FC (1x1x1 convolution).
        self.soft_layer = nn.Conv3d(feature_channels, 1, kernel_size=1)
        self.seed_layer = nn.Conv3d(feature_channels, 1, kernel_size=1)
        self.contrastive_head = ContrastiveHead(feature_channels, embedding_dim)

    def freeze_contrastive_head(self) -> None:
        for parameter in self.contrastive_head.parameters():
            parameter.requires_grad_(False)

    def unfreeze_contrastive_head(self) -> None:
        for parameter in self.contrastive_head.parameters():
            parameter.requires_grad_(True)

    def forward(
        self,
        image: Tensor,
        box_mask: Optional[Tensor] = None,
        radius_tau: int = 8,
        similarity_temperature: float = 0.1,
        refine: bool = True,
    ) -> Dict[str, Tensor]:
        features = self.backbone(image)
        soft_logits = self.soft_layer(features)
        # Equations (2)-(4): z is broadcast and multiplied with every feature channel.
        seed_logits = self.seed_layer(soft_logits * features)
        coarse_probability = torch.sigmoid(seed_logits)
        embeddings = self.contrastive_head(features)
        result = {
            "features": features,
            "embeddings": embeddings,
            "soft_logits": soft_logits,
            "seed_logits": seed_logits,
            "coarse_probability": coarse_probability,
        }
        if not refine:
            result["probability"] = coarse_probability
            return result

        if box_mask is None:
            box_mask = torch.ones_like(coarse_probability)
        centers = _centers_from_box_mask(box_mask)
        center_embeddings = torch.stack(
            [embeddings[b, :, z, y, x] for b, (z, y, x) in enumerate(centers)]
        )
        similarity = torch.einsum("bcdhw,bc->bdhw", embeddings, center_embeddings).unsqueeze(1)
        similarity_logits = similarity / max(float(similarity_temperature), 1e-6)
        similarity_probability = torch.sigmoid(similarity_logits)

        edge = _binary_boundary(coarse_probability.detach() >= 0.5)
        edge_band = _dilate(edge, radius_tau) & (box_mask > 0.5)
        refined_probability = torch.where(edge_band, similarity_probability, coarse_probability)
        refined_probability = refined_probability * (box_mask > 0.5).to(refined_probability.dtype)
        result.update(
            {
                "centers": centers,
                "similarity": similarity,
                "similarity_logits": similarity_logits,
                "edge_band": edge_band,
                "probability": refined_probability,
            }
        )
        return result

