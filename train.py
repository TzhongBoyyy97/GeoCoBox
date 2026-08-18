from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from geocobox.checkpoint import load_checkpoint, save_checkpoint
from geocobox.config import load_config
from geocobox.data import TumorPatchDataset
from geocobox.losses import (
    adcam_loss,
    geometric_coembedding_loss,
    local_supervised_contrastive_loss,
)
from geocobox.metrics import dice_score
from geocobox.model import GeoCoBox
from geocobox.utils import resolve_device, save_json, seed_everything, worker_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GeoCoBox reproduction")
    parser.add_argument("--config", default="configs/geocobox.yaml")
    parser.add_argument("--stage", choices=("pretrain", "train", "all"), default="all")
    parser.add_argument("--checkpoint", help="Initial/pretrained checkpoint")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config key (repeatable)",
    )
    return parser.parse_args()


def build_model(config: Dict[str, Any]) -> GeoCoBox:
    return GeoCoBox(**config["model"])


def build_loader(
    config: Dict[str, Any], manifest: str, batch_size: int, shuffle: bool
) -> DataLoader:
    data_config = config["data"]
    dataset = TumorPatchDataset(
        manifest=manifest,
        patch_size=data_config["patch_size"],
        ct_window=data_config.get("ct_window"),
        zscore=data_config.get("zscore", False),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_config.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_seed,
        persistent_workers=data_config.get("num_workers", 0) > 0,
    )


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def pretrain_contrastive_head(
    model: GeoCoBox,
    loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> Path:
    section = config["contrastive"]
    model.unfreeze_contrastive_head()
    parameters: Iterable[Tensor] = list(model.backbone.parameters()) + list(
        model.contrastive_head.parameters()
    )
    optimizer = torch.optim.SGD(
        parameters,
        lr=section["learning_rate"],
        momentum=config["training"].get("momentum", 0.9),
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    amp_enabled = device.type == "cuda" and config["training"].get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    checkpoint_path = output_dir / "contrastive_pretrained.pt"
    for epoch in range(section["epochs"]):
        model.train()
        total_loss = 0.0
        progress = tqdm(loader, desc=f"contrastive {epoch + 1}/{section['epochs']}")
        for batch in progress:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(batch["image"], refine=False)
                loss = local_supervised_contrastive_loss(
                    output["embeddings"],
                    batch["raw_image"],
                    batch["box_mask"],
                    radius_tau=section["radius_tau"],
                    intensity_delta_hu=section["intensity_delta_hu"],
                    temperature=section["temperature"],
                    max_positive=section["max_positive"],
                    max_negative=section["max_negative"],
                    denominator=section.get("denominator", "negatives"),
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        print(json.dumps({"stage": "contrastive", "epoch": epoch + 1, "loss": total_loss / len(loader)}))
        save_checkpoint(
            checkpoint_path,
            model,
            config,
            epoch=epoch + 1,
            stage="contrastive",
            optimizer=optimizer,
        )
    return checkpoint_path


def train_segmentation(
    model: GeoCoBox,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> Path:
    section = config["training"]
    contrastive = config["contrastive"]
    model.freeze_contrastive_head()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=section["learning_rate"],
        momentum=section.get("momentum", 0.9),
        weight_decay=section.get("weight_decay", 0.0),
    )
    amp_enabled = device.type == "cuda" and section.get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_dice = -1.0
    final_path = output_dir / "geocobox_final.pt"
    for epoch in range(section["epochs"]):
        model.train()
        use_gcl = epoch >= section["adcam_only_epochs"]
        running = {"loss": 0.0, "adcam": 0.0, "gcl": 0.0}
        progress = tqdm(train_loader, desc=f"train {epoch + 1}/{section['epochs']}")
        for batch in progress:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(
                    batch["image"],
                    box_mask=batch["box_mask"],
                    radius_tau=contrastive["radius_tau"],
                    similarity_temperature=contrastive["temperature"],
                    refine=use_gcl,
                )
                adcam, _ = adcam_loss(output["seed_logits"], batch["box_mask"])
                if use_gcl:
                    gcl = geometric_coembedding_loss(
                        output["similarity_logits"],
                        output["coarse_probability"],
                        output["edge_band"],
                        max_edge_points=section.get("max_edge_points", 4096),
                    )
                else:
                    gcl = adcam.new_zeros(())
                loss = adcam + section.get("gcl_weight", 1.0) * gcl
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running["loss"] += float(loss.detach())
            running["adcam"] += float(adcam.detach())
            running["gcl"] += float(gcl.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", gcl=use_gcl)

        log = {"stage": "segmentation", "epoch": epoch + 1, "gcl_enabled": use_gcl}
        log.update({key: value / len(train_loader) for key, value in running.items()})
        validate_now = val_loader is not None and (epoch + 1) % section.get("validate_every", 1) == 0
        if validate_now:
            val_dice = validate(model, val_loader, config, device)
            log["val_dice"] = val_dice
            if val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(
                    output_dir / "geocobox_best.pt",
                    model,
                    config,
                    epoch=epoch + 1,
                    stage="segmentation",
                    optimizer=optimizer,
                    best_metric=best_dice,
                )
        print(json.dumps(log))
        save_checkpoint(
            final_path,
            model,
            config,
            epoch=epoch + 1,
            stage="segmentation",
            optimizer=optimizer,
            best_metric=best_dice,
        )
    return final_path


@torch.no_grad()
def validate(
    model: GeoCoBox,
    loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
) -> float:
    model.eval()
    scores = []
    threshold = config["training"].get("threshold", 0.5)
    contrastive = config["contrastive"]
    for batch in loader:
        batch = to_device(batch, device)
        output = model(
            batch["image"],
            box_mask=batch["box_mask"],
            radius_tau=contrastive["radius_tau"],
            similarity_temperature=contrastive["temperature"],
            refine=True,
        )
        available = batch["has_mask"].bool()
        if available.any():
            prediction = output["probability"][available] >= threshold
            scores.append(float(dice_score(prediction, batch["mask"][available] > 0.5)))
    return float(sum(scores) / len(scores)) if scores else float("nan")


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    seed_everything(config.get("seed", 2026))
    device = resolve_device(config.get("device", "auto"))
    output_dir = Path(config.get("output_dir", "outputs/geocobox"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")

    model = build_model(config).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device)

    if args.stage in ("pretrain", "all"):
        pretrain_loader = build_loader(
            config,
            config["data"]["train_manifest"],
            config["contrastive"]["batch_size"],
            shuffle=True,
        )
        pretrain_contrastive_head(model, pretrain_loader, config, device, output_dir)

    if args.stage in ("train", "all"):
        if args.stage == "train" and not args.checkpoint:
            raise ValueError("--stage train requires --checkpoint with a pretrained contrastive head")
        train_loader = build_loader(
            config,
            config["data"]["train_manifest"],
            config["training"]["batch_size"],
            shuffle=True,
        )
        val_manifest = config["data"].get("val_manifest")
        val_loader = (
            build_loader(config, val_manifest, 1, shuffle=False) if val_manifest else None
        )
        train_segmentation(model, train_loader, val_loader, config, device, output_dir)


if __name__ == "__main__":
    main()

