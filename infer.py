from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from geocobox.checkpoint import load_checkpoint
from geocobox.config import load_config
from geocobox.data import TumorPatchDataset
from geocobox.model import GeoCoBox
from geocobox.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GeoCoBox inference on a manifest")
    parser.add_argument("--config", default="configs/geocobox.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-probability", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    device = resolve_device(config.get("device", "auto"))
    dataset = TumorPatchDataset(
        args.manifest,
        patch_size=config["data"]["patch_size"],
        ct_window=config["data"].get("ct_window"),
        zscore=config["data"].get("zscore", False),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = GeoCoBox(**config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.freeze_contrastive_head()
    model.eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contrastive = config["contrastive"]
    for batch in tqdm(loader, desc="inference"):
        image = batch["image"].to(device)
        box_mask = batch["box_mask"].to(device)
        output = model(
            image,
            box_mask=box_mask,
            radius_tau=contrastive["radius_tau"],
            similarity_temperature=contrastive["temperature"],
            refine=True,
        )
        probability = output["probability"][0, 0].cpu().numpy()
        sample_id = batch["id"][0]
        np.save(output_dir / f"{sample_id}_mask.npy", (probability >= args.threshold).astype(np.uint8))
        if args.save_probability:
            np.save(output_dir / f"{sample_id}_prob.npy", probability.astype(np.float32))


if __name__ == "__main__":
    main()

