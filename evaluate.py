from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geocobox.config import load_config
from geocobox.data import TumorPatchDataset
from geocobox.metrics import hd95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved GeoCoBox patch predictions")
    parser.add_argument("--config", default="configs/geocobox.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    dataset = TumorPatchDataset(
        args.manifest,
        patch_size=config["data"]["patch_size"],
        ct_window=config["data"].get("ct_window"),
        zscore=config["data"].get("zscore", False),
    )
    prediction_dir = Path(args.predictions)
    rows = []
    for item in dataset:
        if not bool(item["has_mask"]):
            continue
        prediction_path = prediction_dir / f"{item['id']}_mask.npy"
        prediction = np.load(prediction_path) > 0
        target = item["mask"][0].numpy() > 0
        intersection = np.logical_and(prediction, target).sum()
        dice = (2.0 * intersection + 1e-6) / (prediction.sum() + target.sum() + 1e-6)
        rows.append(
            {
                "id": item["id"],
                "dice": float(dice),
                "hd95": hd95(prediction, target, spacing=args.spacing),
            }
        )
    finite_hd95 = [row["hd95"] for row in rows if np.isfinite(row["hd95"])]
    summary = {
        "samples": len(rows),
        "dice_percent": 100.0 * float(np.mean([row["dice"] for row in rows])) if rows else None,
        "hd95": float(np.mean(finite_hd95)) if finite_hd95 else None,
        "per_case": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
