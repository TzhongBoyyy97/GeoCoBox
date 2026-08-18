from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic GeoCoBox smoke-test dataset")
    parser.add_argument("--output-dir", default="demo")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    volume_dir = output_dir / "volumes"
    mask_dir = output_dir / "masks"
    volume_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    grid = np.indices((args.size,) * 3)
    rows = []
    for index in range(args.samples):
        center = rng.integers(args.size // 3, 2 * args.size // 3, size=3)
        radii = rng.integers(max(3, args.size // 10), max(4, args.size // 5), size=3)
        distance = sum(((grid[axis] - center[axis]) / radii[axis]) ** 2 for axis in range(3))
        mask = distance <= 1.0
        image = rng.normal(-700.0, 35.0, size=mask.shape).astype(np.float32)
        image[mask] = rng.normal(60.0, 18.0, size=int(mask.sum()))
        image_path = volume_dir / f"case_{index:03d}.npy"
        mask_path = mask_dir / f"case_{index:03d}.npy"
        np.save(image_path, image)
        np.save(mask_path, mask.astype(np.uint8))
        points = np.argwhere(mask)
        lo, hi = points.min(axis=0), points.max(axis=0) + 1
        rows.append(
            {
                "id": f"case_{index:03d}",
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
                "box": [int(value) for value in np.concatenate([lo, hi])],
            }
        )
    split = max(1, round(args.samples * 0.75))
    for name, subset in (("train", rows[:split]), ("val", rows[split:])):
        with (output_dir / f"{name}.json").open("w", encoding="utf-8") as stream:
            json.dump({"samples": subset}, stream, indent=2)
    config = {
        "seed": args.seed,
        "device": "auto",
        "output_dir": str((output_dir / "output").resolve()),
        "data": {
            "train_manifest": str((output_dir / "train.json").resolve()),
            "val_manifest": str((output_dir / "val.json").resolve()),
            "patch_size": [args.size] * 3,
            "ct_window": [-1000.0, 1000.0],
            "zscore": False,
            "num_workers": 0,
        },
        "model": {"in_channels": 1, "base_channels": 4, "embedding_dim": 8, "dropout": 0.0},
        "contrastive": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.01,
            "temperature": 0.1,
            "radius_tau": 3,
            "intensity_delta_hu": 40.0,
            "max_positive": 32,
            "max_negative": 64,
            "denominator": "negatives",
        },
        "training": {
            "epochs": 2,
            "adcam_only_epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "gcl_weight": 1.0,
            "max_edge_points": 256,
            "amp": False,
            "validate_every": 1,
            "threshold": 0.5,
        },
    }
    with (output_dir / "demo.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print(f"Synthetic data and config written to {output_dir}")
    print(f"Run: python train.py --config {output_dir / 'demo.yaml'} --stage all")


if __name__ == "__main__":
    main()

