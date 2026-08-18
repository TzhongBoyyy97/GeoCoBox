from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np

from geocobox.data import load_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create box manifests from pixel masks")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--masks", required=True, help="Mask directory with matching filenames")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--pattern", default="*.npy")
    parser.add_argument("--labels", type=int, nargs="+", help="Tumor label values; default: all >0")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--margin", type=int, default=0, help="Optional box expansion in voxels")
    return parser.parse_args()


def component_boxes(mask: np.ndarray, labels: Iterable[int] | None, margin: int = 0) -> list[list[int]]:
    try:
        from scipy.ndimage import label
    except ImportError as exc:
        raise ImportError("Connected components require `pip install scipy`.") from exc
    foreground = np.isin(mask, list(labels)) if labels else mask > 0
    components, count = label(foreground)
    boxes = []
    for component_id in range(1, count + 1):
        points = np.argwhere(components == component_id)
        if points.size == 0:
            continue
        lo = np.maximum(points.min(axis=0) - margin, 0)
        hi = np.minimum(points.max(axis=0) + 1 + margin, mask.shape)
        boxes.append([int(value) for value in np.concatenate([lo, hi])])
    return boxes


def write_manifest(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump({"samples": rows}, stream, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    image_dir = Path(args.images).resolve()
    mask_dir = Path(args.masks).resolve()
    rows = []
    for image_path in sorted(image_dir.glob(args.pattern)):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"No matching mask for {image_path.name}: {mask_path}")
        boxes = component_boxes(load_array(mask_path), args.labels, args.margin)
        if boxes:
            rows.append(
                {
                    "id": image_path.name.split(".")[0],
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "boxes": boxes,
                    "mask_labels": args.labels,
                }
            )
    random.Random(args.seed).shuffle(rows)
    split = round(len(rows) * args.train_ratio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(rows[:split], output_dir / "train.json")
    write_manifest(rows[split:], output_dir / "val.json")
    print(f"Wrote {split} training and {len(rows) - split} validation volumes to {output_dir}")


if __name__ == "__main__":
    main()
