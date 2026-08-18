from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VolumeRecord:
    sample_id: str
    image: Path
    box: tuple[int, int, int, int, int, int]
    mask: Optional[Path] = None
    mask_labels: Optional[tuple[int, ...]] = None


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        array = np.load(path)
    elif suffixes.endswith(".npz"):
        archive = np.load(path)
        key = "image" if "image" in archive.files else archive.files[0]
        array = archive[key]
    elif suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("Reading NIfTI requires `pip install nibabel`.") from exc
        array = np.asarray(nib.load(str(path)).dataobj)
    else:
        raise ValueError(f"Unsupported volume format: {path}")
    array = np.asarray(array)
    if array.ndim == 4 and 1 in (array.shape[0], array.shape[-1]):
        array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {array.shape} from {path}")
    return array


def read_manifest(path: str | Path) -> list[VolumeRecord]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        rows = payload.get("samples", []) if isinstance(payload, dict) else payload
    records = []
    for row_index, row in enumerate(rows):
        image_path = _resolve_path(path.parent, row["image"])
        mask_path = _resolve_path(path.parent, row["mask"]) if row.get("mask") else None
        boxes = row.get("boxes")
        if boxes is None:
            boxes = [row["box"]]
        base_id = str(row.get("id", image_path.stem))
        mask_labels = (
            tuple(int(value) for value in row["mask_labels"])
            if row.get("mask_labels") is not None
            else None
        )
        for box_index, box in enumerate(boxes):
            if len(box) != 6:
                raise ValueError(f"Row {row_index}: a box must have six zyx coordinates")
            sample_id = base_id if len(boxes) == 1 else f"{base_id}_box{box_index:03d}"
            records.append(
                VolumeRecord(
                    sample_id=sample_id,
                    image=image_path,
                    box=tuple(int(value) for value in box),
                    mask=mask_path,
                    mask_labels=mask_labels,
                )
            )
    if not records:
        raise ValueError(f"No samples found in {path}")
    return records


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def box_to_mask(box: Sequence[int], shape: Sequence[int]) -> np.ndarray:
    z0, y0, x0, z1, y1, x1 = (int(value) for value in box)
    output = np.zeros(tuple(shape), dtype=np.float32)
    z0, y0, x0 = max(z0, 0), max(y0, 0), max(x0, 0)
    z1, y1, x1 = min(z1, shape[0]), min(y1, shape[1]), min(x1, shape[2])
    if z0 < z1 and y0 < y1 and x0 < x1:
        output[z0:z1, y0:y1, x0:x1] = 1.0
    return output


def extract_centered_patch(
    array: np.ndarray,
    box: Sequence[int],
    patch_size: Sequence[int],
    fill_value: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int, int, int, int, int], tuple[int, int, int]]:
    """Crop around a box center and return patch, local box, and source start."""
    patch_size = np.asarray(patch_size, dtype=np.int64)
    lo = np.asarray(box[:3], dtype=np.int64)
    hi = np.asarray(box[3:], dtype=np.int64)
    center = np.floor_divide(lo + hi - 1, 2)
    start = center - np.floor_divide(patch_size, 2)
    end = start + patch_size
    source_lo = np.maximum(start, 0)
    source_hi = np.minimum(end, np.asarray(array.shape))
    destination_lo = source_lo - start
    destination_hi = destination_lo + (source_hi - source_lo)

    patch = np.full(tuple(patch_size), fill_value, dtype=array.dtype)
    source_slices = tuple(slice(int(a), int(b)) for a, b in zip(source_lo, source_hi))
    destination_slices = tuple(slice(int(a), int(b)) for a, b in zip(destination_lo, destination_hi))
    patch[destination_slices] = array[source_slices]
    local_lo = lo - start
    local_hi = hi - start
    local_lo = np.maximum(local_lo, 0)
    local_hi = np.minimum(local_hi, patch_size)
    local_box = tuple(int(v) for v in np.concatenate([local_lo, local_hi]))
    return patch, local_box, tuple(int(v) for v in start)


class TumorPatchDataset(Dataset[Dict[str, Any]]):
    """One item per tumor box, following the paper's 96^3 patch protocol."""

    def __init__(
        self,
        manifest: str | Path,
        patch_size: Sequence[int] = (96, 96, 96),
        ct_window: Optional[Sequence[float]] = (-1000.0, 1000.0),
        zscore: bool = False,
    ):
        self.records = read_manifest(manifest)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.ct_window = tuple(float(value) for value in ct_window) if ct_window else None
        self.zscore = zscore

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = load_array(record.image).astype(np.float32, copy=False)
        fill = float(self.ct_window[0]) if self.ct_window else float(image.min())
        raw_patch, local_box, crop_start = extract_centered_patch(
            image, record.box, self.patch_size, fill_value=fill
        )
        if record.mask is None:
            mask_patch = np.zeros(self.patch_size, dtype=np.float32)
            has_mask = False
        else:
            mask = load_array(record.mask)
            if record.mask_labels is not None:
                mask = np.isin(mask, record.mask_labels)
            mask_patch, _, _ = extract_centered_patch(
                mask, record.box, self.patch_size, fill_value=0
            )
            mask_patch = (mask_patch > 0).astype(np.float32)
            has_mask = True

        box_mask = box_to_mask(local_box, self.patch_size)
        network_patch = self._normalize(raw_patch)
        return {
            "id": record.sample_id,
            "image": torch.from_numpy(network_patch[None]),
            "raw_image": torch.from_numpy(raw_patch.copy()[None]),
            "box_mask": torch.from_numpy(box_mask[None]),
            "mask": torch.from_numpy(mask_patch[None]),
            "has_mask": torch.tensor(has_mask, dtype=torch.bool),
            "box": torch.tensor(local_box, dtype=torch.long),
            "crop_start": torch.tensor(crop_start, dtype=torch.long),
            "original_shape": torch.tensor(image.shape, dtype=torch.long),
        }

    def _normalize(self, patch: np.ndarray) -> np.ndarray:
        output = patch.astype(np.float32, copy=True)
        if self.ct_window:
            low, high = self.ct_window
            output = np.clip(output, low, high)
            output = (output - low) / max(high - low, 1e-6)
            output = output * 2.0 - 1.0
        if self.zscore:
            output = (output - output.mean()) / max(float(output.std()), 1e-6)
        return np.ascontiguousarray(output, dtype=np.float32)
