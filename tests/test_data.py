import json

import numpy as np

from geocobox.data import TumorPatchDataset, box_to_mask, extract_centered_patch


def test_box_mask_uses_half_open_coordinates():
    mask = box_to_mask((1, 2, 3, 4, 6, 8), (8, 8, 8))
    assert mask.sum() == 3 * 4 * 5


def test_centered_patch_and_dataset(tmp_path):
    image = np.arange(20**3, dtype=np.float32).reshape(20, 20, 20)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[8:12, 7:13, 9:14] = 1
    image_path = tmp_path / "image.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(image_path, image)
    np.save(mask_path, mask)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "case",
                        "image": "image.npy",
                        "mask": "mask.npy",
                        "box": [8, 7, 9, 12, 13, 14],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = TumorPatchDataset(manifest, patch_size=(16, 16, 16), ct_window=None)
    item = dataset[0]
    assert tuple(item["image"].shape) == (1, 16, 16, 16)
    assert item["has_mask"]
    assert item["mask"].sum() == 4 * 6 * 5
    assert item["box_mask"].sum() == 4 * 6 * 5


def test_border_patch_is_padded():
    array = np.ones((10, 10, 10), dtype=np.float32)
    patch, local_box, start = extract_centered_patch(array, (0, 0, 0, 2, 2, 2), (8, 8, 8), -1)
    assert patch.shape == (8, 8, 8)
    assert start[0] < 0
    assert patch[0, 0, 0] == -1
    assert all(0 <= value <= 8 for value in local_box)

