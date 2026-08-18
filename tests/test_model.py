import torch

from geocobox.losses import adcam_loss, geometric_coembedding_loss
from geocobox.model import GeoCoBox


def test_model_forward_and_losses():
    model = GeoCoBox(base_channels=2, embedding_dim=4)
    image = torch.randn(1, 1, 32, 32, 32)
    box = torch.zeros_like(image)
    box[:, :, 8:24, 7:25, 9:23] = 1
    output = model(image, box_mask=box, radius_tau=2, refine=True)
    assert output["probability"].shape == image.shape
    assert output["embeddings"].shape == (1, 4, 32, 32, 32)
    assert output["probability"][:, :, :5].max() == 0
    adcam, parts = adcam_loss(output["seed_logits"], box)
    gcl = geometric_coembedding_loss(
        output["similarity_logits"], output["coarse_probability"], output["edge_band"], 64
    )
    (adcam + gcl).backward()
    assert torch.isfinite(adcam)
    assert set(parts) == {"soft_margin", "dice"}

