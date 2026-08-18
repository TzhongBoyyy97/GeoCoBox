import torch

from geocobox.losses import local_supervised_contrastive_loss


def test_local_contrastive_loss_has_gradients():
    embeddings = torch.randn(1, 4, 12, 12, 12, requires_grad=True)
    raw = torch.full((1, 1, 12, 12, 12), -800.0)
    box = torch.zeros((1, 1, 12, 12, 12))
    box[:, :, 3:9, 3:9, 3:9] = 1
    raw[box.bool()] = 50.0
    loss = local_supervised_contrastive_loss(
        embeddings,
        raw,
        box,
        radius_tau=2,
        intensity_delta_hu=20,
        max_positive=16,
        max_negative=32,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert embeddings.grad is not None

