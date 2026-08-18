from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: Dict[str, Any],
    epoch: int,
    stage: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    best_metric: Optional[float] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "stage": stage,
        "best_metric": best_metric,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload

