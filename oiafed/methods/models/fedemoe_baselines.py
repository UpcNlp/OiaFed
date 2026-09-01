"""Models used by the FedEMoE paper's comparison-suite implementations."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from ...registry import model
from ..fedemoe_baselines_reference.baseline import BaselineModel
from ..fedemoe_baselines_reference.moe import MoE


def _seed_reference(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@model(
    name="fedemoe_baseline",
    description="CNN classifier from the FedEMoE comparison artifact",
    version="1.0",
    author="FedEMoE Authors",
    model_type="image_classifier",
    upstream="Stephen-Chow1/FedEMoE-CEGA",
)
class FedEMoEBaselineModel(BaselineModel):
    """Register the artifact's ``BaselineModel`` without changing its math."""

    def __init__(
        self,
        num_classes: int = 10,
        backbone: str = "cnn",
        input_channels: int = 3,
        input_size: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        seed: int = 42,
        **_: Any,
    ) -> None:
        _seed_reference(int(seed))
        super().__init__(
            num_classes=num_classes,
            backbone=backbone,
            input_channels=input_channels,
            input_size=input_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )


@model(
    name="fedemoe_moeda",
    description="Softmax MoE from the FedEMoE comparison artifact",
    version="1.0",
    author="FedEMoE Authors",
    model_type="mixture_of_experts",
    upstream="Stephen-Chow1/FedEMoE-CEGA",
)
class FedEMoEDAModel(MoE):
    """Register the exact FedMoE-DA client/server model architecture."""

    def __init__(
        self,
        num_classes: int = 10,
        num_experts: int = 8,
        backbone: str = "cnn",
        input_channels: int = 3,
        input_size: int = 32,
        expert_hidden_dim: int = 256,
        router_hidden_dim: int = 256,
        top_k: int = 2,
        dropout: float = 0.1,
        seed: int = 42,
        **_: Any,
    ) -> None:
        _seed_reference(int(seed))
        super().__init__(
            num_classes=num_classes,
            num_experts=num_experts,
            backbone=backbone,
            input_channels=input_channels,
            input_size=input_size,
            expert_hidden_dim=expert_hidden_dim,
            router_hidden_dim=router_hidden_dim,
            top_k=top_k,
            dropout=dropout,
        )


__all__ = ["FedEMoEBaselineModel", "FedEMoEDAModel"]
