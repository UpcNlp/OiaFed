"""OiaFed model registration for the validated FedEMoE EMoE network."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from ...registry import model
from ..fedemoe_reference.emoe import EMoE


@model(
    name="fedemoe",
    description="FedEMoE evidential mixture-of-experts model",
    version="1.0",
    author="FedEMoE Authors",
    model_type="mixture_of_experts",
    upstream="Stephen-Chow1/FedEMoE-CEGA",
)
class FedEMoEModel(EMoE):
    """Register the upstream EMoE unchanged while making initialization stable.

    OiaFed constructs one model per node.  The reference simulator constructs
    one template and deep-copies it.  Re-seeding each construction therefore
    gives every OiaFed node that exact template and records the post-template
    RNG state so the trainer can reproduce the reference DataLoader sequence.
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_experts: int = 8,
        backbone: str = "cnn",
        input_channels: int = 3,
        input_size: int = 32,
        expert_hidden_dim: int = 256,
        router_hidden_dim: int = 256,
        k_min: int = 1,
        k_max: int = 7,
        dropout: float = 0.1,
        seed: int = 42,
        **_: Any,
    ) -> None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        super().__init__(
            num_classes=num_classes,
            num_experts=num_experts,
            backbone=backbone,
            input_channels=input_channels,
            input_size=input_size,
            expert_hidden_dim=expert_hidden_dim,
            router_hidden_dim=router_hidden_dim,
            k_min=k_min,
            k_max=k_max,
            dropout=dropout,
        )

        self.fedemoe_seed = int(seed)
        self._fedemoe_post_init_rng_state = torch.random.get_rng_state().clone()


__all__ = ["FedEMoEModel"]
