"""Models shared by validated one-shot federation baselines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry import model


def _logits(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


def _features(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if hasattr(module, "forward_features"):
        return module.forward_features(inputs)  # type: ignore[attr-defined]
    if hasattr(module, "feature"):
        return module.feature(inputs)  # type: ignore[attr-defined]
    output = module(inputs)
    if isinstance(output, (tuple, list)) and len(output) > 1:
        return output[1]
    raise TypeError(f"{type(module).__name__} does not expose features")


class OneShotEnsemble(nn.Module):
    """Uniform or explicitly weighted logit ensemble."""

    def __init__(
        self,
        models: Sequence[nn.Module],
        weights: Sequence[int | float] | torch.Tensor | None = None,
        *,
        trainable_weights: bool = False,
    ):
        super().__init__()
        if not models:
            raise ValueError("OneShotEnsemble requires at least one model")
        self.models = nn.ModuleList(models)
        initial = torch.ones(len(models), dtype=torch.float32)
        if weights is not None:
            initial = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
        if initial.numel() != len(models) or (initial < 0).any():
            raise ValueError("weights must be non-negative and match the models")
        if not bool(initial.sum() > 0):
            raise ValueError("at least one ensemble weight must be positive")
        if trainable_weights:
            self.mixture_weights = nn.Parameter(initial)
        else:
            self.register_buffer("mixture_weights", initial)

    def normalized_weights(self) -> torch.Tensor:
        weights = self.mixture_weights.clamp_min(0)
        return weights / weights.sum().clamp_min(1e-12)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.normalized_weights()
        outputs = [weight * _logits(module(inputs)) for weight, module in zip(weights, self.models)]
        return torch.stack(outputs).sum(dim=0)


@model(
    name="fedcgs_resnet18",
    description="ImageNet-pretrained ResNet-18 feature extractor used by FedCGS",
    version="1.0",
    author="FedCGS",
    model_type="feature_statistics",
)
class FedCGSResNet18(nn.Module):
    """Torchvision ImageNet ResNet-18 with the FedCGS feature interface."""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            backbone = resnet18(weights=weights)
        except Exception:
            if not pretrained:
                raise
            backbone = resnet18(weights=None)
        self.feature_dim = int(backbone.fc.in_features)
        self.num_classes = int(num_classes)
        backbone.fc = nn.Linear(self.feature_dim, self.num_classes)
        self.backbone = backbone

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        outputs = model.conv1(inputs)
        outputs = model.bn1(outputs)
        outputs = model.relu(outputs)
        outputs = model.maxpool(outputs)
        outputs = model.layer1(outputs)
        outputs = model.layer2(outputs)
        outputs = model.layer3(outputs)
        outputs = model.layer4(outputs)
        return torch.flatten(model.avgpool(outputs), 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone.fc(self.forward_features(inputs))


class FedCGSServerModel(nn.Module):
    """Shared-covariance Gaussian classifier constructed from sufficient statistics."""

    def __init__(self, backbone: nn.Module, weights: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("lda_weight", weights.detach().clone().float())
        self.register_buffer("lda_bias", bias.detach().clone().float())

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return _features(self.backbone, inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(self.forward_features(inputs), self.lda_weight, self.lda_bias)


__all__ = [
    "OneShotEnsemble",
    "FedCGSResNet18",
    "FedCGSServerModel",
]
