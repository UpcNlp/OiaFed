"""Models shared by the one-shot federation baselines.

The implementations in this module are native PyTorch modules so they can be
serialized by OiaFed's existing transport.  In particular, an ensemble keeps
the uploaded client models, FAFI keeps the client encoders and the averaged
learnable prototypes, and FuseFL materializes the progressively fused branch
graph instead of reducing it to parameter averaging.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry import model
from .cifar import _BasicBlock, _ResNet


@contextmanager
def _isolated_seed(seed: int | None):
    """Build replicated client models without advancing the process RNG."""
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


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
    """Logit ensemble used by the Ensemble and Co-Boosting baselines."""

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
    name="fafi_resnet18",
    description="FAFI learnable-prototype CIFAR ResNet-18",
    version="1.0",
    author="FAFI",
    model_type="prototype_classifier",
)
class FAFIResNet18(_ResNet):
    """ResNet-18 encoder plus the learnable prototype matrix used by FAFI."""

    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 512,
        initialization_seed: int | None = 0,
    ):
        if int(feature_dim) != 512:
            raise ValueError("FAFI ResNet-18 has a fixed 512-dimensional encoder")
        # The artifact creates one global model and deep-copies it to every
        # client.  OiaFed constructs nodes independently, so isolate and reuse
        # the same seed to preserve that shared feature coordinate system.
        with _isolated_seed(initialization_seed):
            super().__init__(_BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
            self.num_classes = int(num_classes)
            self.feature_dim = 512
            del self.fc
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            self.learnable_proto = nn.Parameter(torch.randn(self.num_classes, self.feature_dim))

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return raw encoder features (the server aggregates before L2 normalization)."""
        return super().forward_features(inputs)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = F.normalize(self.forward_features(inputs), dim=1, eps=1e-12)
        return features @ self.learnable_proto.T, features

    def get_proto(self) -> torch.Tensor:
        return self.learnable_proto


class FAFIServerModel(nn.Module):
    """Data-size weighted feature ensemble with an unweighted prototype mean."""

    def __init__(
        self,
        client_models: Sequence[nn.Module],
        sample_counts: Sequence[int | float],
        global_prototypes: torch.Tensor,
    ):
        super().__init__()
        if len(client_models) != len(sample_counts) or not client_models:
            raise ValueError("FAFI requires one sample count per client model")
        self.client_models = nn.ModuleList(client_models)
        counts = torch.as_tensor(sample_counts, dtype=torch.float32)
        if (counts <= 0).any():
            raise ValueError("FAFI client sample counts must be positive")
        self.register_buffer("feature_weights", counts / counts.sum())
        self.register_buffer("global_prototypes", global_prototypes.detach().clone().float())

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = [
            weight * _features(module, inputs)
            for weight, module in zip(self.feature_weights, self.client_models)
        ]
        return F.normalize(torch.stack(features).sum(dim=0), dim=1, eps=1e-12)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_features(inputs) @ self.global_prototypes.T


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


class DataFreeGenerator(nn.Module):
    """Generator architecture used by the Co-Boosting reference code."""

    def __init__(self, latent_dim: int = 256, width: int = 64, image_size: int = 32, channels: int = 3):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.init_size = int(image_size) // 4
        self.project = nn.Linear(self.latent_dim, width * 2 * self.init_size**2)
        self.blocks = nn.Sequential(
            nn.BatchNorm2d(width * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(width * 2, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width, channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        outputs = self.project(noise).view(noise.size(0), -1, self.init_size, self.init_size)
        return self.blocks(outputs)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d) and module.affine:
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.zeros_(module.bias)


class FuseFLLocalStage(nn.Module):
    """One trainable FuseFL segment, optionally preceded by a 1x1 adapter."""

    def __init__(self, block: nn.Module, output_channels: int):
        super().__init__()
        self.block = block
        self.output_channels = int(output_channels)
        self.adapter: nn.Module | None = None

    def install_adapter(self, input_channels: int, output_channels: int) -> None:
        if self.adapter is None:
            self.adapter = nn.Conv2d(int(input_channels), int(output_channels), 1)

    def forward(self, inputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, list):
            if self.adapter is None:
                raise RuntimeError("FuseFL branch input requires a configured adapter")
            inputs = self.adapter(torch.cat(inputs, dim=1))
        return self.block(inputs)


class FuseFLFusedStage(nn.Module):
    """Frozen horizontal concatenation of all client versions of one segment."""

    def __init__(self, branches: Sequence[FuseFLLocalStage]):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        return [branch(inputs) for branch in self.branches]


@model(
    name="fusefl_resnet18",
    description="Four-stage expandable ResNet-18 used by FuseFL",
    version="1.0",
    author="FuseFL",
    model_type="progressive_ensemble",
)
class FuseFLResNet18(nn.Module):
    """Four-stage ResNet-18 that can install progressively fused client branches."""

    def __init__(
        self,
        num_classes: int = 10,
        base_width: int = 20,
        initialization_seed: int | None = 0,
    ):
        super().__init__()
        with _isolated_seed(initialization_seed):
            width = int(base_width)
            self.num_classes = int(num_classes)
            self.base_width = width

            stem = nn.Sequential(
                nn.Conv2d(3, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
            stage1 = self._make_layer(width, width, 2, stride=1)
            stage1.add_module("layer2", self._make_layer(width, width * 2, 2, stride=2))
            stage2 = self._make_layer(width * 2, width * 4, 2, stride=2)
            stage3 = nn.Sequential(
                self._make_layer(width * 4, width * 8, 2, stride=2),
                nn.AdaptiveAvgPool2d(1),
            )
            channels = [width, width * 2, width * 4, width * 8]
            self.stages = nn.ModuleList(
                [FuseFLLocalStage(block, out) for block, out in zip([stem, stage1, stage2, stage3], channels)]
            )
            self.stage_channels = channels
            self.classifier = nn.Linear(width * 8, self.num_classes)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [_BasicBlock(in_channels, out_channels, stride)]
        layers.extend(_BasicBlock(out_channels, out_channels, 1) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def export_stage_state(self, stage_index: int) -> dict[str, torch.Tensor]:
        stage = self.stages[int(stage_index)]
        if not isinstance(stage, FuseFLLocalStage):
            raise RuntimeError(f"FuseFL stage {stage_index} is already fused")
        return {key: value.detach().cpu().clone() for key, value in stage.state_dict().items()}

    def install_fused_stage(
        self,
        stage_index: int,
        branch_states: Sequence[dict[str, torch.Tensor]],
    ) -> None:
        stage_index = int(stage_index)
        current = self.stages[stage_index]
        if isinstance(current, FuseFLFusedStage):
            return
        if not isinstance(current, FuseFLLocalStage) or not branch_states:
            raise ValueError("FuseFL requires local stage states from every client")
        branches = []
        for state in branch_states:
            branch = copy.deepcopy(current)
            branch.load_state_dict(state, strict=True)
            branches.append(branch)
        self.stages[stage_index] = FuseFLFusedStage(branches)
        if stage_index + 1 < len(self.stages):
            next_stage = self.stages[stage_index + 1]
            if not isinstance(next_stage, FuseFLLocalStage):
                raise RuntimeError("FuseFL stages must be installed in order")
            previous_channels = self.stage_channels[stage_index]
            next_stage.install_adapter(previous_channels * len(branches), previous_channels)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs: torch.Tensor | list[torch.Tensor] = inputs
        for stage in self.stages:
            outputs = stage(outputs)
        if isinstance(outputs, list):
            flattened = [output.flatten(1) for output in outputs]
            return torch.stack(flattened).sum(dim=0)
        return outputs.flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))


__all__ = [
    "OneShotEnsemble",
    "FAFIResNet18",
    "FAFIServerModel",
    "FedCGSResNet18",
    "FedCGSServerModel",
    "DataFreeGenerator",
    "FuseFLResNet18",
    "FuseFLLocalStage",
    "FuseFLFusedStage",
]
