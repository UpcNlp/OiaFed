"""FedSRA backbone and server-side RGA ensemble."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry import model
from ..fedsra import rga_aggregate


class _BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = F.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return F.relu(outputs + self.shortcut(inputs))


@model(
    name="fedsra_resnet18",
    description="CIFAR ResNet-18 feature backbone for FedSRA",
    version="1.0",
    author="FedSRA",
    model_type="feature_extractor",
)
class FedSRAResNet18Backbone(nn.Module):
    """CIFAR-adapted ResNet-18 matching the FedSRA reference implementation."""

    def __init__(self, feature_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, self.feature_dim)

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        blocks = [_BasicBlock(in_channels, out_channels, stride)]
        blocks.extend(_BasicBlock(out_channels, out_channels) for _ in range(1, num_blocks))
        return nn.Sequential(*blocks)

    def forward_raw(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the pre-L2 projection features consumed by RGA."""
        features = F.relu(self.bn1(self.conv1(inputs)))
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = self.pool(features).flatten(1)
        return self.fc(features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return normalized features used by local ERL training."""
        return F.normalize(self.forward_raw(inputs), dim=1)


class FedSRAEnsemble(nn.Module):
    """Server model retaining all one-shot client backbones and applying RGA.

    ``predict_loader`` reproduces the reference evaluation exactly by computing
    z-score statistics over the complete loader. ``forward`` uses calibration
    statistics when supplied; otherwise it uses the current batch and therefore
    should not be used for paper-number reproduction.
    """

    def __init__(
        self,
        backbones: Sequence[nn.Module],
        etf: torch.Tensor,
        sample_counts: Sequence[int | float],
        client_ids: Sequence[str] | None = None,
        class_counts: Sequence[dict[int, int]] | None = None,
    ):
        super().__init__()
        if not backbones:
            raise ValueError("FedSRAEnsemble requires at least one client backbone")
        if len(backbones) != len(sample_counts):
            raise ValueError("sample_counts must contain one entry per backbone")

        self.backbones = nn.ModuleList(backbones)
        self.register_buffer("etf", etf.detach().clone().float())
        self.register_buffer(
            "sample_counts",
            torch.as_tensor(sample_counts, dtype=torch.float32),
        )
        self.client_ids = list(client_ids or [f"client_{i}" for i in range(len(backbones))])
        self.class_counts = list(class_counts or [{} for _ in backbones])
        self.register_buffer("calibration_mean", torch.empty(0), persistent=False)
        self.register_buffer("calibration_std", torch.empty(0), persistent=False)

    def set_calibration_statistics(
        self,
        means: torch.Tensor,
        stds: torch.Tensor,
    ) -> None:
        expected = (len(self.backbones), self.etf.size(1))
        if tuple(means.shape) != expected or tuple(stds.shape) != expected:
            raise ValueError(f"Calibration statistics must have shape {expected}")
        self.calibration_mean = means.detach().clone()
        self.calibration_std = stds.detach().clone()

    def aggregate_raw_features(self, raw_features: torch.Tensor) -> torch.Tensor:
        if self.calibration_mean.numel():
            means = self.calibration_mean[:, None, :].to(raw_features)
            stds = self.calibration_std[:, None, :].to(raw_features)
            standardized = (raw_features - means) / (stds + 1e-8)
            weights = self.sample_counts.to(raw_features).sqrt().view(-1, 1, 1)
            aggregated = (standardized * weights).sum(0) / weights.sum().clamp_min(1e-12)
            return F.normalize(aggregated, dim=1)
        return rga_aggregate(raw_features, self.sample_counts.to(raw_features))

    @staticmethod
    def _forward_raw(backbone: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        forward_raw = getattr(backbone, "forward_raw", None)
        if callable(forward_raw):
            return forward_raw(inputs)
        return backbone(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raw = torch.stack(
            [self._forward_raw(backbone, inputs) for backbone in self.backbones],
            dim=0,
        )
        if not self.calibration_mean.numel() and raw.size(1) < 2:
            raise ValueError(
                "Single-sample RGA requires calibration statistics; use "
                "set_calibration_statistics() or predict_loader()"
            )
        features = self.aggregate_raw_features(raw)
        return features @ self.etf.to(features).T

    @torch.no_grad()
    def predict_loader(
        self,
        loader: Iterable,
        *,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact full-loader RGA logits and labels on CPU."""
        target_device = torch.device(device)
        all_client_features: List[torch.Tensor] = []
        labels: torch.Tensor | None = None

        for client_index, backbone in enumerate(self.backbones):
            try:
                original_device = next(backbone.parameters()).device
            except StopIteration:
                original_device = torch.device("cpu")
            backbone.to(target_device)
            backbone.eval()

            feature_parts: List[torch.Tensor] = []
            label_parts: List[torch.Tensor] = []
            for inputs, batch_labels in loader:
                inputs = inputs.to(target_device, non_blocking=True)
                feature_parts.append(self._forward_raw(backbone, inputs).float().cpu())
                if client_index == 0:
                    label_parts.append(torch.as_tensor(batch_labels).long().cpu())

            if not feature_parts:
                raise ValueError("Evaluation loader is empty")
            all_client_features.append(torch.cat(feature_parts, dim=0))
            if client_index == 0:
                labels = torch.cat(label_parts, dim=0)
            backbone.to(original_device)

        raw = torch.stack(all_client_features, dim=0)
        aggregated = rga_aggregate(raw, self.sample_counts.cpu())
        logits = aggregated @ self.etf.cpu().T
        assert labels is not None
        return logits, labels


__all__ = ["FedSRAResNet18Backbone", "FedSRAEnsemble"]
