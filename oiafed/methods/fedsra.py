"""Core math utilities for FedSRA.

The functions in this module intentionally mirror the reference implementation
in fedETF/ETF-pesuade: a seeded simplex ETF, the ERL losses, and RGA with
per-client feature-wise z-score normalization, sqrt(n) weighting, and post-L2
normalization.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, TensorDataset


def generate_simplex_etf(
    num_classes: int,
    feature_dim: int,
    seed: int = 42,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate the frozen simplex equiangular tight frame used by FedSRA."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if feature_dim < num_classes:
        raise ValueError(
            "feature_dim must be greater than or equal to num_classes for "
            "the reference simplex ETF construction"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    eye = torch.eye(num_classes, dtype=dtype)
    centered = eye - torch.ones(num_classes, num_classes, dtype=dtype) / num_classes
    etf = math.sqrt(num_classes / (num_classes - 1)) * centered

    if feature_dim > num_classes:
        projection, _ = torch.linalg.qr(
            torch.randn(feature_dim, num_classes, generator=generator, dtype=dtype),
            mode="reduced",
        )
        etf = etf @ projection.T

    if device is not None:
        etf = etf.to(device)
    return etf


def etf_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    etf: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """ERL relational term (``etf_cl`` in the reference implementation)."""
    features = F.normalize(features, dim=1)
    labels = labels.long()
    batch_size = features.size(0)

    prototype_loss = F.cross_entropy(features @ etf.T / temperature, labels)
    sample_loss = features.new_zeros(())

    if batch_size > 1:
        diagonal = torch.eye(batch_size, device=features.device, dtype=torch.bool)
        non_self = ~diagonal
        similarity = features @ features.T / temperature
        positive_mask = (
            (labels.unsqueeze(0) == labels.unsqueeze(1)).to(features.dtype)
            * non_self.to(features.dtype)
        )
        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0

        if valid.any():
            stabilized = similarity - similarity.max(dim=1, keepdim=True).values.detach()
            exp_similarity = torch.exp(stabilized) * non_self.to(features.dtype)
            log_prob = stabilized - torch.log(exp_similarity.sum(dim=1) + 1e-8).unsqueeze(1)
            sample_loss = -(
                (positive_mask * log_prob).sum(dim=1)[valid]
                / (positive_count[valid] + 1e-8)
            ).mean()

    return prototype_loss + 0.5 * sample_loss


def etf_alignment_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    etf: torch.Tensor,
) -> torch.Tensor:
    """ERL intrinsic alignment term (``etf_al`` in the reference code)."""
    features = F.normalize(features, dim=1)
    return (1 - (features * etf[labels.long()]).sum(dim=1)).mean()


def joint_erl_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    etf: torch.Tensor,
    *,
    loss_type: str = "J",
    alignment_weight: float = 0.5,
    temperature: float = 0.1,
    num_local_classes: int | None = None,
) -> torch.Tensor:
    """Compute the R/I/J ERL variants used by the FedSRA experiments."""
    loss_type = loss_type.upper()
    if loss_type not in {"R", "I", "J"}:
        raise ValueError(f"loss_type must be R, I, or J; got {loss_type!r}")

    if num_local_classes is None:
        num_local_classes = int(labels.unique().numel())

    alignment = etf_alignment_loss(features, labels, etf)
    if loss_type == "I" or num_local_classes < 2:
        return alignment

    relational = etf_contrastive_loss(features, labels, etf, temperature)
    if loss_type == "R":
        return relational
    return relational + alignment_weight * alignment


def rga_aggregate(
    raw_features: torch.Tensor | Sequence[torch.Tensor],
    sample_counts: torch.Tensor | Sequence[int | float],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply exact RGA over a shared sample axis.

    Args:
        raw_features: ``[K, N, D]`` tensor or K tensors shaped ``[N, D]``.
            ``N`` must cover the complete evaluation/calibration set when
            reproducing the paper because the z-score statistics are computed
            over that axis.
        sample_counts: Number of local training samples for each client.
    """
    if isinstance(raw_features, torch.Tensor):
        stacked = raw_features
    else:
        if not raw_features:
            raise ValueError("raw_features must contain at least one client")
        stacked = torch.stack(list(raw_features), dim=0)

    if stacked.ndim != 3:
        raise ValueError(f"raw_features must have shape [K, N, D], got {stacked.shape}")

    counts = torch.as_tensor(sample_counts, dtype=stacked.dtype, device=stacked.device)
    if counts.ndim != 1 or counts.numel() != stacked.size(0):
        raise ValueError("sample_counts must contain one value per client")
    if (counts <= 0).any():
        raise ValueError("sample_counts must be positive")

    num_samples = stacked.size(1)
    means = stacked.mean(dim=1, keepdim=True)
    stds = stacked.std(dim=1, keepdim=True, unbiased=num_samples > 1)
    standardized = (stacked - means) / (stds + eps)

    weights = counts.sqrt().view(-1, 1, 1)
    aggregated = (standardized * weights).sum(dim=0) / weights.sum().clamp_min(1e-12)
    return F.normalize(aggregated, dim=1)


def extract_targets(dataset: object) -> torch.Tensor:
    """Extract integer targets from common Dataset/Subset wrapper layouts."""
    if isinstance(dataset, Subset):
        base_targets = extract_targets(dataset.dataset)
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return base_targets[indices]

    if isinstance(dataset, TensorDataset) and len(dataset.tensors) >= 2:
        return torch.as_tensor(dataset.tensors[1], dtype=torch.long).reshape(-1)

    for attribute in ("targets", "labels"):
        if hasattr(dataset, attribute):
            value = getattr(dataset, attribute)
            if value is not None:
                return torch.as_tensor(value, dtype=torch.long).reshape(-1)

    wrapped = getattr(dataset, "dataset", None)
    if wrapped is not None and wrapped is not dataset:
        return extract_targets(wrapped)

    labels: List[int] = []
    if not hasattr(dataset, "__iter__"):
        raise TypeError(f"Cannot extract labels from {type(dataset).__name__}")
    for item in dataset:  # type: ignore[operator]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise TypeError("Dataset samples must contain (input, label)")
        label = item[1]
        labels.append(int(label.item()) if torch.is_tensor(label) else int(label))
    return torch.tensor(labels, dtype=torch.long)


def extract_class_counts(dataset: object, num_classes: int) -> Dict[int, int]:
    """Return non-zero class counts for a local dataset partition."""
    targets = extract_targets(dataset)
    if targets.numel() == 0:
        return {}
    if targets.min().item() < 0 or targets.max().item() >= num_classes:
        raise ValueError("Dataset labels are outside the configured class range")
    counts = torch.bincount(targets, minlength=num_classes)
    return {index: int(value) for index, value in enumerate(counts.tolist()) if value > 0}


__all__ = [
    "generate_simplex_etf",
    "etf_contrastive_loss",
    "etf_alignment_loss",
    "joint_erl_loss",
    "rga_aggregate",
    "extract_targets",
    "extract_class_counts",
]
