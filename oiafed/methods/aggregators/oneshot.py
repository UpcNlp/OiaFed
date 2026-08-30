"""Update packagers for native one-shot baselines."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch

from ...core.aggregator import Aggregator
from ...core.types import ClientUpdate
from ...registry import aggregator


def _cpu_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return type(value)((key, _cpu_clone(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


@aggregator(
    name="oneshot_bundle",
    description="Package the complete one-shot client cohort without premature averaging",
    version="1.0",
    author="OiaFed",
    weighted=False,
)
class OneShotBundleAggregator(Aggregator):
    """Validate and retain client models/metadata for a method-specific trainer."""

    def __init__(self, strict: bool = True, **_: Any):
        self.strict = bool(strict)

    def aggregate(self, updates: list[ClientUpdate], global_model: Any = None) -> dict[str, Any]:
        del global_model
        if not updates:
            raise ValueError("a one-shot method requires at least one client update")
        seen: set[str] = set()
        clients = []
        for update in updates:
            if update.client_id in seen:
                raise ValueError(f"duplicate one-shot client id: {update.client_id}")
            seen.add(update.client_id)
            if update.num_samples <= 0:
                raise ValueError(f"client {update.client_id} has no samples")
            if self.strict and not isinstance(update.weights, Mapping):
                raise TypeError("one-shot client weights must be a state_dict mapping")
            clients.append(
                {
                    "client_id": update.client_id,
                    "state_dict": _cpu_clone(update.weights),
                    "num_samples": int(update.num_samples),
                    "metrics": copy.deepcopy(update.metrics),
                    "metadata": _cpu_clone(update.metadata or {}),
                }
            )
        return {"format": "oiafed.oneshot.bundle", "version": 1, "clients": clients}


__all__ = ["OneShotBundleAggregator"]
