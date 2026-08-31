"""FedSRA update packager.

FedSRA does not average client state dictionaries. The aggregator validates and
packages all one-shot client backbones so the dedicated trainer can construct
the RGA server model.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Dict, List

import torch

from ...core.aggregator import Aggregator
from ...core.types import ClientUpdate
from ...registry import aggregator


def _clone_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return type(value)((key, _clone_to_cpu(item)) for key, item in value.items())
    return copy.deepcopy(value)


@aggregator(
    name="fedsra",
    description="Package one-shot FedSRA client backbones without parameter averaging",
    version="1.0",
    author="FedSRA",
    weighted=False,
)
class FedSRAAggregator(Aggregator):
    def __init__(self, strict: bool = True):
        self.strict = bool(strict)

    def aggregate(
        self,
        updates: List[ClientUpdate],
        global_model: Any = None,
    ) -> Dict[str, Any]:
        del global_model
        if not updates:
            raise ValueError("FedSRA requires at least one successful client update")

        clients = []
        seen_ids = set()
        shared_context = None
        for update in updates:
            if update.client_id in seen_ids:
                raise ValueError(f"Duplicate FedSRA client id: {update.client_id}")
            seen_ids.add(update.client_id)
            if not isinstance(update.weights, Mapping):
                raise TypeError("FedSRA client weights must be a state_dict mapping")
            if update.num_samples <= 0:
                raise ValueError(f"Client {update.client_id} has no training samples")

            metadata = dict(update.metadata or {})
            class_counts = {
                int(label): int(count)
                for label, count in metadata.get("class_counts", {}).items()
                if int(count) > 0
            }
            if self.strict and sum(class_counts.values()) != int(update.num_samples):
                raise ValueError(
                    f"Client {update.client_id} class counts do not match num_samples"
                )

            context = (
                int(metadata.get("num_classes", -1)),
                int(metadata.get("feature_dim", -1)),
                int(metadata.get("etf_seed", -1)),
            )
            if shared_context is None:
                shared_context = context
            elif self.strict and context != shared_context:
                raise ValueError("FedSRA clients did not train against the same ETF context")

            clients.append(
                {
                    "client_id": update.client_id,
                    "state_dict": _clone_to_cpu(update.weights),
                    "num_samples": int(update.num_samples),
                    "class_counts": class_counts,
                    "metadata": metadata,
                }
            )

        return {
            "format": "oiafed.fedsra.bundle",
            "version": 1,
            "clients": clients,
            "shared_context": {
                "num_classes": shared_context[0],
                "feature_dim": shared_context[1],
                "etf_seed": shared_context[2],
            },
        }


__all__ = ["FedSRAAggregator"]
