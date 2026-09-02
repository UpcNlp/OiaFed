"""Stateful Evidence-Guided Class-Aware Aggregation for FedEMoE."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ...core.aggregator import Aggregator
from ...core.types import ClientUpdate
from ...registry import aggregator
from ..fedemoe_reference.evidence_symbiosis import EvidenceGuidedSymbiosisPool


@aggregator(
    name="fedemoe",
    description="FedEMoE evidence-guided multi-parent symbiosis pool (EGCA)",
    version="1.0",
    author="FedEMoE Authors",
    weighted=False,
    upstream="Stephen-Chow1/FedEMoE-CEGA",
)
class FedEMoEAggregator(Aggregator):
    """Own the persistent server pool and run the validated EGCA operations."""

    def __init__(
        self,
        pool_size: int = 10,
        num_experts: int = 8,
        num_classes: int = 10,
        num_parents: int = 4,
        symbiosis_mode: str = "adaptive",
        adaptive_mode: str = "diversity",
        endo_ratio: float = 0.5,
        switch_round: int = 100,
        diversity_threshold: float = 0.5,
        smoothing_alpha: float = 0.0,
        ema_momentum: float = 1.0,
    ) -> None:
        self.pool_size = int(pool_size)
        self.num_experts = int(num_experts)
        self.num_classes = int(num_classes)
        self.num_parents = int(num_parents)
        self.symbiosis_mode = str(symbiosis_mode)
        self.adaptive_mode = str(adaptive_mode)
        self.endo_ratio = float(endo_ratio)
        self.switch_round = int(switch_round)
        self.diversity_threshold = float(diversity_threshold)
        self.smoothing_alpha = float(smoothing_alpha)
        self.ema_momentum = float(ema_momentum)
        self.pool: Optional[EvidenceGuidedSymbiosisPool] = None
        self.template_model: Optional[torch.nn.Module] = None
        self.last_round = 0

    def initialize(self, template_model: torch.nn.Module) -> None:
        """Create the reference pool once from the server template."""
        if self.pool is not None:
            return
        self.template_model = template_model
        self.pool = EvidenceGuidedSymbiosisPool(
            pool_size=self.pool_size,
            model_template=template_model,
            num_experts=self.num_experts,
            num_classes=self.num_classes,
            num_parents=self.num_parents,
            symbiosis_mode=self.symbiosis_mode,
            adaptive_mode=self.adaptive_mode,
            endo_ratio=self.endo_ratio,
            switch_round=self.switch_round,
            diversity_threshold=self.diversity_threshold,
            smoothing_alpha=self.smoothing_alpha,
            ema_momentum=self.ema_momentum,
        )

    def distribute(
        self,
        client_ids: Sequence[int],
    ) -> List[Tuple[int, int, Dict[str, torch.Tensor]]]:
        """Snapshot the one-to-one pool distribution before local training."""
        if self.pool is None:
            raise RuntimeError("FedEMoEAggregator.initialize must be called first")
        distributed = []
        for position, client_id in enumerate(client_ids):
            pool_index = position % self.pool_size
            state = {
                name: value.detach().clone()
                for name, value in self.pool.get_model(pool_index).state_dict().items()
            }
            distributed.append((int(client_id), pool_index, state))
        return distributed

    def aggregate(
        self,
        updates: List[ClientUpdate],
        global_model: Any = None,
    ) -> Dict[str, Any]:
        if not updates:
            raise ValueError("FedEMoE requires at least one client update")
        if self.pool is None:
            if global_model is None:
                raise ValueError("FedEMoE requires a server template model")
            self.initialize(global_model)
        assert self.pool is not None

        # The upstream server's distribution map keeps the final client for a
        # pool position if a cohort is larger than the pool.  Preserve that
        # behavior even though the paper configuration is exactly 10-to-10.
        latest_by_pool: Dict[int, ClientUpdate] = {}
        for update in updates:
            metadata = update.metadata or {}
            pool_index = int(metadata["pool_index"])
            if not 0 <= pool_index < self.pool_size:
                raise ValueError(f"Invalid FedEMoE pool index: {pool_index}")
            latest_by_pool[pool_index] = update

        self.pool.save_pool_snapshot()
        round_numbers = set()
        for pool_index, update in latest_by_pool.items():
            if not isinstance(update.weights, Mapping):
                raise TypeError("FedEMoE client weights must be a state_dict mapping")
            metadata = update.metadata or {}
            round_numbers.add(int(metadata.get("global_round", 0)))
            evidence_profile = metadata.get("evidence_profile")
            if not isinstance(evidence_profile, Mapping):
                raise TypeError("FedEMoE update is missing its evidence profile")

            updated_model = copy.deepcopy(self.pool.get_model(pool_index))
            updated_model.load_state_dict(update.weights, strict=True)
            updated_model.eval()
            self.pool.set_model(
                pool_index,
                updated_model,
                signature=dict(evidence_profile),
            )

        if len(round_numbers) != 1:
            raise ValueError("FedEMoE updates must come from one global round")
        self.last_round = round_numbers.pop()
        new_models = self.pool.perform_symbiosis(self.last_round)
        self.pool.update_pool(new_models)

        return {
            "format": "oiafed.fedemoe.pool",
            "version": 1,
            "round": self.last_round,
            "pool_size": self.pool_size,
            "pool_diversity": float(self.pool.get_pool_diversity()),
        }

    def aggregate_global_model(self) -> torch.nn.Module:
        """Return the upstream equal-weight pool model used for evaluation."""
        if self.pool is None:
            raise RuntimeError("FedEMoE pool has not been initialized")
        return self.pool.aggregate_to_global()

    def get_pool_diversity(self) -> float:
        if self.pool is None:
            return 0.0
        return float(self.pool.get_pool_diversity())


__all__ = ["FedEMoEAggregator"]
