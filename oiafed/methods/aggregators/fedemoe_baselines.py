"""Server-state adapters for the FedEMoE comparison-suite baselines."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import torch

from ...core.aggregator import Aggregator
from ...core.types import ClientUpdate
from ...registry import aggregator
from ..fedemoe_baselines_reference.baselines import (
    FedAvgServer,
    FedEviServer,
    FedLESAMServer,
    FedMoEDAServer,
    FedNTDServer,
    FedProcServer,
    FedProtoServer,
    FedProxServer,
    FedSOLServer,
    FedSymServer,
    pFedHBServer,
)
from ..learners.fl.fedemoe_baselines import REFERENCE_METHODS


def _state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return module.state_dict()


class FedEMoEBaselineAggregator(Aggregator):
    """Keep the artifact server object intact and adapt OiaFed updates to it."""

    REFERENCE_METHOD = ""

    def __init__(
        self,
        num_classes: int = 10,
        backbone: str = "cnn",
        input_channels: int = 3,
        input_size: int = 32,
        hidden_dim: int = 256,
        device: str = "cuda",
        symbiosis_pool_size: int = 10,
        symbiosis_mode: str = "endo",
        pfedhb_prior_var: float = 1.0,
        num_experts: int = 8,
        expert_hidden_dim: int = 256,
        fedmoeda_top_k: int = 2,
        **_: Any,
    ) -> None:
        if self.REFERENCE_METHOD not in REFERENCE_METHODS:
            raise ValueError(f"Unknown FedEMoE baseline: {self.REFERENCE_METHOD}")
        self._requested_device = device
        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model_config = {
            "num_classes": int(num_classes),
            "backbone": str(backbone),
            "input_channels": int(input_channels),
            "input_size": int(input_size),
            "hidden_dim": int(hidden_dim),
        }
        self.symbiosis_pool_size = int(symbiosis_pool_size)
        self.symbiosis_mode = str(symbiosis_mode)
        self.pfedhb_prior_var = float(pfedhb_prior_var)
        self.moe_config = {
            "num_classes": int(num_classes),
            "num_experts": int(num_experts),
            "backbone": str(backbone),
            "input_channels": int(input_channels),
            "input_size": int(input_size),
            "expert_hidden_dim": int(expert_hidden_dim),
            "top_k": int(fedmoeda_top_k),
        }
        self.server: Any = None
        self._fedproto_models: Dict[int, tuple[torch.nn.Module, int]] = {}

    def initialize(self) -> None:
        """Construct the same server, in the same order, as BaselineTrainer."""
        method = self.REFERENCE_METHOD
        common = {
            "model_config": self.model_config,
            "device": self.device,
            "test_loader": None,
        }
        if method == "fedavg":
            self.server = FedAvgServer(**common)
        elif method == "fedprox":
            self.server = FedProxServer(**common)
        elif method == "fedsym":
            self.server = FedSymServer(
                **common,
                pool_size=self.symbiosis_pool_size,
                symbiosis_mode=self.symbiosis_mode,
            )
        elif method == "fedproto":
            self.server = FedProtoServer(**common, num_classes=self.model_config["num_classes"])
        elif method == "fedproc":
            self.server = FedProcServer(**common, num_classes=self.model_config["num_classes"])
        elif method == "fedntd":
            self.server = FedNTDServer(**common)
        elif method == "fedsol":
            self.server = FedSOLServer(**common)
        elif method == "fedlesam":
            self.server = FedLESAMServer(**common)
        elif method == "pfedhb":
            self.server = pFedHBServer(**common, prior_var=self.pfedhb_prior_var)
        elif method == "fedmoeda":
            self.server = FedMoEDAServer(**common, moe_config=self.moe_config)
        else:
            self.server = FedEviServer(**common)

    def set_test_loader(self, loader: Any) -> None:
        if self.server is None:
            raise RuntimeError("Aggregator must be initialized first")
        self.server.test_loader = loader

    def get_global_model(self) -> torch.nn.Module:
        if self.REFERENCE_METHOD == "fedmoeda":
            return self.server.global_moe
        return self.server.global_model

    def make_client_payload(
        self, client_id: int, selected_position: int, round_num: int
    ) -> Dict[str, Any]:
        method = self.REFERENCE_METHOD
        if method == "fedsym":
            model = self.server.model_pool[selected_position % self.server.pool_size]
        else:
            model = self.get_global_model()
        payload: Dict[str, Any] = {
            "round_num": int(round_num),
            "client_id": int(client_id),
            "model_weights": _state_dict(model),
        }
        if method == "fedproto":
            payload["global_prototypes"] = self.server.get_global_prototypes()
            payload["prototype_mask"] = self.server.get_prototype_mask()
        elif method == "fedproc":
            payload["global_prototypes"] = self.server.get_global_prototypes()
        elif method == "fedlesam":
            previous = self.server.get_prev_global_model()
            payload["prev_global_weights"] = (
                _state_dict(previous) if previous is not None else None
            )
        return payload

    def _model_from_update(self, update: ClientUpdate) -> torch.nn.Module:
        template = self.get_global_model()
        model = copy.deepcopy(template).to(self.device)
        model.load_state_dict(update.weights, strict=True)
        return model

    def aggregate_round(
        self, updates: List[ClientUpdate], round_num: int
    ) -> Dict[str, torch.Tensor]:
        if not updates:
            raise ValueError("No updates to aggregate")
        method = self.REFERENCE_METHOD
        models = [self._model_from_update(update) for update in updates]
        weights = [update.num_samples for update in updates]

        if method == "fedsym":
            client_models = {
                int(update.metadata["client_id"]): model
                for update, model in zip(updates, models)
            }
            self.server.receive_models(client_models)
            self.server.perform_symbiosis(round_num)
        elif method == "fedproto":
            for update, model in zip(updates, models):
                client_id = int(update.metadata["client_id"])
                self.server.update_client_prototypes(
                    client_id,
                    update.metadata["local_prototypes"],
                    update.metadata["prototype_mask"],
                    update.metadata["class_counts"],
                )
                self._fedproto_models[client_id] = (model, update.num_samples)
            self.server.aggregate_cached_prototypes()
            ordered = [self._fedproto_models[key] for key in sorted(self._fedproto_models)]
            self.server.aggregate(
                [item[0] for item in ordered], [item[1] for item in ordered]
            )
        elif method == "fedproc":
            self.server.aggregate(models, weights)
            self.server.aggregate_prototypes(
                [
                    (
                        update.metadata["local_prototypes"],
                        update.metadata["prototype_mask"],
                    )
                    for update in updates
                ],
                weights,
            )
        elif method == "pfedhb":
            self.server.aggregate(models, weights)
            self.server.aggregate_posteriors(
                [update.metadata["posterior"] for update in updates], weights
            )
        elif method == "fedmoeda":
            self.server.aggregate_domain_aware(
                models,
                weights,
                [update.metadata["router_weights"] for update in updates],
            )
        elif method == "fedevi":
            self.server.aggregate_with_uncertainty(
                models,
                weights,
                [update.metadata["uncertainty"] for update in updates],
            )
        else:
            self.server.aggregate(models, weights)

        return _state_dict(self.get_global_model())

    def aggregate(
        self,
        updates: List[ClientUpdate],
        global_model: Optional[Any] = None,
    ) -> Any:
        del global_model
        round_num = int(updates[0].metadata.get("round_num", 0)) if updates else 0
        return self.aggregate_round(updates, round_num)

    def evaluate(self) -> Dict[str, float]:
        if self.REFERENCE_METHOD == "fedsym":
            self.server.aggregate_global_model()
        return self.server.evaluate()


def _register_reference_aggregators() -> None:
    for method in REFERENCE_METHODS:
        class_name = "FedEMoE" + "".join(part.title() for part in method.split("_")) + "Aggregator"
        component_name = f"fedemoe_{method}"
        cls = type(
            class_name,
            (FedEMoEBaselineAggregator,),
            {"REFERENCE_METHOD": method, "__module__": __name__},
        )
        cls = aggregator(
            name=component_name,
            description=f"FedEMoE artifact {method} server adapter",
            version="1.0",
            author="FedEMoE Authors",
            weighted=True,
            upstream="Stephen-Chow1/FedEMoE-CEGA",
        )(cls)
        globals()[class_name] = cls


_register_reference_aggregators()

__all__ = ["FedEMoEBaselineAggregator"] + [
    "FedEMoE" + "".join(part.title() for part in method.split("_")) + "Aggregator"
    for method in REFERENCE_METHODS
]
