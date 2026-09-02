"""Native OiaFed orchestration for the FedEMoE comparison-suite baselines."""

from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...core.trainer import Trainer
from ...core.types import ClientUpdate, RoundMetrics, RoundResult, TrainResult
from ...registry import trainer
from ..learners.fl.fedemoe_baselines import REFERENCE_METHODS


def _proxy_client_number(proxy: Any) -> int:
    target_id = str(getattr(proxy, "_target_id", ""))
    try:
        return int(target_id.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Learner id must end in an integer: {target_id}") from exc


class FedEMoEBaselineTrainer(Trainer):
    """Map ``BaselineTrainer.train_round`` onto OiaFed learner proxies."""

    REFERENCE_METHOD = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.REFERENCE_METHOD not in REFERENCE_METHODS:
            raise ValueError(f"Unknown FedEMoE baseline: {self.REFERENCE_METHOD}")
        self.best_accuracy = 0.0
        self.best_round = 0
        self.last_eval_metrics: Dict[str, float] = {}

        requested_device = str(self.config.get("device", "cuda"))
        self._device_obj = torch.device(
            "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
        )

    def _seed_reference(self) -> None:
        seed = int(self.config.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    async def run(self) -> Dict[str, Any]:
        rounds = int(self.config.get("max_rounds", self.config.get("num_rounds", 500)))
        self._config["max_rounds"] = rounds
        self._seed_reference()

        initialize = getattr(self.aggregator, "initialize", None)
        if not callable(initialize):
            raise TypeError("FedEMoE baseline trainer requires its reference aggregator")
        initialize()

        if self.has_global_test:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=int(
                    self.config.get("eval_batch_size", self.config.get("batch_size", 64))
                ),
                shuffle=False,
                num_workers=int(
                    self.config.get("eval_num_workers", self.config.get("num_workers", 4))
                ),
                pin_memory=True,
            )
            self.aggregator.set_test_loader(test_loader)
        self._model = self.aggregator.get_global_model()

        summary = await super().run()
        summary.update(
            {
                "algorithm": self.REFERENCE_METHOD,
                "implementation": "fedemoe_comparison_artifact",
                "best_accuracy": float(self.best_accuracy),
                "best_round": int(self.best_round),
                "final_accuracy": float(
                    self.last_eval_metrics.get("accuracy", self.best_accuracy)
                ),
                "final_metrics": dict(self.last_eval_metrics),
            }
        )
        return summary

    def _learner_map(self) -> Dict[int, Any]:
        connected = self.get_connected_learners()
        learner_map = {_proxy_client_number(proxy): proxy for proxy in connected}
        expected = int(self.config.get("num_clients", len(learner_map)))
        missing = sorted(set(range(expected)) - set(learner_map))
        if missing:
            raise RuntimeError(f"Missing reference learners: {missing[:10]}")
        return learner_map

    async def train_round(self, round_num: int) -> RoundResult:
        method = self.REFERENCE_METHOD
        if self.callbacks:
            await self.callbacks.on_round_begin(
                self, round_num, {"algorithm": method}
            )

        learner_map = self._learner_map()
        total_clients = int(self.config.get("num_clients", len(learner_map)))
        clients_per_round = int(
            self.config.get(
                "clients_per_round",
                max(1, round(total_clients * float(self.config.get("client_fraction", 0.1)))),
            )
        )
        selected_ids = random.sample(
            range(total_clients), min(clients_per_round, total_clients)
        )
        fit_config = dict(self.config.get("fit_config", {}))
        fit_config.setdefault("epochs", int(self.config.get("local_epochs", 5)))
        fit_config["global_round"] = round_num

        updates: List[ClientUpdate] = []
        failures: List[str] = []
        for position, client_id in enumerate(selected_ids):
            proxy = learner_map[client_id]
            try:
                payload = self.aggregator.make_client_payload(
                    client_id, position, round_num
                )
                installed = await proxy.prepare_reference_round(payload)
                if installed is not True:
                    raise RuntimeError(f"state installation returned {installed!r}")
                result = await proxy.fit(fit_config)
                if not isinstance(result, TrainResult) and type(result).__name__ == "TrainResult":
                    result = TrainResult.from_dict(result.to_dict())
                if not isinstance(result, TrainResult):
                    raise TypeError(f"learner returned {type(result).__name__}")
                updates.append(
                    ClientUpdate.from_result(
                        str(getattr(proxy, "_target_id", client_id)), result
                    )
                )
            except Exception as exc:
                failures.append(f"client {client_id}: {exc}")

        if failures:
            raise RuntimeError(
                f"{method} round {round_num} has {len(failures)} failed client(s): "
                f"{failures[0]}"
            )

        self.aggregator.aggregate_round(updates, round_num)
        self._model = self.aggregator.get_global_model()
        metrics = self._aggregate_client_metrics(updates)

        eval_interval = int(
            self.config.get("eval_interval", self.config.get("log_interval", 10))
        )
        if round_num % eval_interval == 0 or round_num == self.config["max_rounds"]:
            self.last_eval_metrics = {
                key: float(value) for key, value in self.aggregator.evaluate().items()
            }
            metrics.update(
                {f"eval_{key}": value for key, value in self.last_eval_metrics.items()}
            )
            accuracy = float(self.last_eval_metrics.get("accuracy", 0.0))
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_round = round_num
            self.logger.info(
                f"[{method.upper()}] round={round_num} accuracy={accuracy:.4f} "
                f"best={self.best_accuracy:.4f}@{self.best_round}"
            )

        total_samples = sum(update.num_samples for update in updates)
        result = RoundResult(
            round_num=round_num,
            updates=updates,
            aggregated_weights={"algorithm": method, "round_num": round_num},
            metrics=RoundMetrics(
                round_num=round_num,
                num_clients=len(updates),
                total_samples=total_samples,
                metrics=metrics,
            ),
            metadata={"algorithm": method, "selected_client_ids": selected_ids},
        )

        if self.callbacks:
            await self.callbacks.on_round_end(
                self,
                round_num,
                {"round_result": result, "algorithm": method},
            )
        result.updates.clear()
        return result

    @staticmethod
    def _aggregate_client_metrics(updates: List[ClientUpdate]) -> Dict[str, float]:
        if not updates:
            return {}
        keys = list(updates[0].metrics.keys())
        return {
            key: sum(float(update.metrics[key]) for update in updates) / len(updates)
            for key in keys
        }


def _register_reference_trainers() -> None:
    for method in REFERENCE_METHODS:
        class_name = "FedEMoE" + "".join(part.title() for part in method.split("_")) + "Trainer"
        component_name = f"fedemoe_{method}"
        cls = type(
            class_name,
            (FedEMoEBaselineTrainer,),
            {"REFERENCE_METHOD": method, "__module__": __name__},
        )
        cls = trainer(
            name=component_name,
            description=f"FedEMoE artifact {method} native orchestration",
            version="1.0",
            author="FedEMoE Authors",
            algorithms=[method],
            upstream="Stephen-Chow1/FedEMoE-CEGA",
        )(cls)
        globals()[class_name] = cls


_register_reference_trainers()

__all__ = ["FedEMoEBaselineTrainer"] + [
    "FedEMoE" + "".join(part.title() for part in method.split("_")) + "Trainer"
    for method in REFERENCE_METHODS
]
