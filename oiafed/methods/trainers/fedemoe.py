"""Native OiaFed orchestration for FedEMoE with EGCA."""

from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...core.trainer import Trainer
from ...core.types import ClientUpdate, RoundMetrics, RoundResult, TrainResult
from ...registry import trainer
from ..fedemoe_reference.metrics import Metrics


def _proxy_client_number(proxy: Any) -> int:
    target_id = str(getattr(proxy, "_target_id", ""))
    try:
        return int(target_id.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"FedEMoE learner id must end in an integer: {target_id}") from exc


@trainer(
    name="fedemoe",
    description="FedEMoE sequential cohort orchestration with EGCA model pool",
    version="1.0",
    author="FedEMoE Authors",
    algorithms=["fedemoe"],
    upstream="Stephen-Chow1/FedEMoE-CEGA",
)
class FedEMoETrainer(Trainer):
    """Map the validated FedEGSTrainer sequence onto OiaFed proxies."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.best_accuracy = 0.0
        self.best_round = 0
        self.last_eval_metrics: Dict[str, float] = {}
        self._reference_rng_restored = False

        requested_device = str(self.config.get("device", "cuda"))
        if requested_device == "cuda" and torch.cuda.is_available():
            self._device_obj = torch.device("cuda")
        else:
            self._device_obj = torch.device("cpu")

    async def run(self) -> Dict[str, Any]:
        rounds = int(self.config.get("max_rounds", self.config.get("num_rounds", 500)))
        self._config["max_rounds"] = rounds
        if self.model is None:
            raise ValueError("FedEMoETrainer requires the FedEMoE server model")
        self._model = self.model.to(self._device_obj)

        initialize = getattr(self.aggregator, "initialize", None)
        if not callable(initialize):
            raise TypeError("FedEMoETrainer requires FedEMoEAggregator")
        initialize(self.model)

        summary = await super().run()
        summary.update(
            {
                "algorithm": "fedemoe",
                "best_accuracy": float(self.best_accuracy),
                "best_round": int(self.best_round),
                "final_accuracy": float(
                    self.last_eval_metrics.get("accuracy", self.best_accuracy)
                ),
                "final_metrics": dict(self.last_eval_metrics),
            }
        )
        return summary

    def _restore_reference_rng(self) -> None:
        if self._reference_rng_restored:
            return
        seed = int(self.config.get("seed", getattr(self.model, "fedemoe_seed", 42)))
        random.seed(seed)
        np.random.seed(seed)

        post_init_state = getattr(self.model, "_fedemoe_post_init_rng_state", None)
        if post_init_state is None:
            torch.manual_seed(seed)
        else:
            torch.random.set_rng_state(post_init_state.cpu())
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self._reference_rng_restored = True

    def _learner_map(self) -> Dict[int, Any]:
        connected = self.get_connected_learners()
        learner_map = {_proxy_client_number(proxy): proxy for proxy in connected}
        expected = int(self.config.get("num_clients", len(learner_map)))
        missing = sorted(set(range(expected)) - set(learner_map))
        if missing:
            raise RuntimeError(
                f"FedEMoE requires the configured cohort; missing learners: {missing[:10]}"
            )
        return learner_map

    async def train_round(self, round_num: int) -> RoundResult:
        self._restore_reference_rng()
        if self.callbacks:
            await self.callbacks.on_round_begin(
                self, round_num, {"algorithm": "fedemoe"}
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
        distribution = self.aggregator.distribute(selected_ids)

        fit_config = dict(self.config.get("fit_config", {}))
        fit_config.setdefault(
            "epochs", int(self.config.get("local_epochs", 5))
        )
        fit_config["global_round"] = round_num

        updates: List[ClientUpdate] = []
        failures: List[str] = []
        for client_id, pool_index, weights in distribution:
            proxy = learner_map[client_id]
            try:
                installed = await proxy.set_fedemoe_weights(weights)
                if installed is not True:
                    raise RuntimeError(f"weight installation returned {installed!r}")
                result = await proxy.fit(fit_config)
                if not isinstance(result, TrainResult) and type(result).__name__ == "TrainResult":
                    result = TrainResult.from_dict(result.to_dict())
                if not isinstance(result, TrainResult):
                    raise TypeError(f"learner returned {type(result).__name__}")
                result.metadata = {
                    **(result.metadata or {}),
                    "pool_index": pool_index,
                    "global_round": round_num,
                }
                updates.append(
                    ClientUpdate.from_result(str(getattr(proxy, "_target_id", client_id)), result)
                )
            except Exception as exc:  # fail the complete paper cohort below
                failures.append(f"client {client_id}: {exc}")

        if failures:
            raise RuntimeError(
                f"FedEMoE round {round_num} has {len(failures)} failed client(s): "
                f"{failures[0]}"
            )

        pool_result = self.aggregator.aggregate(updates, self.model)
        train_metrics = self._aggregate_client_metrics(updates)
        metrics = dict(train_metrics)

        eval_interval = int(
            self.config.get("eval_interval", self.config.get("log_interval", 10))
        )
        if round_num % eval_interval == 0 or round_num == self.config["max_rounds"]:
            self.last_eval_metrics = self._evaluate_global_model()
            metrics.update(
                {f"eval_{key}": value for key, value in self.last_eval_metrics.items()}
            )
            accuracy = float(self.last_eval_metrics.get("accuracy", 0.0))
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_round = round_num
            self.logger.info(
                f"[FedEMoE] round={round_num} accuracy={accuracy:.4f} "
                f"best={self.best_accuracy:.4f}@{self.best_round} "
                f"pool_diversity={pool_result['pool_diversity']:.4f}"
            )
            if self.tracker:
                self.tracker.log_metrics(
                    {f"eval_{key}": value for key, value in self.last_eval_metrics.items()},
                    step=round_num,
                )

        total_samples = sum(update.num_samples for update in updates)
        round_metrics = RoundMetrics(
            round_num=round_num,
            num_clients=len(updates),
            total_samples=total_samples,
            metrics=metrics,
        )
        result = RoundResult(
            round_num=round_num,
            updates=updates,
            aggregated_weights=pool_result,
            metrics=round_metrics,
            metadata={
                "algorithm": "fedemoe",
                "selected_client_ids": selected_ids,
                "pool_diversity": pool_result["pool_diversity"],
            },
        )

        if self.callbacks:
            await self.callbacks.on_round_end(
                self,
                round_num,
                {"round_result": result, "algorithm": "fedemoe"},
            )

        # ``Trainer.run`` retains every RoundResult until training finishes.
        # The full client state_dicts have already been consumed by EGCA and
        # keeping them would pin roughly 185 MiB of GPU memory per paper round.
        # Callbacks above still observe the complete payload; only the
        # no-longer-needed history copy is released.
        self._release_consumed_updates(result)
        return result

    @staticmethod
    def _release_consumed_updates(result: RoundResult) -> None:
        """Drop state_dict payloads after EGCA and callbacks consume them."""
        result.updates.clear()

    @staticmethod
    def _aggregate_client_metrics(updates: List[ClientUpdate]) -> Dict[str, float]:
        if not updates:
            return {}
        keys = list(updates[0].metrics.keys())
        return {
            key: sum(float(update.metrics[key]) for update in updates) / len(updates)
            for key in keys
        }

    @torch.no_grad()
    def _evaluate_global_model(self) -> Dict[str, float]:
        if not self.has_global_test:
            self.logger.warning("FedEMoE server has no test dataset")
            return {}

        global_model = self.aggregator.aggregate_global_model()
        global_model.eval()
        global_model.to(self._device_obj)
        self._model = global_model

        loader = DataLoader(
            self.test_dataset,
            batch_size=int(self.config.get("eval_batch_size", self.config.get("batch_size", 64))),
            shuffle=False,
            num_workers=int(self.config.get("eval_num_workers", self.config.get("num_workers", 4))),
            pin_memory=True,
        )
        metric = Metrics()
        for data, target in loader:
            data = data.to(self._device_obj)
            target = target.to(self._device_obj)
            output = global_model(data)
            prediction = output.logits.argmax(dim=1)
            confidence = torch.softmax(output.logits, dim=1).max(dim=1)[0]
            metric.update(
                preds=prediction,
                targets=target,
                uncertainties=output.router_output.uncertainty.squeeze(),
                confidences=confidence,
                dynamic_ks=output.router_output.dynamic_k,
            )
        return metric.compute()


__all__ = ["FedEMoETrainer"]
