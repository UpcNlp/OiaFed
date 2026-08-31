from __future__ import annotations

import asyncio
import copy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import oiafed.methods  # noqa: F401 - triggers built-in component registration
from oiafed.core.types import ClientUpdate
from oiafed.config.tracker import parse_tracker_config
from oiafed.infra.logging import setup_logging
from oiafed.methods.aggregators.oneshot import OneShotBundleAggregator
from oiafed.methods.learners.fl.oneshot import OFedAvgLearner
from oiafed.methods.models.oneshot import OneShotEnsemble
from oiafed.methods.trainers.oneshot import FedCGSTrainer
from oiafed.papers.loader import reload_registry
from oiafed.registry import registry


class _ConstantLogits(nn.Module):
    def __init__(self, logits: list[float]):
        super().__init__()
        self.register_buffer("value", torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.value.expand(inputs.size(0), -1)


def test_validated_oneshot_methods_are_registered_and_generate_configs():
    methods = ["ofedavg", "ensemble", "fedcgs"]
    paper_registry = reload_registry()
    for method in methods:
        assert registry.get(f"trainer.{method}") is not None
        assert registry.get(f"learner.{method}") is not None
        assert paper_registry.get(method) is not None
        assert paper_registry.get(method).category == "OFL"
        configs = paper_registry.generate_node_configs(method, num_clients=2)
        assert len(configs) == 3
        for config in configs:
            for dataset in config.get("datasets", []):
                assert "server_test" not in dataset.get("args", {})
        train_partitions = [
            dataset["partition"]
            for config in configs
            for dataset in config.get("datasets", [])
            if dataset.get("split") == "train"
        ]
        assert train_partitions
        assert all(partition["seed"] == 0 for partition in train_partitions)
        assert all(partition["alpha"] == 0.05 for partition in train_partitions)

    assert paper_registry.get("fedsra").category == "OFL"
    assert set(methods + ["fedsra"]).issubset(paper_registry.list_by_category("OFL"))
    assert "OFL" in paper_registry.get_categories()
    for method in ("ofedavg", "ensemble"):
        assert paper_registry.get_defaults(method)["trainer"]["local_epochs"] == 300

    for method in ("fafi", "fusefl", "coboosting"):
        assert not registry.exists(f"trainer.{method}")
        assert not registry.exists(f"learner.{method}")
        assert paper_registry.get(method) is None


def test_disabled_tracker_accepts_generator_null_backends():
    config = parse_tracker_config({"enabled": False, "backends": None})
    assert config.enabled is False
    assert config.get_backends() == []


def test_supervised_oneshot_checkpoint_resumes_completed_client(tmp_path):
    setup_logging(
        node_id="learner_0",
        console=False,
        log_dir=str(tmp_path / "logs"),
    )
    dataset = TensorDataset(torch.randn(6, 2), torch.tensor([0, 1, 0, 1, 0, 1]))
    config = {
        "batch_size": 2,
        "device": "cpu",
        "num_classes": 2,
        "checkpoint_dir": str(tmp_path),
        "checkpoint_signature": "effect-cell-v1",
        "resume": True,
    }
    first = OFedAvgLearner(
        model=nn.Linear(2, 2),
        datasets={"train": [dataset]},
        config=config,
        node_id="learner_0",
    )
    trained = asyncio.run(first.fit({"epochs": 1}))
    assert (tmp_path / "learner_0" / "model.pt").exists()

    second = OFedAvgLearner(
        model=nn.Linear(2, 2),
        datasets={"train": [dataset]},
        config=config,
        node_id="learner_0",
    )
    resumed = asyncio.run(second.fit({"epochs": 1}))
    assert trained.metrics.total_epochs == 1
    assert resumed.metrics.total_epochs == 0
    assert resumed.metadata["resumed_from_checkpoint"] is True
    for name, value in first.model.state_dict().items():
        assert torch.equal(value.cpu(), second.model.state_dict()[name].cpu())


def test_bundle_aggregator_retains_every_client_and_clones_tensors():
    tensor = torch.tensor([1.0])
    updates = [
        ClientUpdate("a", {"weight": tensor}, 2, metadata={"class_counts": {0: 2}}),
        ClientUpdate("b", {"weight": torch.tensor([3.0])}, 4, metadata={"class_counts": {1: 4}}),
    ]
    bundle = OneShotBundleAggregator().aggregate(updates)
    tensor.add_(10)
    assert bundle["format"] == "oiafed.oneshot.bundle"
    assert [client["client_id"] for client in bundle["clients"]] == ["a", "b"]
    assert bundle["clients"][0]["state_dict"]["weight"].item() == 1.0


def test_direct_ensemble_uses_requested_logit_weights():
    ensemble = OneShotEnsemble(
        [_ConstantLogits([2.0, 0.0]), _ConstantLogits([0.0, 4.0])],
        [1, 3],
    )
    output = ensemble(torch.zeros(2, 1))
    expected = torch.tensor([[0.5, 3.0], [0.5, 3.0]])
    torch.testing.assert_close(output, expected)


def test_fedcgs_sufficient_statistics_match_direct_pooled_covariance():
    features = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [3.0, 2.0], [4.0, 2.0]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 0, 1, 1])

    metadata = []
    for indices in ([0, 2], [1, 3]):
        local = features[indices]
        local_labels = labels[indices]
        class_sums = torch.zeros(2, 2, dtype=torch.float64)
        class_counts = torch.zeros(2, dtype=torch.float64)
        class_sums.index_add_(0, local_labels, local)
        class_counts.index_add_(0, local_labels, torch.ones(2, dtype=torch.float64))
        metadata.append(
            {
                "class_sums": class_sums,
                "class_count_vector": class_counts,
                "feature_sum": local.sum(0),
                "second_moment": local.T @ local,
            }
        )

    weights, bias, covariance = FedCGSTrainer.aggregate_statistics(metadata, ridge=0.1)
    centered = features - features.mean(0)
    expected = centered.T @ centered / (features.size(0) - 1) + 0.1 * torch.eye(2, dtype=torch.float64)
    torch.testing.assert_close(covariance.double(), expected)
    logits = torch.nn.functional.linear(features.float(), weights, bias)
    assert torch.equal(logits.argmax(1), labels)
