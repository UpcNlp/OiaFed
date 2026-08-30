from __future__ import annotations

import copy

import torch
import torch.nn as nn

import oiafed.methods  # noqa: F401 - triggers built-in component registration
from oiafed.core.types import ClientUpdate
from oiafed.config.tracker import parse_tracker_config
from oiafed.methods.aggregators.oneshot import OneShotBundleAggregator
from oiafed.methods.models.oneshot import (
    DataFreeGenerator,
    FAFIServerModel,
    FuseFLResNet18,
    OneShotEnsemble,
)
from oiafed.methods.trainers.oneshot import CoBoostingTrainer, FedCGSTrainer
from oiafed.papers.loader import reload_registry
from oiafed.registry import registry


class _ConstantLogits(nn.Module):
    def __init__(self, logits: list[float]):
        super().__init__()
        self.register_buffer("value", torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.value.expand(inputs.size(0), -1)


class _ConstantFeatures(nn.Module):
    def __init__(self, features: list[float]):
        super().__init__()
        self.register_buffer("value", torch.tensor(features, dtype=torch.float32))

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.value.expand(inputs.size(0), -1)


def test_all_six_methods_are_registered_and_generate_configs():
    methods = ["ofedavg", "ensemble", "fafi", "fusefl", "fedcgs", "coboosting"]
    paper_registry = reload_registry()
    for method in methods:
        assert registry.get(f"trainer.{method}") is not None
        assert registry.get(f"learner.{method}") is not None
        assert paper_registry.get(method) is not None
        configs = paper_registry.generate_node_configs(method, num_clients=2)
        assert len(configs) == 3
        for config in configs:
            for dataset in config.get("datasets", []):
                assert "server_test" not in dataset.get("args", {})


def test_disabled_tracker_accepts_generator_null_backends():
    config = parse_tracker_config({"enabled": False, "backends": None})
    assert config.enabled is False
    assert config.get_backends() == []


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


def test_fafi_server_uses_data_size_features_and_uniform_global_prototypes():
    prototypes = torch.eye(2)
    server = FAFIServerModel(
        [_ConstantFeatures([1.0, 0.0]), _ConstantFeatures([0.0, 1.0])],
        [1, 3],
        prototypes,
    )
    logits = server(torch.zeros(1, 1))
    expected_features = torch.nn.functional.normalize(torch.tensor([[0.25, 0.75]]), dim=1)
    torch.testing.assert_close(logits, expected_features)


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


def test_fusefl_installs_all_client_branches_in_order():
    clients = [FuseFLResNet18(num_classes=3, base_width=2) for _ in range(2)]
    stage_states = []
    previous = None
    for stage in range(4):
        if previous is not None:
            for client in clients:
                client.install_fused_stage(stage - 1, previous)
        current = [client.export_stage_state(stage) for client in clients]
        stage_states.append(current)
        previous = current

    server = FuseFLResNet18(num_classes=3, base_width=2)
    for stage, states in enumerate(stage_states):
        server.install_fused_stage(stage, states)
    output = server(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 3)
    assert sum(isinstance(stage, nn.Module) for stage in server.stages) == 4


def test_coboosting_generator_and_kl_objective_are_finite():
    generator = DataFreeGenerator(latent_dim=8, width=4, image_size=32, channels=3)
    images = generator(torch.randn(2, 8))
    assert images.shape == (2, 3, 32, 32)
    assert 0 <= float(images.detach().min()) <= float(images.detach().max()) <= 1
    loss = CoBoostingTrainer._kl(torch.randn(2, 3), torch.randn(2, 3), 4.0)
    assert torch.isfinite(loss)
    assert loss >= 0
