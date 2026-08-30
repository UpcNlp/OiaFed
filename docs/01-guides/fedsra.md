# FedSRA integration

FedSRA is integrated as a dedicated learner, trainer, aggregator, backbone, and
server ensemble. It is not a one-round FedAvg configuration: the server keeps
all client backbones and applies RGA to their features.

## Components

- `learner.fedsra`: frozen shared ETF plus R/I/J ERL training.
- `aggregator.fedsra`: validates and packages every client state and class-count
  record without parameter averaging.
- `trainer.fedsra`: enforces one round and full cohort participation, installs
  the shared ETF, collects updates, builds the server ensemble, and evaluates it.
- `model.fedsra_resnet18`: CIFAR-adapted ResNet-18 feature backbone.
- `FedSRAEnsemble`: exact full-loader RGA evaluation and optional calibrated
  online inference.

## RGA evaluation scope

The reference implementation computes each client's feature-wise mean and
standard deviation over the complete evaluation set. Consequently, reproducing
paper numbers must use `FedSRAEnsemble.predict_loader`, which collects the full
sample axis before z-score standardization. Calling the ensemble directly uses
the current batch unless calibration statistics have been installed and can
therefore produce different results.

RGA applies, in order:

1. Per-client, per-feature z-score standardization of the pre-L2 `fc` features.
2. A `sqrt(n_k)` weighted sum using each client's local training size.
3. Post-aggregation L2 normalization.
4. Nearest-ETF classification.

## Configuration

The paper definition is `oiafed/papers/defs/ofl/fedsra.yaml`. Generate node
configurations with the normal OiaFed paper workflow and keep these invariants:

```yaml
trainer:
  type: fedsra
  args:
    max_rounds: 1
    client_fraction: 1.0
    local_epochs: 600

learner:
  type: fedsra

aggregator:
  type: fedsra

model:
  type: fedsra_resnet18
  args:
    feature_dim: 256
    num_classes: 10
```

The trainer and learners must use the same `num_classes`, `feature_dim`, and
`etf_seed`. The trainer sends the actual ETF tensor before the sole local fit and
the strict aggregator verifies the returned metadata.
