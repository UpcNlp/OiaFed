# FedEMoE integration

FedEMoE is classified as **horizontal federated learning (HFL)**. It is a
multi-round method with ten sampled clients per round, so it does not belong to
the one-shot (OFL), federated continual learning (FCL), vertical FL (VFL), or
federated unlearning (FU) groups.

The implementation is native to OiaFed's component system:

- `model.fedemoe`: evidential MoE with uncertainty-driven dynamic Top-K routing.
- `learner.fedemoe`: the validated local EDL objective and evidence-signature extraction.
- `aggregator.fedemoe`: the persistent model pool and Evidence-Guided Class-Aware Aggregation (EGCA).
- `trainer.fedemoe`: reference-order client sampling, pool distribution, sequential local training, pool update, and centralized evaluation.
- `fedemoe_dirichlet`: the reference cumulative-rounding Dirichlet partition.

The numerical core under `oiafed/methods/fedemoe_reference/` is copied from the
supplied `FedEMoE-CEGA-master.zip` artifact. The supplied ZIP has SHA-256
`636118A6A96A309202FF093F64B89E4C88205A431E7A7867186580AA1D04DBCB`.
All core files are byte-for-byte identical; `client.py` changes only three
repository-absolute imports to package-relative imports. The upstream MIT
license is included beside those files.

## Paper-default run

The registered paper cell is CIFAR-10, CNN, Dirichlet alpha 0.5, 100 clients,
10 clients per round, 500 communication rounds, 5 local epochs, batch size 64,
SGD at 0.01, seed 42, and no augmentation. FedEMoE uses 8 experts, dynamic
K=1..7, a 10-model pool, 4 parents, and the diversity threshold 0.5.

```bash
oiafed run --paper fedemoe -n 100 --mode serial
```

For a quick wiring check, override the number of rounds and use fewer clients;
that is not a paper reproduction.

The durable cluster runner provides three fixed presets. `smoke` and `short`
validate wiring and the 100-client lifecycle; only `paper` is the exact
500-round reproduction:

```bash
python examples/cluster/fedemoe/run_validation.py --preset smoke
python examples/cluster/fedemoe/run_validation.py --preset short
python examples/cluster/fedemoe/run_validation.py --preset paper
```

Each run records the exact Git commit, protocol, generated node configs, final
accuracy, elapsed time, GPU peak memory, and a durable `DONE`/`FAILED` marker.

## Baselines in the supplied artifact

All archive baselines belong to HFL for this comparison. They are tracked
explicitly so an existing OiaFed method is not silently treated as equivalent
to a differently implemented paper baseline.

| Baseline | Archive role | OiaFed status for this integration |
|---|---|---|
| FedAvg | Main paper baseline | Existing native component; archive implementation remains the reproduction reference |
| FedProx | Main paper baseline | Existing native component; archive implementation remains the reproduction reference |
| FedProto | Additional archive baseline | Existing native component; not in the paper's main eight-baseline table |
| FedProc | Main paper baseline | Catalogued; validated through the immutable archive, not reimplemented here |
| FedLESAM | Main paper baseline | Catalogued; validated through the immutable archive, not reimplemented here |
| pFedHB | Main paper baseline | Catalogued; validated through the immutable archive, not reimplemented here |
| FedEvi | Main paper baseline | Catalogued; validated through the immutable archive, not reimplemented here |
| FedSym | Main paper baseline | Catalogued; validated through the immutable archive, not reimplemented here |
| FedMoE-DA | Main paper baseline | Catalogued; validated through the immutable archive, not reimplemented here |
| FedNTD | Additional archive baseline | Catalogued; not in the paper's main eight-baseline table |
| FedSOL | Additional archive baseline | Catalogued; not in the paper's main eight-baseline table |

The immutable-source golden run for the paper-default cell produced 80.96%
final accuracy for FedEMoE versus 81.03% in the paper. That number is a parity
target for the native OiaFed run. The completed native run produced 79.80% at
round 500; see [the validation report](../05-papers/fedemoe-native-validation.md)
for the exact protocol, integrity checks, and comparison.
