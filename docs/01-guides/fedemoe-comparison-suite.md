# FedEMoE comparison suite

OiaFed exposes every algorithm implementation in the supplied FedEMoE
comparison artifact as a horizontal federated learning (HFL) paper entry.
The `fedemoe_` prefix intentionally distinguishes these implementations from
same-name methods already present in OiaFed.

| Paper ID | Artifact method | Model/server state |
|---|---|---|
| `fedemoe_fedavg` | FedAvg | sample-weighted global CNN |
| `fedemoe_fedprox` | FedProx | global CNN, proximal coefficient 0.1 |
| `fedemoe_fedsym` | FedSym | ten-model symbiosis pool |
| `fedemoe_fedproto` | FedProto | persistent local models and prototype cache |
| `fedemoe_fedproc` | FedProc | global CNN and class prototypes |
| `fedemoe_fedntd` | FedNTD | global student/teacher CNN |
| `fedemoe_fedsol` | FedSOL | proximal perturbation state |
| `fedemoe_fedlesam` | FedLESAM | current and previous global CNN |
| `fedemoe_pfedhb` | pFedHB | global prior and client posteriors |
| `fedemoe_fedmoeda` | FedMoE-DA | domain-matched router/expert MoE |
| `fedemoe_fedevi` | FedEvi | uncertainty-weighted global CNN |

For example:

```bash
oiafed papers show fedemoe_fedproc
oiafed papers run fedemoe_fedproc
```

All entries default to the comparison cell used for the FedEMoE CIFAR-10
result: CNN, Dirichlet alpha 0.5, 100 clients, 10 clients per round, 500
rounds, 5 local epochs, batch size 64, SGD learning rate 0.01, momentum 0.9,
weight decay 1e-4, seed 42, and no augmentation. Method-specific constants
come from `configs/comparison_config.yaml` in the artifact.

The numerical implementation is vendored in
`oiafed/methods/fedemoe_baselines_reference`. Seven source files are
byte-identical to the artifact. `baselines.py` differs only in six import
statements that were made package-relative (plus a final newline); its
algorithm statements are unchanged. Both upstream and vendored SHA-256
manifests are stored beside the source.

The native adapters preserve the original sequential client order and map the
artifact's client/server state to OiaFed learners, aggregators, and trainers.
`tests/methods/test_fedemoe_baseline_suite.py` executes one round of every
method through both paths and requires exact equality for every global tensor
and reported metric.

For durable cluster validation:

```bash
python examples/cluster/fedemoe/run_baseline_validation.py \
  --method fedproc --preset smoke

python examples/cluster/fedemoe/run_baseline_validation.py \
  --method fedproc --preset paper
```

The validation runner writes `result.json` plus `DONE` or `FAILED` markers and
resumes completed protocol-identical runs without rerunning them.
