# FedEMoE native OiaFed validation

## Outcome

The native OiaFed integration completed the paper-default 500-round
CIFAR-10 experiment on cluster8. The official round-500 accuracy is **79.80%**.
This preserves the paper's qualitative advantage over the strongest reproduced
baseline, but it is not a strict numerical reproduction of the paper's 81.03%.

| Result | Final accuracy | Difference from native OiaFed |
|---|---:|---:|
| Paper | 81.03% | -1.23 pp |
| Validated upstream source | 80.96% | -1.16 pp |
| Native OiaFed, round 500 | **79.80%** | - |
| Native OiaFed best checkpoint, round 490 | 80.30% | +0.50 pp |
| Strongest reproduced baseline, FedMoE-DA | 78.32% | +1.48 pp |

The paper and upstream-source comparisons use final accuracy, not best
accuracy. The native result therefore reproduces the direction and ranking of
the reported effect, with a smaller margin, while falling 1.23 percentage
points below the paper value.

## Exact protocol

- Dataset/model: CIFAR-10, CNN, no augmentation.
- Partition: reference cumulative-rounding Dirichlet, alpha 0.5, seed 42.
- Federation: 100 clients, 10 selected per round, 500 rounds, 5 local epochs.
- Optimization: batch size 64, SGD, learning rate 0.01, momentum 0.9,
  weight decay 1e-4.
- FedEMoE: 8 experts, dynamic K=1..7, pool size 10, 4 parents,
  adaptive diversity threshold 0.5, smoothing 0, EMA momentum 1.
- Evaluation: the complete 10,000-example test set every 10 rounds; the
  reported result is round 500.

The generated federation contained 101 node configurations: one trainer and
100 learners. Learners had only their partitioned training split; the complete
test split was instantiated once on the trainer.

## Validation stages

| Stage | Scope | Result | Runtime | GPU peak allocation |
|---|---|---:|---:|---:|
| Smoke | 10 clients, 1 round, 1 local epoch, 1,000 samples | 9.20% | 12.9 s | 1,504 MB |
| Lifecycle | 100 clients, 2 rounds, 1 local epoch, complete data | 11.57% | 43.6 s | 2,136 MB |
| Paper default | 100 clients, 500 rounds, 5 local epochs, complete data | **79.80%** | 32,397.4 s | 6,597 MB |

The paper-default run took approximately 9 hours. Its best evaluation was
80.30% at round 490, but this value is not used for the paper comparison.

## Integration-only memory correction

The first native long run exposed an OiaFed history-retention issue: the base
trainer retained ten already-aggregated GPU state dictionaries in every
`RoundResult`, increasing device memory by roughly 185 MB per round. It reached
about 50.2 GB at round 210 and was stopped before the predicted out-of-memory
failure.

Commit `071bb957fe94f7d0a117d2bd3099149090d2acab` releases the consumed
`RoundResult.updates` only after EGCA and callbacks finish. This does not alter
client sampling, local optimization, evidence profiles, pool state, aggregation,
metrics, or any vendored FedEMoE numerical code. The replacement run held GPU
memory at about 9.7 GB through round 500 and completed normally.

## Integrity and evidence

- Branch: `codex/fedemoe-egca`
- Validated commit: `071bb957fe94f7d0a117d2bd3099149090d2acab`
- Local tests: 25 passed.
- Cluster FedEMoE tests: 8 passed.
- Cluster source worktree was clean after completion.
- All eight files in `fedemoe_reference/UPSTREAM_MANIFEST.sha256` passed
  `sha256sum -c` after completion.
- Result JSON SHA-256:
  `b305b06195b6c28141a25eed9da62ec3aa45e7482dc04c6560d2f5a5336ced95`
- Launch log SHA-256:
  `308288782059f486d90e82d8f7b5479f2082ed272a5ae6eee39abd56a885b33a`

Cluster evidence directory:

```text
/public/home/dongshou/OiaFed_FedEMoE_validation/abc5611/runs/native_oiafed/071bb95/paper
```

The directory contains the 101 generated YAML configurations, `launch.log`,
`result.json`, and the durable `DONE` marker.
