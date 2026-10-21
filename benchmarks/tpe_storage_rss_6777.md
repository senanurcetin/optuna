# Attribution probe for TPE / JournalStorage RSS growth (#6777)

This is **fork-only diagnostic evidence**, not a user-facing Optuna API proposal or proof
that `TPESampler` retains trial history. Before changing the sampler, isolate the contribution
of storage from sampler allocations as requested in [optuna/optuna#6777](https://github.com/optuna/optuna/issues/6777).

## Reproduction

[GitHub Actions run 39604046880](https://github.com/senanurcetin/optuna/actions/runs/39604046880)
succeeded for all four combinations. Full trial-by-trial checkpoints are attached to
the run as downloadable JSON artifacts.

- Code: `benchmarks/tpe_storage_rss_probe.py`
- Environment: GitHub-hosted Ubuntu, Python 3.12, Optuna `5.1.0.dev`
- Workload: 150 sequential, single-worker trials; 12 parameters (stepped integer,
  log float, categorical); seed 123; no objective-side model allocations
- TPE: `multivariate=True, constant_liar=True`; control: `RandomSampler`
- Storage: either `InMemoryStorage` or file-backed `JournalStorage`
- Measurement: process resident-set size (RSS) via `psutil`, checkpoints at
  trials 0, 30, 60, 90, 120 and 150
- Crucially, **each sampler/storage combination runs in a fresh GitHub runner process**.

| Sampler | Storage | RSS at trial 0 (MiB) | RSS at trial 150 (MiB) | Total RSS increase (MiB) | Increase between trial 30 and 150 (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Random | InMemory | 55.613 | 58.258 | 2.645 | 0.492 |
| Random | Journal | 49.363 | 50.398 | 1.035 | 0.554 |
| TPE | InMemory | 49.605 | 52.055 | 2.450 | 0.688 |
| TPE | Journal | 55.559 | 59.426 | 3.867 | 0.828 |

## Interpretation / limitations

RSS increases slightly in **all four** scenarios, including `RandomSampler`. Since
the initial resident sets differ across independently scheduled runners, comparing
the absolute initial or final RSS of two rows would be misleading.

The increase after the first 30 trials is below 1 MiB in each scenario.
With only 150 trials, one run per configuration, a single worker and a smaller
search space than the reported 36-parameter, multi-worker problem, these results
**do not establish an unbounded memory leak, identify its owner, or refute #6777**.
The experiment also cannot distinguish live references from allocator retention.

Before proposing changes: replicate several times with fixed platform/runtime,
test the exact reporter's 36-parameter and multi-process settings, profile storage
replay snapshots and live-object retention, and measure the memory trajectory over
thousands of trials (with a bounded resource budget). Compare slopes after warmup,
not only end-to-end RSS delta.

The workflow is intentionally diagnostic and has **no RSS failure threshold**.
