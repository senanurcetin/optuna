"""Isolate sampler and storage RSS growth in Optuna (issue #6777).

This diagnostic is intentionally not a performance acceptance test: absolute RSS
varies by Python version, platform, allocator, and storage implementation.
Each experiment must run in a fresh process to avoid cross-run memory retention.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import optuna
import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler", choices=("random", "tpe"), required=True)
    parser.add_argument("--storage", choices=("memory", "journal"), required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.trials, args.features, args.interval) < 1:
        parser.error("trials, features and interval must be positive")
    return args


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    sampler = (
        optuna.samplers.TPESampler(seed=args.seed, multivariate=True, constant_liar=True)
        if args.sampler == "tpe"
        else optuna.samplers.RandomSampler(seed=args.seed)
    )
    proc = psutil.Process(os.getpid())
    records = []

    def record(n: int) -> None:
        records.append({"trials": n, "rss_mb": round(proc.memory_info().rss / (1024**2), 3)})

    def objective(trial: optuna.trial.Trial) -> float:
        # Same deterministic suggestion search space and objective for both samplers.
        score = 0.0
        for i in range(args.features):
            if i % 3 == 0:
                x = trial.suggest_int(f"integer_{i}", 0, 100, step=5)
                score += ((x - 40) / 100) ** 2
            elif i % 3 == 1:
                x = trial.suggest_float(f"float_{i}", 1e-3, 10, log=True)
                score += (x - 1) ** 2 / 100
            else:
                x = trial.suggest_categorical(f"cat_{i}", ["a", "b", "c"])
                score += 0.0 if x == "a" else 1.0
        return score

    with tempfile.TemporaryDirectory(prefix="optuna-rss-") as tempdir:
        storage = (
            optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(str(Path(tempdir) / "trials.log"))
            )
            if args.storage == "journal"
            else optuna.storages.InMemoryStorage()
        )
        study = optuna.create_study(storage=storage, sampler=sampler)
        record(0)
        for number in range(1, args.trials + 1):
            study.optimize(objective, n_trials=1, show_progress_bar=False)
            if number % args.interval == 0 or number == args.trials:
                record(number)

    out = {
        "optuna_version": optuna.__version__,
        "sampler": args.sampler,
        "storage": args.storage,
        "features": args.features,
        "seed": args.seed,
        "records": records,
        "rss_change_mb": round(records[-1]["rss_mb"] - records[0]["rss_mb"], 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
