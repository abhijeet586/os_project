"""Multi-policy, multi-seed headless experiment runner.

Two rules this module exists to enforce:

* **One trace per seed, replayed across every policy.** The workload is
  generated once from the seed and every policy runs that identical trace.
  Comparing policies on different workloads is not a comparison.
* **No single-seed result is reportable.** A minimum of five seeds is required
  and every aggregate carries a 95% confidence interval, because run-to-run
  variation here is large enough to reverse the ranking of two policies.

Every CSV carries the full :class:`~warehouse_allocator.config.SimConfig` JSON
in a header comment, so a result can always be traced to the parameters that
produced it.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from warehouse_allocator.config import SimConfig
from warehouse_allocator.metrics.analysis import RunAnalysis, analyse
from warehouse_allocator.simulation import Simulation, build_trace

__all__ = [
    "POLICIES",
    "Aggregate",
    "aggregate",
    "main",
    "run_experiment",
    "write_csv",
]

POLICIES: tuple[str, ...] = ("null", "radius", "eta", "lease", "adaptive")

# Below five seeds a t-interval is so wide as to be meaningless, and the spec
# forbids reporting single-seed results at all.
MIN_SEEDS = 5


@dataclass(frozen=True, slots=True)
class Aggregate:
    """A metric aggregated across seeds for one policy.

    Attributes:
        policy: The policy aggregated.
        metric: The metric name.
        mean: Sample mean across seeds.
        low: Lower bound of the 95% confidence interval.
        high: Upper bound of the 95% confidence interval.
        n: Number of seeds.
        values: The per-seed values, retained so the figures can show spread.
    """

    policy: str
    metric: str
    mean: float
    low: float
    high: float
    n: int
    values: tuple[float, ...]

    @property
    def half_width(self) -> float:
        """Return the CI half-width, for error bars.

        Returns:
            Half the interval width, zero if undefined.
        """
        if math.isnan(self.low) or math.isnan(self.high):
            return 0.0
        return (self.high - self.low) / 2.0


def aggregate(policy: str, metric: str, values: Sequence[float]) -> Aggregate:
    """Aggregate one metric across seeds with a 95% confidence interval.

    Uses a Student t interval rather than a normal one because the seed count
    is small; at five seeds the difference is substantial and the normal
    interval would overstate confidence.

    Args:
        policy: The policy the values came from.
        metric: The metric name.
        values: One value per seed.

    Returns:
        The aggregate. With fewer than two values or zero variance the
        interval collapses to the mean.
    """
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean()) if array.size else 0.0
    if array.size < 2:
        return Aggregate(policy, metric, mean, mean, mean, array.size, tuple(array))

    sem = float(stats.sem(array))
    if sem == 0.0 or math.isnan(sem):
        return Aggregate(policy, metric, mean, mean, mean, array.size, tuple(array))

    low, high = stats.t.interval(0.95, array.size - 1, loc=mean, scale=sem)
    return Aggregate(
        policy, metric, mean, float(low), float(high), array.size, tuple(array)
    )


def run_experiment(
    config: SimConfig,
    seeds: Sequence[int],
    policies: Sequence[str] = POLICIES,
    *,
    verbose: bool = True,
) -> list[RunAnalysis]:
    """Run every policy against every seed's replayed trace.

    Args:
        config: Base configuration; ``seed`` and ``policy`` are overridden.
        seeds: Master seeds to run.
        policies: Policy names to compare.
        verbose: Whether to print per-run progress.

    Returns:
        One :class:`RunAnalysis` per (seed, policy) pair.
    """
    results: list[RunAnalysis] = []
    for seed in seeds:
        seeded = config.evolve(seed=seed)
        # Generated once per seed and shared by every policy below. This is the
        # line that makes the comparison a comparison.
        trace = build_trace(seeded)
        if verbose:
            print(f"seed {seed}: {len(trace)} tasks arrive over {seeded.duration:.0f}s")

        for policy in policies:
            run_config = seeded.evolve(policy=policy)
            simulation = Simulation(run_config, trace=trace)
            simulation.run()
            analysis = analyse(simulation.collector, run_config)
            results.append(analysis)
            if verbose:
                print(
                    f"  {policy:>6}: "
                    f"{analysis.throughput_per_hour:7.2f} tasks/h  "
                    f"waste={analysis.total_unused_fraction:.3f}  "
                    f"p95_wait={analysis.wait_p95:7.1f}s  "
                    f"forced={analysis.force_admissions:4d}  "
                    f"unsafe={analysis.unsafe_refusals:4d}"
                )
    return results


def summarise(
    results: Sequence[RunAnalysis], metrics: Sequence[str] | None = None
) -> list[Aggregate]:
    """Aggregate results across seeds, per policy and metric.

    Args:
        results: Per-run analyses.
        metrics: Metric attribute names to aggregate. A sensible default set
            is used if omitted.

    Returns:
        One :class:`Aggregate` per (policy, metric).
    """
    chosen = list(
        metrics
        if metrics is not None
        else (
            "throughput_per_hour",
            "wait_p50",
            "wait_p95",
            "wait_p99",
            "wait_max",
            "total_held_fraction",
            "total_unused_fraction",
            "force_admissions",
            "lease_revocations",
            "unsafe_refusals",
            "tasks_completed",
        )
    )

    by_policy: dict[str, list[RunAnalysis]] = {}
    for result in results:
        by_policy.setdefault(result.policy, []).append(result)

    aggregates: list[Aggregate] = []
    for policy in POLICIES:
        runs = by_policy.get(policy)
        if not runs:
            continue
        for metric in chosen:
            values = [float(getattr(run, metric)) for run in runs]
            aggregates.append(aggregate(policy, metric, values))
    return aggregates


def write_csv(path: Path, rows: Sequence[dict[str, Any]], config: SimConfig) -> None:
    """Write rows to CSV with the config embedded as a header comment.

    Args:
        path: Destination file.
        rows: Rows to write; the first row's keys define the columns.
        config: Configuration recorded for provenance.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# SimConfig: {config.to_json(indent=None)}\n")
        handle.write(buffer.getvalue())


def write_summary_csv(
    path: Path, aggregates: Sequence[Aggregate], config: SimConfig
) -> None:
    """Write the aggregated summary with confidence intervals.

    Args:
        path: Destination file.
        aggregates: Aggregates to write.
        config: Configuration recorded for provenance.
    """
    rows = [
        {
            "policy": item.policy,
            "metric": item.metric,
            "mean": round(item.mean, 6),
            "ci95_low": round(item.low, 6),
            "ci95_high": round(item.high, 6),
            "ci95_half_width": round(item.half_width, 6),
            "n_seeds": item.n,
        }
        for item in aggregates
    ]
    write_csv(path, rows, config)


def _report(aggregates: Sequence[Aggregate], seeds: Sequence[int]) -> None:
    """Print the headline comparison to stdout.

    Args:
        aggregates: Aggregated results.
        seeds: Seeds the results came from.
    """
    lookup = {(item.policy, item.metric): item for item in aggregates}

    print(f"\n{'=' * 78}")
    print(f"RESULTS over {len(seeds)} seeds (mean +/- 95% CI)")
    print("=" * 78)
    header = (
        f"{'policy':>8} {'throughput/h':>22} {'held':>8} {'unused':>8} "
        f"{'p95 wait':>10} {'forced':>8}"
    )
    print(header)
    print("-" * 78)
    for policy in POLICIES:
        throughput = lookup.get((policy, "throughput_per_hour"))
        if throughput is None:
            continue
        held = lookup[(policy, "total_held_fraction")]
        unused = lookup[(policy, "total_unused_fraction")]
        wait = lookup[(policy, "wait_p95")]
        forced = lookup[(policy, "force_admissions")]
        print(
            f"{policy:>8} "
            f"{throughput.mean:9.2f} +/- {throughput.half_width:<8.2f} "
            f"{held.mean:8.3f} {unused.mean:8.3f} "
            f"{wait.mean:9.1f}s {forced.mean:8.1f}"
        )
    print("-" * 78)

    ranked = sorted(
        (
            lookup[(policy, "throughput_per_hour")]
            for policy in POLICIES
            if (policy, "throughput_per_hour") in lookup
        ),
        key=lambda item: item.mean,
        reverse=True,
    )
    if len(ranked) >= 2:
        best, second = ranked[0], ranked[1]
        delta = best.mean - second.mean
        pct = (delta / second.mean * 100.0) if second.mean else 0.0
        overlaps = best.low <= second.high
        print(
            f"\nHighest throughput: {best.policy} at {best.mean:.2f} tasks/h, "
            f"{delta:+.2f} ({pct:+.1f}%) over {second.policy}."
        )
        print(
            "  Confidence intervals OVERLAP, so this ranking is NOT "
            "statistically supported."
            if overlaps
            else "  Confidence intervals are disjoint; the ranking is supported."
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the experiment runner.

    Args:
        argv: Argument vector; ``sys.argv[1:]`` if omitted.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        prog="python -m warehouse_allocator.experiments.runner",
        description="Compare admission policies across seeds, headless.",
    )
    parser.add_argument(
        "--seeds",
        default="1,2,3,4,5",
        help="comma-separated master seeds (minimum 5 for reportable results)",
    )
    parser.add_argument(
        "--duration", type=float, default=3600.0, help="simulated seconds per run"
    )
    parser.add_argument(
        "--policies", default=",".join(POLICIES), help="comma-separated policy names"
    )
    parser.add_argument("--robots", type=int, default=None, help="override robot count")
    parser.add_argument("--rate", type=float, default=None, help="task arrival rate")
    parser.add_argument(
        "--radius", type=float, default=None, help="RadiusPolicy radius"
    )
    parser.add_argument(
        "--eta-threshold", type=float, default=None, help="ETAPolicy threshold"
    )
    parser.add_argument(
        "--lease-duration", type=float, default=None, help="LeasePolicy duration"
    )
    parser.add_argument(
        "--max-deferrals", type=int, default=None, help="deferrals before forcing"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results"), help="output directory"
    )
    parser.add_argument(
        "--figures", action="store_true", help="also render the figures"
    )
    args = parser.parse_args(argv)

    seeds = [int(token) for token in args.seeds.split(",") if token.strip()]
    policies = [token.strip() for token in args.policies.split(",") if token.strip()]

    overrides: dict[str, Any] = {"duration": args.duration}
    if args.robots is not None:
        overrides["robot_count"] = args.robots
    if args.rate is not None:
        overrides["task_arrival_rate"] = args.rate
    if args.radius is not None:
        overrides["radius"] = args.radius
    if args.eta_threshold is not None:
        overrides["eta_threshold"] = args.eta_threshold
    if args.lease_duration is not None:
        overrides["lease_duration"] = args.lease_duration
    if args.max_deferrals is not None:
        overrides["max_deferrals"] = args.max_deferrals

    config = SimConfig(**overrides)

    if len(seeds) < MIN_SEEDS:
        print(
            f"WARNING: {len(seeds)} seed(s) given. Results below {MIN_SEEDS} "
            "seeds are not reportable and must not appear in RESULTS.md."
        )

    results = run_experiment(config, seeds, policies)
    aggregates = summarise(results)

    out = Path(args.out)
    for result in results:
        write_csv(
            out / f"run_{result.policy}_seed{result.seed}.csv",
            [result.to_row()],
            config.evolve(policy=result.policy, seed=result.seed),
        )
    write_csv(out / "all_runs.csv", [result.to_row() for result in results], config)
    write_summary_csv(out / "summary.csv", aggregates, config)

    _report(aggregates, seeds)
    print(f"\nWrote {len(results)} run CSVs, all_runs.csv and summary.csv to {out}/")

    if args.figures:
        from warehouse_allocator.metrics.plots import render_all

        paths = render_all(results, aggregates, config, out)
        print(f"Rendered {len(paths)} figures to {out}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
