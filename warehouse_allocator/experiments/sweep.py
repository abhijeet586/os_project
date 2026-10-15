"""Parameter sweeps over the admission thresholds.

Reporting one arbitrary parameter value is weak evidence: any policy can be made
to look good or bad by choosing its threshold badly. A sweep shows the shape of
the tradeoff and locates the optimum, which is the only honest way to compare a
tuned policy against an untuned baseline.

Each parameter is swept against the policy it controls: ``radius`` against
``RadiusPolicy``, ``eta_threshold`` against ``ETAPolicy``, and
``lease_duration`` against ``LeasePolicy``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from warehouse_allocator.config import SimConfig
from warehouse_allocator.experiments.runner import aggregate, write_csv
from warehouse_allocator.metrics.analysis import analyse
from warehouse_allocator.simulation import Simulation, build_trace

__all__ = [
    "PARAMETER_POLICY",
    "SweepPoint",
    "default_sweep",
    "main",
    "run_sweep",
]

# Which policy each swept parameter actually controls.
PARAMETER_POLICY: dict[str, str] = {
    "radius": "radius",
    "eta_threshold": "eta",
    "lease_duration": "lease",
}

DEFAULT_GRID: dict[str, tuple[float, ...]] = {
    "radius": (10.0, 20.0, 30.0, 40.0, 60.0, 90.0),
    "eta_threshold": (5.0, 10.0, 20.0, 30.0, 45.0, 70.0),
    "lease_duration": (60.0, 120.0, 180.0, 300.0, 450.0, 600.0),
}


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One swept parameter value, aggregated across seeds.

    Attributes:
        parameter: The parameter swept.
        value: The value tested.
        policy: The policy the parameter controls.
        throughput_mean: Mean throughput in tasks per simulated hour.
        throughput_half_width: 95% CI half-width on throughput.
        unused_mean: Mean held-but-unused fraction.
        wait_p95_mean: Mean 95th-percentile wait, seconds.
        n_seeds: Seeds aggregated.
    """

    parameter: str
    value: float
    policy: str
    throughput_mean: float
    throughput_half_width: float
    unused_mean: float
    wait_p95_mean: float
    n_seeds: int

    def to_row(self) -> dict[str, Any]:
        """Flatten to a CSV row.

        Returns:
            A flat mapping of column name to value.
        """
        return {
            "parameter": self.parameter,
            "value": self.value,
            "policy": self.policy,
            "throughput_per_hour": round(self.throughput_mean, 4),
            "throughput_ci95_half_width": round(self.throughput_half_width, 4),
            "held_but_unused_fraction": round(self.unused_mean, 6),
            "wait_p95_s": round(self.wait_p95_mean, 4),
            "n_seeds": self.n_seeds,
        }


def run_sweep(
    config: SimConfig,
    parameter: str,
    values: Sequence[float],
    seeds: Sequence[int],
    *,
    verbose: bool = True,
) -> list[SweepPoint]:
    """Sweep one parameter across values, aggregating over seeds.

    Args:
        config: Base configuration.
        parameter: Parameter name to sweep.
        values: Values to test.
        seeds: Master seeds.
        verbose: Whether to print progress.

    Returns:
        One :class:`SweepPoint` per value.

    Raises:
        ValueError: If the parameter does not control a policy.
    """
    if parameter not in PARAMETER_POLICY:
        raise ValueError(
            f"unknown sweep parameter {parameter!r}; "
            f"expected one of {sorted(PARAMETER_POLICY)}"
        )
    policy = PARAMETER_POLICY[parameter]
    points: list[SweepPoint] = []

    for value in values:
        throughputs: list[float] = []
        unused: list[float] = []
        waits: list[float] = []
        for seed in seeds:
            run_config = config.evolve(seed=seed, policy=policy, **{parameter: value})
            # Regenerated per seed but independent of the swept parameter, so
            # every point on the curve sees the identical workload.
            trace = build_trace(run_config)
            simulation = Simulation(run_config, trace=trace)
            simulation.run()
            analysis = analyse(simulation.collector, run_config)
            throughputs.append(analysis.throughput_per_hour)
            unused.append(analysis.total_unused_fraction)
            waits.append(analysis.wait_p95)

        summary = aggregate(policy, "throughput_per_hour", throughputs)
        point = SweepPoint(
            parameter=parameter,
            value=float(value),
            policy=policy,
            throughput_mean=summary.mean,
            throughput_half_width=summary.half_width,
            unused_mean=float(sum(unused) / len(unused)),
            wait_p95_mean=float(sum(waits) / len(waits)),
            n_seeds=len(seeds),
        )
        points.append(point)
        if verbose:
            print(
                f"  {parameter}={value:<7g} "
                f"{point.throughput_mean:7.2f} +/- "
                f"{point.throughput_half_width:5.2f} tasks/h  "
                f"waste={point.unused_mean:.3f}"
            )
    return points


def default_sweep(
    config: SimConfig,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    *,
    verbose: bool = True,
) -> list[SweepPoint]:
    """Sweep every tunable admission threshold.

    Args:
        config: Base configuration.
        seeds: Master seeds.
        verbose: Whether to print progress.

    Returns:
        Every sweep point across all three parameters.
    """
    points: list[SweepPoint] = []
    for parameter, values in DEFAULT_GRID.items():
        if verbose:
            print(f"sweeping {parameter} ({PARAMETER_POLICY[parameter]} policy)")
        points.extend(run_sweep(config, parameter, values, seeds, verbose=verbose))
    return points


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for parameter sweeps.

    Args:
        argv: Argument vector; ``sys.argv[1:]`` if omitted.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        prog="python -m warehouse_allocator.experiments.sweep",
        description="Sweep admission thresholds and report the tradeoff curve.",
    )
    parser.add_argument("--seeds", default="1,2,3,4,5", help="comma-separated seeds")
    parser.add_argument("--duration", type=float, default=3600.0)
    parser.add_argument(
        "--parameter",
        default="all",
        choices=["all", *sorted(PARAMETER_POLICY)],
        help="which parameter to sweep",
    )
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    seeds = [int(token) for token in args.seeds.split(",") if token.strip()]
    config = SimConfig(duration=args.duration)

    if args.parameter == "all":
        points = default_sweep(config, seeds)
    else:
        print(f"sweeping {args.parameter}")
        points = run_sweep(config, args.parameter, DEFAULT_GRID[args.parameter], seeds)

    out = Path(args.out)
    write_csv(out / "sweep.csv", [point.to_row() for point in points], config)

    print(f"\n{'=' * 70}")
    print("SWEEP OPTIMA")
    print("=" * 70)
    by_parameter: dict[str, list[SweepPoint]] = {}
    for point in points:
        by_parameter.setdefault(point.parameter, []).append(point)
    for parameter, group in by_parameter.items():
        best = max(group, key=lambda item: item.throughput_mean)
        worst = min(group, key=lambda item: item.throughput_mean)
        print(
            f"{parameter:>15}: best {best.value:g} at "
            f"{best.throughput_mean:.2f} tasks/h  "
            f"(worst {worst.value:g} at {worst.throughput_mean:.2f})"
        )
    print(f"\nWrote sweep.csv to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
