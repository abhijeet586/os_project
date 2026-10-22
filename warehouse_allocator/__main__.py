r"""Command-line entry point.

Every parameter adjustable in the interactive UI is also settable here, and
vice versa.

Examples:
    Run the interactive view with the lease policy::

        python -m warehouse_allocator --policy lease --seed 42 --duration 600

    Run the same thing headless and print the derived statistics::

        python -m warehouse_allocator --policy lease --seed 42 --headless \\
            --duration 3600
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from warehouse_allocator.config import SimConfig
from warehouse_allocator.metrics.analysis import analyse
from warehouse_allocator.simulation import Simulation

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m warehouse_allocator",
        description=(
            "Spatial admission control in front of Banker's algorithm, "
            "simulated on a warehouse robot fleet."
        ),
    )
    parser.add_argument(
        "--policy",
        default="null",
        choices=["null", "radius", "eta", "lease", "adaptive"],
        help="admission policy (default: null, the classical baseline)",
    )
    parser.add_argument("--seed", type=int, default=42, help="master seed")
    parser.add_argument(
        "--duration", type=float, default=600.0, help="simulated seconds to run"
    )
    parser.add_argument(
        "--headless", action="store_true", help="run without opening a window"
    )

    layout = parser.add_argument_group("layout")
    layout.add_argument("--grid-width", type=int, default=None)
    layout.add_argument("--grid-height", type=int, default=None)
    layout.add_argument("--obstacle-density", type=float, default=None)
    layout.add_argument("--cell-size", type=int, default=8, help="pixels per cell")

    fleet = parser.add_argument_group("fleet and workload")
    fleet.add_argument("--robots", type=int, default=None, help="robot count")
    fleet.add_argument("--rate", type=float, default=None, help="task arrival rate /s")
    fleet.add_argument("--robot-speed", type=float, default=None, help="metres/second")
    fleet.add_argument("--service-mean", type=float, default=None)
    fleet.add_argument("--battery-drain", type=float, default=None)
    fleet.add_argument("--dt", type=float, default=None, help="fixed timestep")
    fleet.add_argument(
        "--burst-interval", type=float, default=None, help="seconds between burst waves"
    )
    fleet.add_argument(
        "--burst-size", type=int, default=None, help="tasks per burst wave"
    )

    policy = parser.add_argument_group("policy parameters")
    policy.add_argument("--radius", type=float, default=None, help="metres")
    policy.add_argument("--eta-threshold", type=float, default=None, help="seconds")
    policy.add_argument("--lease-duration", type=float, default=None, help="seconds")
    policy.add_argument(
        "--max-deferrals", type=int, default=None, help="deferrals before forcing"
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--save-events", type=Path, default=None, help="write the event log to JSONL"
    )
    output.add_argument(
        "--save-trace", type=Path, default=None, help="write the task trace to JSON"
    )
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate parsed arguments into config overrides.

    Args:
        args: Parsed arguments.

    Returns:
        A mapping of config field to value, omitting unset options.
    """
    mapping = {
        "grid_width": args.grid_width,
        "grid_height": args.grid_height,
        "obstacle_density": args.obstacle_density,
        "robot_count": args.robots,
        "task_arrival_rate": args.rate,
        "robot_speed": args.robot_speed,
        "service_duration_mean": args.service_mean,
        "battery_drain_rate": args.battery_drain,
        "dt": args.dt,
        "radius": args.radius,
        "eta_threshold": args.eta_threshold,
        "lease_duration": args.lease_duration,
        "max_deferrals": args.max_deferrals,
        "burst_interval": args.burst_interval,
        "burst_size": args.burst_size,
    }
    overrides: dict[str, Any] = {
        "policy": args.policy,
        "seed": args.seed,
        "duration": args.duration,
    }
    overrides.update(
        {key: value for key, value in mapping.items() if value is not None}
    )
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    """Run the simulator from the command line.

    Args:
        argv: Argument vector; ``sys.argv[1:]`` if omitted.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    config = SimConfig(**_overrides(args))

    simulation = Simulation(config)
    if args.save_trace is not None:
        simulation.trace.save(args.save_trace)

    if args.headless:
        simulation.run()
    else:
        # Imported lazily so that a headless run never touches Pygame, and so
        # the package remains usable on a machine with no display libraries.
        from warehouse_allocator.viz.app import run_app

        run_app(simulation, cell_size=args.cell_size)

    if args.save_events is not None:
        simulation.collector.save(args.save_events)

    analysis = analyse(simulation.collector, config)
    print(f"\npolicy            {analysis.policy}")
    print(f"seed              {analysis.seed}")
    print(f"simulated         {simulation.clock.now():.1f} s")
    print(f"tasks created     {analysis.tasks_created}")
    print(f"tasks completed   {analysis.tasks_completed}")
    print(f"throughput        {analysis.throughput_per_hour:.2f} tasks/hour")
    print(
        f"wait p50/p95/p99  {analysis.wait_p50:.1f} / "
        f"{analysis.wait_p95:.1f} / {analysis.wait_p99:.1f} s"
    )
    print(f"held fraction     {analysis.total_held_fraction:.3f}")
    print(f"held-but-unused   {analysis.total_unused_fraction:.3f}")
    print(f"force-admissions  {analysis.force_admissions}")
    print(f"lease revocations {analysis.lease_revocations}")
    print(f"unsafe refusals   {analysis.unsafe_refusals}")
    if not analysis.comparable:
        print(
            "\nWARNING: a parameter was changed mid-run. This run is NOT "
            "comparable and must not be reported as experimental data."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
