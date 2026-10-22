"""Matplotlib figures for the experiment results.

All figures are written to ``results/`` as both PNG and PDF at 300 dpi, with
axis labels carrying units.

Design rules applied throughout, in the order the data-visualisation method
prescribes -- form first, colour last:

* Colour is assigned by *identity*, from a fixed categorical order that has been
  validated for colour-vision deficiency separation. Hues are never cycled and a
  policy keeps its colour across every figure, so ``lease`` is the same hue in
  figure 1 and figure 5.
* Every figure with two or more series carries a legend, and values are direct-
  labelled where there are few enough marks to do so without clutter. Identity
  is therefore never conveyed by colour alone -- three of these hues fall below
  3:1 contrast on a white surface, which obliges visible labels.
* Grid and axes are recessive; no figure uses two y-scales.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

# Selected before pyplot is imported: the figures are rendered headlessly in
# CI and on machines with no display, and the default backend would fail there.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from warehouse_allocator.config import SimConfig
from warehouse_allocator.metrics.analysis import (
    RunAnalysis,
    state_timeline,
)

if TYPE_CHECKING:
    from warehouse_allocator.experiments.runner import Aggregate
    from warehouse_allocator.experiments.sweep import SweepPoint

__all__ = ["POLICY_COLORS", "render_all", "save"]

# Validated categorical slots 1-4 (blue, orange, aqua, yellow). Worst adjacent
# CVD separation 9.1 dE, worst adjacent normal-vision separation 22.9 dE, both
# clear of their gates. A policy keeps its slot in every figure.
POLICY_COLORS: dict[str, str] = {
    "null": "#2a78d6",
    "radius": "#eb6834",
    "eta": "#1baf7a",
    "lease": "#eda100",
    "adaptive": "#9b51e0",
}

# Two-series slots for the held-versus-unused comparison.
HELD_COLOR = "#2a78d6"
UNUSED_COLOR = "#eb6834"

# Robot-state stack, ordered so related states sit adjacent.
STATE_COLORS: dict[str, str] = {
    "idle": "#c9c9c4",
    "waiting": "#eb6834",
    "moving_to_resource": "#2a78d6",
    "moving_to_dropoff": "#4a3aa7",
    "working": "#1baf7a",
    "charging": "#e87ba4",
}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#dedddA"
DPI = 300


def _style(axes: Axes) -> None:
    """Apply the recessive grid and axis treatment.

    Args:
        axes: The axes to style.
    """
    axes.grid(True, axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.9)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID_COLOR)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    axes.xaxis.label.set_color(TEXT_SECONDARY)
    axes.yaxis.label.set_color(TEXT_SECONDARY)
    axes.title.set_color(TEXT_PRIMARY)


def save(figure: Figure, out_dir: Path, stem: str) -> list[Path]:
    """Write a figure to PNG and PDF at 300 dpi.

    Args:
        figure: The figure to write.
        out_dir: Destination directory, created if absent.
        stem: File name without extension.

    Returns:
        The paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        written.append(path)
    plt.close(figure)
    return written


def _lookup(aggregates: Sequence[Aggregate]) -> dict[tuple[str, str], Aggregate]:
    """Index aggregates by policy and metric.

    Args:
        aggregates: Aggregates to index.

    Returns:
        A mapping from ``(policy, metric)`` to the aggregate.
    """
    return {(item.policy, item.metric): item for item in aggregates}


def figure_throughput(
    aggregates: Sequence[Aggregate], out_dir: Path, config: SimConfig
) -> list[Path]:
    """Figure 1: throughput by policy with 95% confidence intervals.

    Throughput is the headline metric, so it gets the first figure. Error bars
    are mandatory: the seed-to-seed spread here is wide enough to reverse a
    ranking, and a bare bar chart would hide that.

    Args:
        aggregates: Aggregated results.
        out_dir: Destination directory.
        config: Configuration, for the caption.

    Returns:
        The paths written.
    """
    lookup = _lookup(aggregates)
    policies = [p for p in POLICY_COLORS if (p, "throughput_per_hour") in lookup]
    values = [lookup[(p, "throughput_per_hour")].mean for p in policies]
    errors = [lookup[(p, "throughput_per_hour")].half_width for p in policies]
    seeds = lookup[(policies[0], "throughput_per_hour")].n if policies else 0

    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    bars = axes.bar(
        policies,
        values,
        color=[POLICY_COLORS[p] for p in policies],
        width=0.62,
        edgecolor="white",
        linewidth=2.0,
    )
    axes.errorbar(
        policies,
        values,
        yerr=errors,
        fmt="none",
        ecolor=TEXT_SECONDARY,
        elinewidth=1.6,
        capsize=6,
        capthick=1.6,
    )
    for bar, value, error in zip(bars, values, errors, strict=True):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + max(values) * 0.03,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_PRIMARY,
            fontweight="medium",
        )

    axes.set_ylabel("Throughput (tasks completed per simulated hour)")
    axes.set_xlabel("Admission policy")
    axes.set_title(
        "Throughput by admission policy", fontsize=13, fontweight="semibold", pad=12
    )
    axes.set_ylim(0, max(v + e for v, e in zip(values, errors, strict=True)) * 1.22)
    figure.text(
        0.5,
        -0.04,
        f"Mean of {seeds} seeds; error bars are 95% confidence intervals. "
        f"{config.robot_count} robots, arrival rate {config.task_arrival_rate}/s.",
        ha="center",
        fontsize=8.5,
        color=TEXT_SECONDARY,
    )
    _style(axes)
    return save(figure, out_dir, "fig1_throughput_by_policy")


def figure_wait_distribution(
    results: Sequence[RunAnalysis], out_dir: Path
) -> list[Path]:
    """Figure 2: wait time distribution by policy.

    Percentiles, never a bare mean. A mean would show an aggressive filter
    "winning" precisely because it refused the robots that would have waited
    longest.

    Args:
        results: Per-run analyses.
        out_dir: Destination directory.

    Returns:
        The paths written.
    """
    by_policy: dict[str, list[float]] = {}
    for result in results:
        by_policy.setdefault(result.policy, []).extend(
            [result.wait_p50, result.wait_p95, result.wait_p99, result.wait_max]
        )
    policies = [p for p in POLICY_COLORS if p in by_policy]

    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    box = axes.boxplot(
        [by_policy[p] for p in policies],
        tick_labels=policies,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": TEXT_PRIMARY, "linewidth": 1.8},
        whiskerprops={"color": TEXT_SECONDARY, "linewidth": 1.2},
        capprops={"color": TEXT_SECONDARY, "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": TEXT_SECONDARY,
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
    )
    for patch, policy in zip(box["boxes"], policies, strict=True):
        patch.set_facecolor(POLICY_COLORS[policy])
        patch.set_alpha(0.85)
        patch.set_edgecolor("white")
        patch.set_linewidth(2.0)

    axes.set_ylabel("Wait, first request to final grant (simulated seconds)")
    axes.set_xlabel("Admission policy")
    axes.set_title(
        "Wait time distribution by policy",
        fontsize=13,
        fontweight="semibold",
        pad=12,
    )
    figure.text(
        0.5,
        -0.04,
        "Each box pools the p50, p95, p99 and max wait from every seed. "
        "Wait includes all deferrals, so rejection cannot hide as improvement.",
        ha="center",
        fontsize=8.5,
        color=TEXT_SECONDARY,
    )
    _style(axes)
    return save(figure, out_dir, "fig2_wait_distribution")


def figure_sweep(points: Sequence[SweepPoint], out_dir: Path) -> list[Path]:
    """Figure 3: parameter sweep curves.

    Drawn as small multiples rather than one axis: radius is in metres while
    the two thresholds are in seconds, and forcing them onto a shared x-axis
    would be meaningless.

    Args:
        points: Sweep results.
        out_dir: Destination directory.

    Returns:
        The paths written.
    """
    parameters: dict[str, list[SweepPoint]] = {}
    for point in points:
        parameters.setdefault(point.parameter, []).append(point)

    labels = {
        "radius": ("radius (metres)", "radius"),
        "eta_threshold": ("ETA threshold (simulated seconds)", "eta"),
        "lease_duration": ("lease duration (simulated seconds)", "lease"),
    }
    names = [name for name in labels if name in parameters]
    if not names:
        names = list(parameters)

    figure, axes_row = plt.subplots(
        1, len(names), figsize=(4.6 * len(names), 4.4), squeeze=False
    )
    for index, name in enumerate(names):
        axes = axes_row[0][index]
        ordered = sorted(parameters[name], key=lambda item: item.value)
        xs = [item.value for item in ordered]
        ys = [item.throughput_mean for item in ordered]
        errs = [item.throughput_half_width for item in ordered]
        colour = POLICY_COLORS.get(labels.get(name, ("", "null"))[1], "#2a78d6")

        axes.plot(
            xs,
            ys,
            color=colour,
            linewidth=2.0,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="throughput",
        )
        axes.fill_between(
            xs,
            [y - e for y, e in zip(ys, errs, strict=True)],
            [y + e for y, e in zip(ys, errs, strict=True)],
            color=colour,
            alpha=0.16,
            linewidth=0,
        )
        best = max(ordered, key=lambda item: item.throughput_mean)
        axes.annotate(
            f"best {best.value:g}",
            xy=(best.value, best.throughput_mean),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=TEXT_PRIMARY,
        )
        axes.set_xlabel(labels.get(name, (name, ""))[0])
        if index == 0:
            axes.set_ylabel("Throughput (tasks per simulated hour)")
        axes.set_title(name, fontsize=11, fontweight="semibold")
        axes.legend(
            frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right"
        )
        # Horizontal margin keeps the "best" annotation on an end point inside
        # the panel; vertical headroom keeps it clear of the top spine.
        axes.margins(x=0.14)
        top = max(y + e for y, e in zip(ys, errs, strict=True))
        bottom = min(y - e for y, e in zip(ys, errs, strict=True))
        axes.set_ylim(min(bottom, 0.0), top * 1.12)
        _style(axes)

    figure.suptitle(
        "Parameter sweeps: throughput versus admission threshold",
        fontsize=13,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        -0.03,
        "Shaded band is the 95% confidence interval across seeds.",
        ha="center",
        fontsize=8.5,
        color=TEXT_SECONDARY,
    )
    figure.tight_layout()
    return save(figure, out_dir, "fig3_parameter_sweep")


def figure_state_area(
    timelines: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    out_dir: Path,
) -> list[Path]:
    """Figure 4: robot state fractions over simulated time.

    Args:
        timelines: One ``(times, series)`` pair per policy.
        out_dir: Destination directory.

    Returns:
        The paths written.
    """
    policies = [p for p in POLICY_COLORS if p in timelines]
    figure, axes_row = plt.subplots(
        1,
        len(policies),
        figsize=(4.0 * len(policies), 4.2),
        squeeze=False,
        sharey=True,
    )

    # Stacked bottom-to-top: productive states first, idle last, so the
    # growing "waiting" band reads as the thing eating the fleet's time.
    order = (
        "working",
        "moving_to_resource",
        "moving_to_dropoff",
        "waiting",
        "idle",
    )
    for index, policy in enumerate(policies):
        axes = axes_row[0][index]
        times, series = timelines[policy]
        stack = [series.get(name, np.zeros_like(times)) for name in order]
        axes.stackplot(
            times / 60.0,
            *stack,
            labels=[name.replace("_", " ") for name in order],
            colors=[STATE_COLORS[name] for name in order],
            edgecolor="white",
            linewidth=0.4,
        )
        axes.set_xlabel("Simulated time (minutes)")
        if index == 0:
            axes.set_ylabel("Fraction of fleet")
        axes.set_title(policy, fontsize=11, fontweight="semibold")
        axes.set_ylim(0, 1)
        axes.set_xlim(0, float(times.max()) / 60.0)
        _style(axes)
        axes.grid(False)

    handles, labels = axes_row[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(order),
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        bbox_to_anchor=(0.5, -0.08),
    )
    figure.suptitle(
        "Robot state over simulated time, by policy",
        fontsize=13,
        fontweight="semibold",
    )
    figure.tight_layout()
    return save(figure, out_dir, "fig4_robot_state_area")


def figure_held_vs_unused(results: Sequence[RunAnalysis], out_dir: Path) -> list[Path]:
    """Figure 5: held versus held-but-unused time, per class per policy.

    **This is the central figure of the project.** The full bar is how long a
    resource class was locked to some robot. The darker inner bar is how much
    of that time the holder had not yet arrived -- the resource was reserved
    and doing nothing. The gap between them is early-binding waste, and
    shrinking it is the entire purpose of the admission layer.

    Args:
        results: Per-run analyses.
        out_dir: Destination directory.

    Returns:
        The paths written.
    """
    classes: list[str] = []
    for result in results:
        for hold in result.holds:
            if hold.resource_class not in classes:
                classes.append(hold.resource_class)
    classes.sort()
    policies = [p for p in POLICY_COLORS if any(r.policy == p for r in results)]

    held: dict[tuple[str, str], list[float]] = {}
    unused: dict[tuple[str, str], list[float]] = {}
    for result in results:
        for hold in result.holds:
            held.setdefault((result.policy, hold.resource_class), []).append(
                hold.held_fraction
            )
            unused.setdefault((result.policy, hold.resource_class), []).append(
                hold.unused_fraction
            )

    figure, axes_row = plt.subplots(
        1, len(classes), figsize=(3.9 * len(classes), 4.6), squeeze=False, sharey=True
    )
    width = 0.62
    positions = np.arange(len(policies))

    for index, class_name in enumerate(classes):
        axes = axes_row[0][index]
        held_means = [
            float(np.mean(held.get((p, class_name), [0.0]))) for p in policies
        ]
        unused_means = [
            float(np.mean(unused.get((p, class_name), [0.0]))) for p in policies
        ]
        axes.bar(
            positions,
            held_means,
            width=width,
            color=HELD_COLOR,
            edgecolor="white",
            linewidth=2.0,
            label="held (granted to a robot)",
        )
        axes.bar(
            positions,
            unused_means,
            width=width * 0.55,
            color=UNUSED_COLOR,
            edgecolor="white",
            linewidth=2.0,
            label="held but unused (holder not yet arrived)",
        )
        for x, (total, waste) in enumerate(zip(held_means, unused_means, strict=True)):
            share = waste / total if total else 0.0
            axes.text(
                x,
                total + 0.02,
                f"{share:.0%} wasted",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=TEXT_PRIMARY,
            )
        axes.set_xticks(positions)
        axes.set_xticklabels(policies, fontsize=9)
        axes.set_title(class_name, fontsize=11, fontweight="semibold")
        axes.set_xlabel("Admission policy")
        if index == 0:
            axes.set_ylabel("Fraction of total resource-time")
        _style(axes)

    top = max(
        (float(np.mean(held.get((p, c), [0.0]))) for p in policies for c in classes),
        default=1.0,
    )
    for axes in axes_row[0]:
        axes.set_ylim(0, max(top * 1.25, 0.05))

    handles, labels = axes_row[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9.5,
        labelcolor=TEXT_SECONDARY,
        bbox_to_anchor=(0.5, -0.09),
    )
    figure.suptitle(
        "Early-binding waste: resource time held versus held-but-unused",
        fontsize=13,
        fontweight="semibold",
    )
    figure.tight_layout()
    return save(figure, out_dir, "fig5_held_vs_unused")


def render_all(
    results: Sequence[RunAnalysis],
    aggregates: Sequence[Aggregate],
    config: SimConfig,
    out_dir: Path,
    sweep_points: Sequence[SweepPoint] | None = None,
    timelines: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] | None = None,
) -> list[Path]:
    """Render every figure.

    Args:
        results: Per-run analyses.
        aggregates: Aggregated results.
        config: Base configuration.
        out_dir: Destination directory.
        sweep_points: Sweep results for figure 3. Computed if omitted.
        timelines: State timelines for figure 4. Computed if omitted.

    Returns:
        Every path written.
    """
    written: list[Path] = []
    written += figure_throughput(aggregates, out_dir, config)
    written += figure_wait_distribution(results, out_dir)

    if sweep_points is None:
        from warehouse_allocator.experiments.sweep import default_sweep

        sweep_points = default_sweep(config, seeds=(1, 2, 3), verbose=False)
    written += figure_sweep(sweep_points, out_dir)

    if timelines is None:
        timelines = compute_timelines(config)
    written += figure_state_area(timelines, out_dir)

    written += figure_held_vs_unused(results, out_dir)
    return written


def compute_timelines(
    config: SimConfig, seed: int = 1
) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Run one seed per policy and capture the state timeline of each.

    Args:
        config: Base configuration.
        seed: Seed to illustrate.

    Returns:
        A mapping of policy name to ``(times, series)``.
    """
    from warehouse_allocator.simulation import Simulation, build_trace

    seeded = config.evolve(seed=seed)
    trace = build_trace(seeded)
    timelines: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for policy in POLICY_COLORS:
        run_config = seeded.evolve(policy=policy)
        simulation = Simulation(run_config, trace=trace)
        simulation.run()
        timelines[policy] = state_timeline(simulation.collector, run_config)
    return timelines
