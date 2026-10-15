"""Derived statistics, computed after a run from the event log alone.

Nothing here runs during a simulation. Every number is reconstructed from the
append-only log, so a new question can be asked of an old run without
re-simulating it.

Two deliberate reporting choices, both from BUILD_SPEC.md:

* **Throughput is the headline.** Mean wait time is trivially gamed -- reject
  aggressively enough and the rejected robots stop counting as waiting at all.
  ``tasks_completed_per_simulated_hour`` cannot be gamed that way.
* **Wait is reported as percentiles, never as a bare mean**, and is measured
  from *first request* to *final grant* inclusive of every deferral, so that
  rejection cannot disguise itself as improvement.

The function that matters most is :func:`resource_hold_breakdown`. The gap
between *held* and *held-but-unused* time is the direct measurement of early
binding waste, and it is the central finding of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from warehouse_allocator.config import SimConfig
from warehouse_allocator.core.entities import RobotState
from warehouse_allocator.metrics.collector import EventCollector
from warehouse_allocator.metrics.events import (
    LeaseRevoked,
    RequestDeferred,
    RequestForceAdmitted,
    RequestGranted,
    RequestRefusedUnsafe,
    ResourceOccupied,
    ResourceReleased,
    RobotStateChanged,
    TaskCompleted,
)

__all__ = [
    "HoldBreakdown",
    "RunAnalysis",
    "analyse",
    "resource_hold_breakdown",
    "tasks_completed_per_simulated_hour",
    "wait_percentiles",
]

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True, slots=True)
class HoldBreakdown:
    """Held versus held-but-unused time for one resource class.

    Attributes:
        resource_class: The class described.
        instances: How many instances of it exist.
        held_seconds: Total instance-seconds spent granted to some robot.
        unused_seconds: Of those, instance-seconds where the holder had not
            yet physically arrived.
        capacity_seconds: ``instances * duration``, the denominator.
    """

    resource_class: str
    instances: int
    held_seconds: float
    unused_seconds: float
    capacity_seconds: float

    @property
    def held_fraction(self) -> float:
        """Return the fraction of capacity-time the class was held.

        Returns:
            Held instance-seconds over capacity-seconds.
        """
        return (
            self.held_seconds / self.capacity_seconds if self.capacity_seconds else 0.0
        )

    @property
    def unused_fraction(self) -> float:
        """Return the fraction of capacity-time held but not yet reached.

        Returns:
            Unused instance-seconds over capacity-seconds. This is the direct
            measure of early-binding waste.
        """
        return (
            self.unused_seconds / self.capacity_seconds
            if self.capacity_seconds
            else 0.0
        )

    @property
    def waste_ratio(self) -> float:
        """Return the share of held time that was wasted.

        Returns:
            Unused seconds over held seconds; zero when nothing was held.
        """
        return self.unused_seconds / self.held_seconds if self.held_seconds else 0.0


@dataclass(frozen=True, slots=True)
class RunAnalysis:
    """Every derived statistic for one run.

    Attributes:
        policy: Admission policy name.
        seed: Master seed.
        duration: Simulated seconds run.
        comparable: False if a parameter was changed mid-run.
        tasks_created: Tasks that arrived.
        tasks_completed: Tasks that finished.
        throughput_per_hour: The headline metric.
        wait_p50: Median first-request-to-final-grant wait, seconds.
        wait_p95: 95th percentile wait, seconds.
        wait_p99: 99th percentile wait, seconds.
        wait_max: Worst observed wait, seconds.
        state_fractions: Fraction of fleet-time in each robot state.
        holds: Per-class held and held-but-unused breakdown.
        force_admissions: Times the admission policy was bypassed.
        deferrals: Total deferral events.
        deferral_histogram: Count of requests by deferral count reached.
        lease_revocations: Leases that expired before arrival.
        revocation_rate: Revocations per grant.
        unsafe_refusals: Deadlocks the safety checker prevented.
    """

    policy: str
    seed: int
    duration: float
    comparable: bool
    tasks_created: int
    tasks_completed: int
    throughput_per_hour: float
    wait_p50: float
    wait_p95: float
    wait_p99: float
    wait_max: float
    state_fractions: dict[str, float]
    holds: tuple[HoldBreakdown, ...]
    force_admissions: int
    deferrals: int
    deferral_histogram: dict[int, int] = field(default_factory=dict)
    lease_revocations: int = 0
    revocation_rate: float = 0.0
    unsafe_refusals: int = 0

    @property
    def total_unused_fraction(self) -> float:
        """Return early-binding waste aggregated over every class.

        Returns:
            Total unused instance-seconds over total capacity-seconds.
        """
        capacity = sum(hold.capacity_seconds for hold in self.holds)
        if not capacity:
            return 0.0
        return sum(hold.unused_seconds for hold in self.holds) / capacity

    @property
    def total_held_fraction(self) -> float:
        """Return held time aggregated over every class.

        Returns:
            Total held instance-seconds over total capacity-seconds.
        """
        capacity = sum(hold.capacity_seconds for hold in self.holds)
        if not capacity:
            return 0.0
        return sum(hold.held_seconds for hold in self.holds) / capacity

    def to_row(self) -> dict[str, Any]:
        """Flatten to a single CSV row.

        Returns:
            A flat mapping of column name to value, with per-class hold
            columns expanded.
        """
        row: dict[str, Any] = {
            "policy": self.policy,
            "seed": self.seed,
            "duration_s": self.duration,
            "comparable": self.comparable,
            "tasks_created": self.tasks_created,
            "tasks_completed": self.tasks_completed,
            "throughput_per_hour": round(self.throughput_per_hour, 4),
            "wait_p50_s": round(self.wait_p50, 4),
            "wait_p95_s": round(self.wait_p95, 4),
            "wait_p99_s": round(self.wait_p99, 4),
            "wait_max_s": round(self.wait_max, 4),
            "held_fraction": round(self.total_held_fraction, 6),
            "held_but_unused_fraction": round(self.total_unused_fraction, 6),
            "force_admissions": self.force_admissions,
            "deferrals": self.deferrals,
            "lease_revocations": self.lease_revocations,
            "revocation_rate": round(self.revocation_rate, 6),
            "unsafe_refusals": self.unsafe_refusals,
        }
        for state, fraction in sorted(self.state_fractions.items()):
            row[f"state_{state}_fraction"] = round(fraction, 6)
        for hold in self.holds:
            row[f"held_{hold.resource_class}"] = round(hold.held_fraction, 6)
            row[f"unused_{hold.resource_class}"] = round(hold.unused_fraction, 6)
            row[f"waste_ratio_{hold.resource_class}"] = round(hold.waste_ratio, 6)
        return row


def tasks_completed_per_simulated_hour(
    collector: EventCollector, duration: float
) -> float:
    """Return the headline throughput metric.

    Args:
        collector: The run's event log.
        duration: Simulated seconds the run covered.

    Returns:
        Completed tasks scaled to a simulated hour.
    """
    if duration <= 0.0:
        return 0.0
    completed = collector.count_of(TaskCompleted)
    return completed * SECONDS_PER_HOUR / duration


def wait_percentiles(collector: EventCollector) -> tuple[float, float, float, float]:
    """Return p50, p95, p99, and max wait in seconds.

    Wait spans first request to final grant, inclusive of every deferral in
    between, so a policy cannot improve its wait figures by refusing robots.

    Args:
        collector: The run's event log.

    Returns:
        ``(p50, p95, p99, max)``. All zero if nothing completed.
    """
    waits = [
        event.bundle_complete_at - event.first_request_at
        for event in collector.of_type(TaskCompleted)
    ]
    if not waits:
        return (0.0, 0.0, 0.0, 0.0)
    array = np.array(waits, dtype=np.float64)
    p50, p95, p99 = np.percentile(array, [50, 95, 99])
    return (float(p50), float(p95), float(p99), float(array.max()))


def robot_state_fractions(
    collector: EventCollector, config: SimConfig
) -> dict[str, float]:
    """Return the fraction of fleet-time spent in each robot state.

    Reconstructed by replaying transition events per robot. Robots start
    ``IDLE`` at time zero and their final state runs to the end of the run.

    Args:
        collector: The run's event log.
        config: Supplies fleet size and duration.

    Returns:
        A mapping of state name to fraction of total fleet-time.
    """
    totals: dict[str, float] = {state.value: 0.0 for state in RobotState}
    transitions: dict[int, list[RobotStateChanged]] = {}
    for event in collector.of_type(RobotStateChanged):
        transitions.setdefault(event.robot_id, []).append(event)

    for robot_id in range(config.robot_count):
        current = RobotState.IDLE
        last = 0.0
        for event in transitions.get(robot_id, []):
            totals[current.value] += event.at - last
            current = event.current
            last = event.at
        totals[current.value] += config.duration - last

    fleet_seconds = config.robot_count * config.duration
    if fleet_seconds <= 0.0:
        return dict.fromkeys(totals, 0.0)
    return {name: value / fleet_seconds for name, value in totals.items()}


def resource_hold_breakdown(
    collector: EventCollector, config: SimConfig
) -> tuple[HoldBreakdown, ...]:
    """Return held versus held-but-unused time per resource class.

    **This is the central measurement of the project.** A resource is *held*
    from the moment a request is granted, but it is only *used* once the holder
    physically arrives. The interval between the two is time the resource is
    locked and doing nothing while other robots may be starving for it. That
    interval is early binding, and the gap between the two series below is its
    direct measurement.

    Intervals are closed by release or by lease revocation. Any interval still
    open when the run ends is truncated at the run boundary rather than
    dropped, so a policy cannot flatter itself by leaving resources held at the
    end.

    Args:
        collector: The run's event log.
        config: Supplies per-class instance counts and duration.

    Returns:
        One :class:`HoldBreakdown` per resource class, in class-name order.
    """
    counts = config.resource_counts_map
    held: dict[str, float] = dict.fromkeys(counts, 0.0)
    unused: dict[str, float] = dict.fromkeys(counts, 0.0)

    # (resource_id, robot_id) -> [grant time, arrival time or None]
    open_holds: dict[tuple[int, int], list[float | None]] = {}
    classes: dict[int, str] = {}

    for event in collector.events:
        if isinstance(event, RequestGranted):
            open_holds[(event.resource_id, event.robot_id)] = [event.at, None]
            classes[event.resource_id] = str(event.resource_class)
        elif isinstance(event, ResourceOccupied):
            record = open_holds.get((event.resource_id, event.robot_id))
            if record is not None and record[1] is None:
                record[1] = event.at
        elif isinstance(event, ResourceReleased | LeaseRevoked):
            key = (event.resource_id, event.robot_id)
            record = open_holds.pop(key, None)
            if record is None:
                continue
            _accumulate(record, event.at, classes[event.resource_id], held, unused)

    # Anything still held when the run ended is truncated at the boundary.
    for (resource_id, _robot_id), record in open_holds.items():
        _accumulate(record, config.duration, classes[resource_id], held, unused)

    return tuple(
        HoldBreakdown(
            resource_class=name,
            instances=counts[name],
            held_seconds=held[name],
            unused_seconds=unused[name],
            capacity_seconds=counts[name] * config.duration,
        )
        for name in sorted(counts)
    )


def _accumulate(
    record: list[float | None],
    closed_at: float,
    class_name: str,
    held: dict[str, float],
    unused: dict[str, float],
) -> None:
    """Fold one closed hold interval into the running totals.

    Args:
        record: ``[granted_at, arrived_at or None]``.
        closed_at: When the hold ended.
        class_name: The resource's class.
        held: Accumulator for held seconds, mutated in place.
        unused: Accumulator for unused seconds, mutated in place.
    """
    granted_at = record[0]
    arrived_at = record[1]
    assert granted_at is not None
    span = max(0.0, closed_at - granted_at)
    held[class_name] += span
    # Never arrived: the entire hold was waste. Arrived: waste ends there.
    end_of_waste = closed_at if arrived_at is None else min(arrived_at, closed_at)
    unused[class_name] += max(0.0, end_of_waste - granted_at)


def state_timeline(
    collector: EventCollector, config: SimConfig, bins: int = 120
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the fleet's state mix sampled over simulated time.

    Sampled at bin centres rather than integrated per bin: the result feeds a
    stacked area chart, where a sampled mix and an integrated one are visually
    indistinguishable and the sampled form is far simpler to get right.

    Args:
        collector: The run's event log.
        config: Supplies fleet size and duration.
        bins: Number of sample points across the run.

    Returns:
        ``(times, series)`` where ``times`` are the sample points in seconds
        and ``series`` maps each state name to the fraction of the fleet in
        that state at each sample point. The series sum to one at every point.
    """
    import bisect

    times = np.linspace(0.0, config.duration, bins)
    tracks: dict[int, tuple[list[float], list[str]]] = {}
    for robot_id in range(config.robot_count):
        tracks[robot_id] = ([0.0], [RobotState.IDLE.value])
    for event in collector.of_type(RobotStateChanged):
        track = tracks.get(event.robot_id)
        if track is not None:
            track[0].append(event.at)
            track[1].append(event.current.value)

    series = {state.value: np.zeros(bins, dtype=np.float64) for state in RobotState}
    for stamps, states in tracks.values():
        for index, sample in enumerate(times):
            position = bisect.bisect_right(stamps, float(sample)) - 1
            series[states[max(position, 0)]][index] += 1.0

    divisor = float(config.robot_count) if config.robot_count else 1.0
    return times, {name: values / divisor for name, values in series.items()}


def deferral_histogram(collector: EventCollector) -> dict[int, int]:
    """Return how many requests reached each deferral count.

    Args:
        collector: The run's event log.

    Returns:
        A mapping of deferral count to the number of requests that reached it.
    """
    histogram: dict[int, int] = {}
    for event in collector.of_type(RequestDeferred):
        histogram[event.deferral_count] = histogram.get(event.deferral_count, 0) + 1
    return dict(sorted(histogram.items()))


def analyse(collector: EventCollector, config: SimConfig) -> RunAnalysis:
    """Derive every statistic for one run.

    Args:
        collector: The run's event log.
        config: The configuration that produced it.

    Returns:
        The complete :class:`RunAnalysis`.
    """
    from warehouse_allocator.metrics.events import TaskCreated

    p50, p95, p99, worst = wait_percentiles(collector)
    grants = collector.count_of(RequestGranted)
    revocations = collector.count_of(LeaseRevoked)

    return RunAnalysis(
        policy=config.policy,
        seed=config.seed,
        duration=config.duration,
        comparable=collector.comparable,
        tasks_created=collector.count_of(TaskCreated),
        tasks_completed=collector.count_of(TaskCompleted),
        throughput_per_hour=tasks_completed_per_simulated_hour(
            collector, config.duration
        ),
        wait_p50=p50,
        wait_p95=p95,
        wait_p99=p99,
        wait_max=worst,
        state_fractions=robot_state_fractions(collector, config),
        holds=resource_hold_breakdown(collector, config),
        force_admissions=collector.count_of(RequestForceAdmitted),
        deferrals=collector.count_of(RequestDeferred),
        deferral_histogram=deferral_histogram(collector),
        lease_revocations=revocations,
        revocation_rate=revocations / grants if grants else 0.0,
        unsafe_refusals=collector.count_of(RequestRefusedUnsafe),
    )
