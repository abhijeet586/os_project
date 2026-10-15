"""Simulation configuration and run provenance.

Every tunable parameter in the simulation lives on :class:`SimConfig`. The
dataclass is frozen so that a configuration cannot drift mid-run: an
experiment that mutates its own parameters is not reproducible, and the whole
project rests on reproducibility.

Serialization helpers (:meth:`SimConfig.to_dict`, :meth:`SimConfig.to_json`,
:meth:`SimConfig.from_dict`) exist so that every emitted CSV and figure can
carry the exact configuration that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Final

__all__ = ["DEFAULT_RESOURCE_COUNTS", "SimConfig"]

# Resource counts are stored as a tuple of pairs rather than a dict because a
# frozen dataclass holding a dict is only shallowly immutable: callers could
# still mutate the dict in place and silently desynchronise a run from the
# config recorded alongside its results.
# Sized so that several complete bundles can be held at once. With too few
# instances every request is refused for lack of availability, the admission
# policy is never consulted at all, and all four policies produce byte-identical
# results -- measured, not assumed. Contention must come from the admission
# layer for the experiment to have a subject.
DEFAULT_RESOURCE_COUNTS: Final[tuple[tuple[str, int], ...]] = (
    ("crane", 6),
    ("elevator", 6),
    ("dock", 8),
)


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Immutable parameter set defining a single simulation run.

    Attributes:
        grid_width: Warehouse width in grid cells. One cell is one metre, so
            this is also the width in metres. Wide enough that travel time is
            a meaningful fraction of task time, which is what makes early
            binding measurable at all.
        grid_height: Warehouse height in cells (metres).
        obstacle_density: Fraction of cells to mark as impassable racking,
            in ``[0.0, 1.0)``. Obstacles are what make euclidean distance
            diverge from true path cost, which is the entire distinction
            between ``RadiusPolicy`` and ``ETAPolicy``.
        robot_count: Number of robots in the fleet. Contention scales with
            this relative to the total resource count.
        resource_counts: Per-class resource counts as ``(class_name, count)``
            pairs. Kept as a tuple to preserve true immutability.
        dt: Fixed simulation timestep in simulated seconds. Small enough that
            a robot never skips past a waypoint at ``robot_speed``.
        duration: Total simulated seconds to run. One hour by default so that
            the headline per-hour throughput metric needs no extrapolation.
        seed: Master seed. All stochastic draws derive from this via
            :class:`~warehouse_allocator.core.rng.RNGProvider`.
        policy: Admission policy name; one of ``null``, ``radius``, ``eta``,
            ``lease``. Consumed from Stage 3 onward.
        radius: ``RadiusPolicy`` admission threshold in metres. Euclidean
            distance beyond this defers the request.
        eta_threshold: ``ETAPolicy`` admission threshold in simulated seconds
            of A* travel time.
        lease_duration: ``LeasePolicy`` arrival deadline in simulated seconds.
            A holder that has not arrived by then is revoked.
        max_deferrals: Deferrals a request may accumulate before it is
            force-admitted past the admission policy. Bounds starvation. The
            safety check is never bypassed regardless of this value.
        task_arrival_rate: Poisson arrival rate in tasks per simulated second.
        service_duration_mean: Mean simulated seconds of work a task performs
            at its resources, drawn from a clamped exponential. Long enough
            that holding a resource is costly, short enough that travel time
            still matters.
        robot_speed: Robot travel speed in metres per simulated second.
        battery_drain_rate: Battery fraction consumed per simulated second
            while moving, in units of ``[0, 1]`` capacity per second.
    """

    grid_width: int = 100
    grid_height: int = 70
    obstacle_density: float = 0.12
    robot_count: int = 24
    resource_counts: tuple[tuple[str, int], ...] = field(
        default=DEFAULT_RESOURCE_COUNTS
    )
    dt: float = 0.05
    duration: float = 3600.0
    seed: int = 42
    policy: str = "null"
    radius: float = 40.0
    eta_threshold: float = 30.0
    lease_duration: float = 300.0
    max_deferrals: int = 5
    task_arrival_rate: float = 0.10
    service_duration_mean: float = 18.0
    robot_speed: float = 1.5
    battery_drain_rate: float = 0.0015

    def __post_init__(self) -> None:
        """Reject configurations that cannot produce a meaningful run.

        Raises:
            ValueError: If any parameter is outside its physically or
                statistically sensible domain.
        """
        if self.grid_width <= 0 or self.grid_height <= 0:
            raise ValueError("grid dimensions must be positive")
        if not 0.0 <= self.obstacle_density < 1.0:
            raise ValueError("obstacle_density must lie in [0.0, 1.0)")
        if self.robot_count <= 0:
            raise ValueError("robot_count must be positive")
        if any(count < 0 for _, count in self.resource_counts):
            raise ValueError("resource counts must be non-negative")
        if len({name for name, _ in self.resource_counts}) != len(self.resource_counts):
            raise ValueError("duplicate resource class in resource_counts")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if self.max_deferrals < 0:
            raise ValueError("max_deferrals must be non-negative")
        if self.task_arrival_rate <= 0.0:
            raise ValueError("task_arrival_rate must be positive")
        if self.service_duration_mean <= 0.0:
            raise ValueError("service_duration_mean must be positive")
        if self.robot_speed <= 0.0:
            raise ValueError("robot_speed must be positive")
        if self.battery_drain_rate < 0.0:
            raise ValueError("battery_drain_rate must be non-negative")
        # A robot that covers more than one cell per tick can tunnel through a
        # one-cell-thick obstacle wall, which would silently invalidate every
        # path-cost measurement the experiment depends on.
        if self.robot_speed * self.dt > 1.0:
            raise ValueError("robot_speed * dt must not exceed one cell per tick")

    @property
    def resource_counts_map(self) -> dict[str, int]:
        """Return resource counts as a fresh mutable mapping.

        Returns:
            A new ``dict`` of class name to count. A copy is returned each
            call so callers cannot mutate the configuration through it.
        """
        return dict(self.resource_counts)

    @property
    def total_steps(self) -> int:
        """Return the number of fixed timesteps in a full run.

        Returns:
            ``duration / dt`` rounded to the nearest integer.
        """
        return round(self.duration / self.dt)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration to plain JSON-compatible types.

        Returns:
            A dict whose ``resource_counts`` entry is a list of two-element
            lists, matching what :meth:`from_dict` accepts.
        """
        data: dict[str, Any] = asdict(self)
        data["resource_counts"] = [
            [name, count] for name, count in self.resource_counts
        ]
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the configuration to a JSON string for run provenance.

        Args:
            indent: Indentation passed to :func:`json.dumps`. Pass ``None``
                for a single-line form suitable for a CSV header comment.

        Returns:
            The JSON representation, with keys sorted for stable diffing.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimConfig:
        """Reconstruct a configuration from :meth:`to_dict` output.

        Args:
            data: Mapping of field name to value. Unknown keys are rejected
                rather than ignored, so a stale provenance record fails loudly
                instead of silently reverting fields to their defaults.

        Returns:
            The reconstructed :class:`SimConfig`.

        Raises:
            ValueError: If ``data`` contains keys that are not config fields.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown SimConfig fields: {sorted(unknown)}")
        payload = dict(data)
        if "resource_counts" in payload:
            payload["resource_counts"] = tuple(
                (str(name), int(count)) for name, count in payload["resource_counts"]
            )
        return cls(**payload)

    @classmethod
    def from_json(cls, text: str) -> SimConfig:
        """Reconstruct a configuration from a JSON string.

        Args:
            text: JSON produced by :meth:`to_json`.

        Returns:
            The reconstructed :class:`SimConfig`.
        """
        parsed: dict[str, Any] = json.loads(text)
        return cls.from_dict(parsed)

    def evolve(self, **changes: Any) -> SimConfig:
        """Return a copy with selected fields replaced.

        Used by sweeps and multi-seed runners, which need many configs that
        differ in one field.

        Args:
            **changes: Field names mapped to their new values.

        Returns:
            A new validated :class:`SimConfig`.
        """
        return replace(self, **changes)
