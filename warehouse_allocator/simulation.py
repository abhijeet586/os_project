"""Wires the layers into a runnable simulation.

This module owns the *order* of operations within a timestep. That order is
part of the determinism contract: changing it changes results, so it is fixed
here in one place and documented rather than being an emergent property of
however the pieces happened to be called.

Per tick, in order:

1. Admit any tasks whose arrival time has been reached.
2. Expire overdue leases, so a resource freed this tick is available to the
   requests made this tick rather than a tick later.
3. Advance every robot's task lifecycle, in ascending robot id.
4. Advance kinematics.
5. Advance the clock.

The clock advances last so that every event recorded during a tick carries the
time at which that tick *began*.
"""

from __future__ import annotations

from typing import Final

from warehouse_allocator.alloc.allocator import Allocator
from warehouse_allocator.alloc.policies import AdmissionPolicy, build_policy
from warehouse_allocator.alloc.state import ResourceState
from warehouse_allocator.config import SimConfig
from warehouse_allocator.core.clock import SimClock
from warehouse_allocator.core.entities import ResourceClass, Robot, RobotState
from warehouse_allocator.core.lifecycle import LifecycleManager
from warehouse_allocator.core.rng import RNGProvider
from warehouse_allocator.core.task_generator import TaskTrace, generate_trace
from warehouse_allocator.core.tasks import Task
from warehouse_allocator.core.world import World
from warehouse_allocator.metrics.collector import EventCollector
from warehouse_allocator.metrics.events import (
    RobotStateChanged,
    TaskCompleted,
    TaskCreated,
)

__all__ = ["Simulation", "build_trace"]


def build_trace(config: SimConfig) -> TaskTrace:
    """Generate the workload for a configuration.

    The trace depends only on the layout and the ``task_gen`` stream, never on
    the policy, so the same configuration yields the identical workload for
    every policy under comparison.

    Args:
        config: Simulation parameters.

    Returns:
        The generated trace.
    """
    rng = RNGProvider(config.seed)
    world = World(config, rng)
    free_cells = tuple(
        (x, y)
        for y in range(world.grid.height)
        for x in range(world.grid.width)
        if world.grid.is_free((x, y))
    )
    return generate_trace(config, rng, free_cells)


class _MetricsObserver:
    """Translates lifecycle callbacks into recorded events."""

    __slots__ = ("_collector",)

    def __init__(self, collector: EventCollector) -> None:
        """Initialize the observer.

        Args:
            collector: Sink to record into.
        """
        self._collector = collector

    def task_started(self, robot: Robot, task: Task, when: float) -> None:
        """Record nothing; assignment is implied by the first request.

        Args:
            robot: The accepting robot.
            task: The accepted task.
            when: Simulated time.
        """

    def task_completed(self, robot: Robot, task: Task, when: float) -> None:
        """Record a completed task with its wait-defining timestamps.

        Args:
            robot: The completing robot.
            task: The completed task.
            when: Simulated time of completion.
        """
        self._collector.record(
            TaskCompleted(
                at=when,
                task_id=task.task_id,
                robot_id=robot.robot_id,
                template=task.template,
                created_at=task.created_at,
                first_request_at=task.timestamps.get("first_request", when),
                bundle_complete_at=task.timestamps.get("bundle_complete", when),
            )
        )

    def state_changed(self, robot: Robot, previous: RobotState, when: float) -> None:
        """Record a robot state transition.

        Args:
            robot: The transitioning robot.
            previous: The state left.
            when: Simulated time.
        """
        self._collector.record(
            RobotStateChanged(
                at=when,
                robot_id=robot.robot_id,
                previous=previous,
                current=robot.state,
            )
        )


class Simulation:
    """One complete headless simulation run.

    Attributes:
        config: The configuration being run.
        world: The world state.
        clock: The simulation clock.
        allocator: The two-layer allocator.
        lifecycle: The robot task state machine.
        collector: The append-only event log.
        trace: The replayed workload.
    """

    __slots__ = (
        "_next_task",
        "_pending_tasks",
        "allocator",
        "clock",
        "collector",
        "config",
        "lifecycle",
        "policy",
        "rng",
        "state",
        "trace",
        "world",
    )

    def __init__(
        self,
        config: SimConfig,
        trace: TaskTrace | None = None,
        policy: AdmissionPolicy | None = None,
        collector: EventCollector | None = None,
    ) -> None:
        """Build a simulation.

        Args:
            config: Simulation parameters.
            trace: Workload to replay. Generated from the config if omitted;
                pass one explicitly when comparing policies so every policy
                sees the identical workload.
            policy: Admission policy. Built from ``config.policy`` if omitted.
            collector: Event sink. A fresh one is created if omitted.
        """
        self.config: Final[SimConfig] = config
        self.rng: Final[RNGProvider] = RNGProvider(config.seed)
        self.world: Final[World] = World(config, self.rng, wander_when_idle=False)
        self.clock: Final[SimClock] = SimClock(config.dt)
        self.collector: Final[EventCollector] = (
            collector if collector is not None else EventCollector()
        )
        self.policy: Final[AdmissionPolicy] = (
            policy
            if policy is not None
            else build_policy(
                config.policy,
                radius=config.radius,
                eta_threshold=config.eta_threshold,
                lease_duration=config.lease_duration,
            )
        )
        self.trace: Final[TaskTrace] = (
            trace if trace is not None else build_trace(config)
        )

        totals: dict[ResourceClass, int] = {}
        for resource in self.world.resources:
            totals[resource.resource_class] = (
                totals.get(resource.resource_class, 0) + resource.capacity
            )
        self.state: Final[ResourceState] = ResourceState(
            robot_ids=tuple(robot.robot_id for robot in self.world.robots),
            totals=totals,
        )
        self.allocator: Final[Allocator] = Allocator(
            world=self.world,
            state=self.state,
            policy=self.policy,
            clock=self.clock,
            collector=self.collector,
            max_deferrals=config.max_deferrals,
        )
        self.lifecycle: Final[LifecycleManager] = LifecycleManager(
            world=self.world,
            allocator=self.allocator,
            observer=_MetricsObserver(self.collector),
        )

        self._pending_tasks: list[Task] = self.trace.materialize()
        self._next_task = 0

    def step(self) -> None:
        """Advance the simulation by exactly one fixed timestep."""
        now = self.clock.now()
        self._admit_arrivals(now)
        self.allocator.tick()
        self.lifecycle.update(now, self.config.dt)
        self.world.step(self.config.dt)
        self.clock.tick()

    def run(self, until: float | None = None) -> EventCollector:
        """Run to completion.

        Args:
            until: Simulated seconds to run for. Defaults to the config's
                ``duration``.

        Returns:
            The event log, for analysis.
        """
        horizon = until if until is not None else self.config.duration
        steps = round(horizon / self.config.dt)
        for _ in range(steps):
            self.step()
        return self.collector

    def rescale_future_arrivals(self, rate: float) -> None:
        """Resample the not-yet-arrived tail of the trace at a new rate.

        Only reachable from the interactive controls, which tamper-mark the run
        before calling this. Tasks that have already been queued are untouched;
        only future arrival times change.

        Args:
            rate: New arrival rate in tasks per simulated second.
        """
        if rate <= 0.0:
            raise ValueError("arrival rate must be positive")
        stream = self.rng.stream("noise")
        now = self.clock.now()
        when = now
        for index in range(self._next_task, len(self._pending_tasks)):
            when += float(stream.exponential(1.0 / rate))
            self._pending_tasks[index].created_at = when

    def state_hash(self) -> str:
        """Return a stable hash of all mutable simulation state.

        Returns:
            The world's state hash, used by determinism and parity tests.
        """
        return self.world.state_hash()

    def _admit_arrivals(self, now: float) -> None:
        """Queue every task whose arrival time has been reached.

        Args:
            now: Current simulated time.
        """
        while (
            self._next_task < len(self._pending_tasks)
            and self._pending_tasks[self._next_task].created_at <= now
        ):
            task = self._pending_tasks[self._next_task]
            self._next_task += 1
            self.lifecycle.submit(task)
            self.collector.record(
                TaskCreated(at=now, task_id=task.task_id, template=task.template)
            )
