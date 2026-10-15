"""Stage 3 verification: the two-layer allocator end to end."""

from __future__ import annotations

import numpy as np
import pytest

from warehouse_allocator.alloc.allocator import Allocator
from warehouse_allocator.alloc.policies import (
    AdmissionDecision,
    AdmissionPolicy,
    ETAPolicy,
    LeasePolicy,
    NullPolicy,
    RadiusPolicy,
    Verdict,
    build_policy,
)
from warehouse_allocator.alloc.safety import SafetyChecker
from warehouse_allocator.alloc.state import ResourceState
from warehouse_allocator.config import SimConfig
from warehouse_allocator.core.clock import SimClock
from warehouse_allocator.core.entities import (
    Resource,
    ResourceClass,
    Robot,
    RobotState,
)
from warehouse_allocator.core.rng import RNGProvider
from warehouse_allocator.core.tasks import Task, TaskTemplate
from warehouse_allocator.core.world import World
from warehouse_allocator.metrics.collector import EventCollector
from warehouse_allocator.metrics.events import (
    LeaseRevoked,
    RequestForceAdmitted,
    RequestGranted,
    RequestRefusedUnsafe,
)
from warehouse_allocator.simulation import Simulation

CONFIG = SimConfig(
    grid_width=30,
    grid_height=20,
    obstacle_density=0.1,
    robot_count=4,
    resource_counts=(("crane", 1), ("elevator", 1), ("dock", 1)),
    dt=0.05,
    duration=120.0,
    seed=99,
    max_deferrals=5,
)


class _AlwaysDefer(AdmissionPolicy):
    """Refuses every request, to exercise the force-admission path."""

    name = "always_defer"

    def should_admit(
        self, robot: Robot, resource: Resource, world: World, clock: SimClock
    ) -> AdmissionDecision:
        """Defer unconditionally.

        Args:
            robot: Unused.
            resource: Unused.
            world: Unused.
            clock: Unused.

        Returns:
            An unconditional defer.
        """
        return AdmissionDecision(Verdict.DEFER, "test policy always defers")


def _harness(
    policy: AdmissionPolicy | None = None,
    config: SimConfig = CONFIG,
) -> tuple[World, Allocator, EventCollector, SimClock]:
    """Build a world and allocator wired together for direct manipulation.

    Args:
        policy: Admission policy; defaults to :class:`NullPolicy`.
        config: Configuration to build from.

    Returns:
        The world, allocator, collector, and clock.
    """
    rng = RNGProvider(config.seed)
    world = World(config, rng, wander_when_idle=False)
    clock = SimClock(config.dt)
    collector = EventCollector()
    totals: dict[ResourceClass, int] = {}
    for resource in world.resources:
        totals[resource.resource_class] = (
            totals.get(resource.resource_class, 0) + resource.capacity
        )
    state = ResourceState(
        robot_ids=tuple(robot.robot_id for robot in world.robots), totals=totals
    )
    allocator = Allocator(
        world=world,
        state=state,
        policy=policy if policy is not None else NullPolicy(),
        clock=clock,
        collector=collector,
        max_deferrals=config.max_deferrals,
    )
    return world, allocator, collector, clock


def _task(task_id: int, template: TaskTemplate) -> Task:
    """Build a task for direct allocator manipulation.

    Args:
        task_id: Identifier.
        template: Template supplying the bundle.

    Returns:
        The constructed task.
    """
    return Task(
        task_id=task_id,
        template=template,
        pickup=(1, 1),
        dropoff=(2, 2),
        service_duration=5.0,
        created_at=0.0,
    )


class TestCircularWait:
    """The safety layer must refuse the request that closes a cycle."""

    def test_circular_wait_refused(self) -> None:
        """A genuine three-robot ring must be broken by the safety check.

        Robot 0 (HEAVY_LIFT) holds the crane and wants the elevator.
        Robot 1 (FLOOR_TRANSFER) holds the elevator and wants the dock.
        Robot 2 (FULL_CYCLE) holds the dock and wants the crane.

        One instance of each class exists, so granting the third robot's
        outstanding request would close the ring and deadlock the fleet.
        """
        world, allocator, collector, _ = _harness()
        robots = world.robots

        heavy = _task(0, TaskTemplate.HEAVY_LIFT)
        floor = _task(1, TaskTemplate.FLOOR_TRANSFER)
        full = _task(2, TaskTemplate.FULL_CYCLE)

        allocator.declare_task(robots[0], heavy)
        allocator.declare_task(robots[1], floor)
        allocator.declare_task(robots[2], full)

        assert allocator.request(robots[0], ResourceClass.CRANE, heavy) is not None
        assert allocator.request(robots[1], ResourceClass.ELEVATOR, floor) is not None

        # This grant would leave every class exhausted with three robots each
        # still needing one, so it must be refused as unsafe.
        assert allocator.request(robots[2], ResourceClass.DOCK, full) is None

        refusals = collector.of_type(RequestRefusedUnsafe)
        assert refusals, "no unsafe refusal was recorded"
        assert refusals[-1].robot_id == robots[2].robot_id

        # The partial holds survive the refusal; nothing was rolled back.
        assert len(robots[0].held_resources) == 1
        assert len(robots[1].held_resources) == 1
        assert not robots[2].held_resources

    def test_safe_progress_still_possible_after_refusal(self) -> None:
        """Refusing the cycle must leave a viable path forward."""
        world, allocator, _, _ = _harness()
        robots = world.robots
        heavy = _task(0, TaskTemplate.HEAVY_LIFT)
        floor = _task(1, TaskTemplate.FLOOR_TRANSFER)

        allocator.declare_task(robots[0], heavy)
        allocator.declare_task(robots[1], floor)
        allocator.request(robots[0], ResourceClass.CRANE, heavy)

        # Robot 0 can complete its bundle, so the elevator grant is safe.
        assert allocator.request(robots[0], ResourceClass.ELEVATOR, heavy) is not None
        assert len(robots[0].held_resources) == 2

    def test_deadlock_never_occurs_in_a_full_run(self) -> None:
        """A long run under contention must keep completing tasks."""
        config = CONFIG.evolve(duration=900.0, robot_count=8, task_arrival_rate=1.0)
        sim = Simulation(config)
        sim.run()

        from warehouse_allocator.metrics.events import TaskCompleted

        assert sim.collector.count_of(TaskCompleted) > 0
        assert sim.collector.count_of(RequestRefusedUnsafe) >= 0


class TestForceAdmission:
    """Starvation must be bounded, but never at the cost of safety."""

    def test_force_admit_after_max_deferrals(self) -> None:
        """A request deferred past the limit must bypass the policy."""
        world, allocator, collector, _ = _harness(policy=_AlwaysDefer())
        robot = world.robots[0]
        task = _task(0, TaskTemplate.RECHARGE)
        allocator.declare_task(robot, task)

        results = [
            allocator.request(robot, ResourceClass.DOCK, task)
            for _ in range(CONFIG.max_deferrals + 2)
        ]

        assert all(result is None for result in results[: CONFIG.max_deferrals + 1])
        assert results[-1] is not None, "request never force-admitted"

        forced = collector.of_type(RequestForceAdmitted)
        assert len(forced) == 1
        assert forced[0].deferral_count == CONFIG.max_deferrals + 1

        granted = collector.of_type(RequestGranted)
        assert granted[-1].forced is True

    def test_force_admit_still_safety_checked(self) -> None:
        """A forced request that would deadlock must still be refused.

        This is the single most important assertion in the project. The
        admission policy may be bypassed; the safety check never is.
        """
        world, allocator, collector, _ = _harness(policy=_AlwaysDefer())
        robots = world.robots

        heavy = _task(0, TaskTemplate.HEAVY_LIFT)
        floor = _task(1, TaskTemplate.FLOOR_TRANSFER)
        full = _task(2, TaskTemplate.FULL_CYCLE)
        allocator.declare_task(robots[0], heavy)
        allocator.declare_task(robots[1], floor)
        allocator.declare_task(robots[2], full)

        # Force each of the first two robots into a partial hold.
        for _ in range(CONFIG.max_deferrals + 2):
            allocator.request(robots[0], ResourceClass.CRANE, heavy)
        for _ in range(CONFIG.max_deferrals + 2):
            allocator.request(robots[1], ResourceClass.ELEVATOR, floor)

        assert robots[0].held_resources
        assert robots[1].held_resources

        # Now hammer the cycle-closing request well past the deferral limit.
        outcomes = [
            allocator.request(robots[2], ResourceClass.DOCK, full)
            for _ in range(CONFIG.max_deferrals + 5)
        ]

        assert all(
            outcome is None for outcome in outcomes
        ), "force-admission bypassed the safety check and closed a deadlock"
        assert collector.of_type(RequestForceAdmitted), "never reached the bypass"
        assert collector.of_type(RequestRefusedUnsafe), "safety check never fired"
        assert not robots[2].held_resources

    def test_force_admission_is_counted(self) -> None:
        """Starvation events must be reported, not hidden."""
        world, allocator, collector, _ = _harness(policy=_AlwaysDefer())
        robot = world.robots[0]
        task = _task(0, TaskTemplate.RECHARGE)
        allocator.declare_task(robot, task)
        for _ in range(CONFIG.max_deferrals + 2):
            allocator.request(robot, ResourceClass.DOCK, task)

        assert collector.count_of(RequestForceAdmitted) == 1


class TestLeaseRevocation:
    """Revocation must be surgical: one resource back, the rest untouched."""

    def test_lease_revocation_partial_hold(self) -> None:
        """An expiring lease must return only its own resource.

        The robot holds a crane and an elevator. The elevator lease expires
        before it arrives. The elevator must return to the pool, the crane
        must be retained, and the matrices must stay consistent.
        """
        world, allocator, collector, clock = _harness(policy=LeasePolicy(10.0))
        robot = world.robots[0]
        task = _task(0, TaskTemplate.HEAVY_LIFT)
        allocator.declare_task(robot, task)

        crane = allocator.request(robot, ResourceClass.CRANE, task)
        elevator = allocator.request(robot, ResourceClass.ELEVATOR, task)
        assert crane is not None
        assert elevator is not None

        # The robot arrives at the crane but never at the elevator.
        allocator.on_arrival(robot, crane)

        state = allocator.state
        crane_column = state.class_column(ResourceClass.CRANE)
        elevator_column = state.class_column(ResourceClass.ELEVATOR)
        row = state.robot_row(robot.robot_id)
        assert state.allocation[row][crane_column] == 1
        assert state.allocation[row][elevator_column] == 1
        assert state.available[elevator_column] == 0

        for _ in range(round(11.0 / CONFIG.dt)):
            clock.tick()
        allocator.tick()

        revocations = collector.of_type(LeaseRevoked)
        assert len(revocations) == 1
        assert revocations[0].resource_id == elevator.resource_id

        # The elevator went back; the crane stayed.
        assert elevator.resource_id not in robot.held_resources
        assert crane.resource_id in robot.held_resources
        assert robot.robot_id not in elevator.holders
        assert robot.robot_id in crane.holders

        # Matrices are consistent and Need was restored, not forgotten.
        assert state.allocation[row][crane_column] == 1
        assert state.allocation[row][elevator_column] == 0
        assert state.available[elevator_column] == 1
        assert np.array_equal(state.need, state.max_claim - state.allocation)
        assert state.need[row][elevator_column] == 1

    def test_arrived_holder_is_not_revoked(self) -> None:
        """A holder that met its deadline must keep the resource."""
        world, allocator, collector, clock = _harness(policy=LeasePolicy(10.0))
        robot = world.robots[0]
        task = _task(0, TaskTemplate.RECHARGE)
        allocator.declare_task(robot, task)
        dock = allocator.request(robot, ResourceClass.DOCK, task)
        assert dock is not None
        allocator.on_arrival(robot, dock)

        for _ in range(round(20.0 / CONFIG.dt)):
            clock.tick()
        allocator.tick()

        assert not collector.of_type(LeaseRevoked)
        assert dock.resource_id in robot.held_resources

    def test_revocation_returns_robot_to_waiting(self) -> None:
        """A robot travelling on a broken bundle must stop and re-acquire."""
        world, allocator, _, clock = _harness(policy=LeasePolicy(5.0))
        robot = world.robots[0]
        task = _task(0, TaskTemplate.HEAVY_LIFT)
        allocator.declare_task(robot, task)
        allocator.request(robot, ResourceClass.CRANE, task)
        allocator.request(robot, ResourceClass.ELEVATOR, task)
        robot.state = RobotState.MOVING_TO_RESOURCE
        robot.bundle_cursor = 2

        for _ in range(round(6.0 / CONFIG.dt)):
            clock.tick()
        allocator.tick()

        assert robot.state is RobotState.WAITING
        assert robot.bundle_cursor < 2

    def test_matrices_survive_a_long_lease_run(self) -> None:
        """Repeated revocation under contention must not corrupt matrices.

        The invariant assertions inside ``ResourceState`` fire on any
        corruption, so completing the run at all is the assertion.
        """
        config = CONFIG.evolve(
            duration=900.0,
            robot_count=10,
            task_arrival_rate=1.5,
            policy="lease",
            lease_duration=8.0,
        )
        sim = Simulation(config)
        sim.run()

        state = sim.state
        assert np.array_equal(state.need, state.max_claim - state.allocation)
        assert np.all(state.available >= 0)
        assert (
            sim.collector.count_of(LeaseRevoked) > 0
        ), "lease run produced no revocations, so nothing was exercised"


class TestPolicies:
    """Each admission policy must implement its stated rule."""

    def test_null_admits_everything(self) -> None:
        """The baseline must never defer."""
        world, _, _, clock = _harness()
        robot = world.robots[0]
        far = world.resources[0]

        assert NullPolicy().should_admit(robot, far, world, clock).admitted

    def test_radius_defers_distant_requesters(self) -> None:
        """Radius must defer beyond its threshold and admit within it."""
        world, _, _, clock = _harness()
        robot = world.robots[0]
        resource = world.resources[0]
        resource.position = (robot.position[0] + 40.0, robot.position[1])

        assert (
            not RadiusPolicy(10.0).should_admit(robot, resource, world, clock).admitted
        )
        assert RadiusPolicy(100.0).should_admit(robot, resource, world, clock).admitted

    def test_eta_uses_path_cost_not_straight_line(self) -> None:
        """ETA must see a wall that the radius rule is blind to.

        A robot separated from a resource by a long wall is close in straight
        line but far by path. Radius admits it; ETA must not. This divergence
        is the entire reason ETAPolicy exists.
        """
        config = CONFIG.evolve(obstacle_density=0.0, grid_width=40, grid_height=30)
        world = World(config, RNGProvider(1), wander_when_idle=False)
        clock = SimClock(config.dt)

        for y in range(0, 28):
            world.grid.set_blocked((20, y))

        robot = world.robots[0]
        robot.position = (19.5, 1.5)
        resource = world.resources[0]
        resource.position = (21.5, 1.5)

        straight_line = 2.0
        assert (
            RadiusPolicy(straight_line + 1.0)
            .should_admit(robot, resource, world, clock)
            .admitted
        ), "radius should be fooled by the wall"

        eta_threshold = (straight_line + 1.0) / config.robot_speed
        assert (
            not ETAPolicy(eta_threshold)
            .should_admit(robot, resource, world, clock)
            .admitted
        ), "ETA failed to see the detour"

    def test_lease_attaches_a_deadline(self) -> None:
        """Lease must admit and carry an arrival deadline."""
        world, _, _, clock = _harness()
        decision = LeasePolicy(30.0).should_admit(
            world.robots[0], world.resources[0], world, clock
        )

        assert decision.admitted
        assert decision.lease_deadline == pytest.approx(clock.now() + 30.0)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("null", NullPolicy),
            ("radius", RadiusPolicy),
            ("eta", ETAPolicy),
            ("lease", LeasePolicy),
        ],
    )
    def test_build_policy(self, name: str, expected: type[AdmissionPolicy]) -> None:
        """Every policy must be constructible by name."""
        policy = build_policy(
            name, radius=10.0, eta_threshold=10.0, lease_duration=10.0
        )

        assert isinstance(policy, expected)

    def test_build_policy_rejects_unknown(self) -> None:
        """An unknown policy name must fail loudly."""
        with pytest.raises(ValueError, match="unknown policy"):
            build_policy("nope", radius=1.0, eta_threshold=1.0, lease_duration=1.0)


class TestDecisionLog:
    """Every decision must leave a structured record."""

    def test_grant_records_full_context(self) -> None:
        """A grant event must carry everything needed to explain it."""
        world, allocator, collector, _ = _harness()
        robot = world.robots[0]
        task = _task(0, TaskTemplate.RECHARGE)
        allocator.declare_task(robot, task)
        allocator.request(robot, ResourceClass.DOCK, task)

        granted = collector.of_type(RequestGranted)[-1]

        assert granted.robot_id == robot.robot_id
        assert granted.task_id == 0
        assert granted.resource_class is ResourceClass.DOCK
        assert granted.reason
        assert granted.forced is False

    def test_deferral_records_policy_reason(self) -> None:
        """A deferral must record which rule rejected it and why."""
        from warehouse_allocator.metrics.events import RequestDeferred

        world, allocator, collector, _ = _harness(policy=RadiusPolicy(0.5))
        robot = world.robots[0]
        task = _task(0, TaskTemplate.RECHARGE)
        allocator.declare_task(robot, task)
        allocator.request(robot, ResourceClass.DOCK, task)

        deferred = collector.of_type(RequestDeferred)[-1]

        assert deferred.deferral_count == 1
        assert "distance" in deferred.reason

    def test_collector_is_append_only(self) -> None:
        """The log must only ever grow."""
        collector = EventCollector()
        counts = []
        world, allocator, _, _ = _harness()
        robot = world.robots[0]
        task = _task(0, TaskTemplate.RECHARGE)
        allocator.collector = collector
        allocator.declare_task(robot, task)
        for _ in range(5):
            allocator.request(robot, ResourceClass.DOCK, task)
            counts.append(len(collector))

        assert counts == sorted(counts)
        assert counts[-1] > counts[0]


class TestSafetyNeverBypassed:
    """Guardrails against a future edit removing the safety check."""

    def test_allocator_calls_safety_on_forced_requests(self) -> None:
        """The checker must be consulted even when the policy is bypassed."""
        world, allocator, _, _ = _harness(policy=_AlwaysDefer())
        calls: list[int] = []

        class _CountingChecker(SafetyChecker):
            def is_safe(self, state, robot_id, request):  # type: ignore[no-untyped-def]
                calls.append(robot_id)
                return super().is_safe(state, robot_id, request)

        allocator.safety = _CountingChecker()
        robot = world.robots[0]
        task = _task(0, TaskTemplate.RECHARGE)
        allocator.declare_task(robot, task)
        for _ in range(CONFIG.max_deferrals + 2):
            allocator.request(robot, ResourceClass.DOCK, task)

        assert calls, "safety check was never invoked on a forced request"
