"""Stage 2 verification: tasks, bundles, trace replay, and partial holds."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from warehouse_allocator.config import SimConfig
from warehouse_allocator.core.entities import (
    Resource,
    ResourceClass,
    Robot,
    RobotState,
)
from warehouse_allocator.core.lifecycle import LifecycleManager
from warehouse_allocator.core.rng import RNGProvider
from warehouse_allocator.core.task_generator import TaskTrace, generate_trace
from warehouse_allocator.core.tasks import (
    TEMPLATE_ACQUISITION_ORDER,
    Task,
    TaskTemplate,
    bundle_for,
)
from warehouse_allocator.core.world import World

TEST_CONFIG = SimConfig(
    grid_width=40,
    grid_height=30,
    obstacle_density=0.15,
    robot_count=6,
    dt=0.05,
    duration=600.0,
    seed=1234,
    task_arrival_rate=2.0,
)


def _free_cells(config: SimConfig = TEST_CONFIG) -> tuple[tuple[int, int], ...]:
    """Build a world and return its passable cells.

    Args:
        config: Configuration to build the layout from.

    Returns:
        Every passable cell in row-major order.
    """
    world = World(config, RNGProvider(config.seed))
    return tuple(
        (x, y)
        for y in range(world.grid.height)
        for x in range(world.grid.width)
        if world.grid.is_free((x, y))
    )


def _trace(seed: int = 1234, count: int = 1000) -> TaskTrace:
    """Generate a trace of at least ``count`` tasks.

    Args:
        seed: Master seed.
        count: Minimum number of tasks required.

    Returns:
        The generated trace, truncated to exactly ``count`` tasks.
    """
    config = TEST_CONFIG.evolve(seed=seed, duration=count / 1.5)
    trace = generate_trace(config, RNGProvider(seed), _free_cells(config))
    assert len(trace) >= count, f"trace produced only {len(trace)} tasks"
    return TaskTrace(
        seed=trace.seed,
        duration=trace.duration,
        arrival_rate=trace.arrival_rate,
        tasks=trace.tasks[:count],
    )


class TestBundles:
    """Bundle templates must make deadlock constructible."""

    def test_bundles_overlap(self) -> None:
        """At least two templates must share a resource class."""
        shared: Counter[ResourceClass] = Counter()
        for order in TEMPLATE_ACQUISITION_ORDER.values():
            shared.update(set(order))

        overlapping = [cls for cls, n in shared.items() if n >= 2]
        assert overlapping, "no resource class is shared between templates"
        assert ResourceClass.ELEVATOR in overlapping

    def test_three_templates_are_multiclass(self) -> None:
        """At least three templates must need 2+ distinct classes."""
        multiclass = [
            template
            for template, order in TEMPLATE_ACQUISITION_ORDER.items()
            if len(set(order)) >= 2
        ]

        assert len(multiclass) >= 3

    def test_acquisition_orders_are_inconsistent(self) -> None:
        """No single global class ordering may explain every template.

        If one existed, ordered-resource deadlock *prevention* would hold and
        circular wait would be structurally impossible, leaving the Banker's
        safety check with nothing to do.
        """
        pairs: set[tuple[ResourceClass, ResourceClass]] = set()
        for order in TEMPLATE_ACQUISITION_ORDER.values():
            distinct = list(dict.fromkeys(order))
            for i, earlier in enumerate(distinct):
                for later in distinct[i + 1 :]:
                    pairs.add((earlier, later))

        contradictions = [(a, b) for a, b in pairs if (b, a) in pairs]
        assert contradictions, "acquisition orders admit a global ordering"

    def test_circular_wait_is_constructible(self) -> None:
        """Three templates must compose a genuine hold-and-wait cycle."""
        holds_wants = {
            template: (order[0], order[1])
            for template, order in TEMPLATE_ACQUISITION_ORDER.items()
            if len(order) >= 2
        }

        cycle = [
            (TaskTemplate.HEAVY_LIFT, ResourceClass.CRANE, ResourceClass.ELEVATOR),
            (
                TaskTemplate.FLOOR_TRANSFER,
                ResourceClass.ELEVATOR,
                ResourceClass.DOCK,
            ),
            (TaskTemplate.FULL_CYCLE, ResourceClass.DOCK, ResourceClass.CRANE),
        ]
        for template, holds, wants in cycle:
            assert holds_wants[template] == (holds, wants)

        # Each template wants what the next one holds, closing the ring.
        for index, (_, _, wants) in enumerate(cycle):
            following_holds = cycle[(index + 1) % len(cycle)][1]
            assert wants == following_holds or index == len(cycle) - 1

    @pytest.mark.parametrize("template", list(TaskTemplate))
    def test_bundle_counts_positive(self, template: TaskTemplate) -> None:
        """Every template must require at least one resource."""
        bundle = bundle_for(template)

        assert bundle
        assert all(count >= 1 for count in bundle.values())


class TestTrace:
    """Traces must be reproducible and replayable."""

    def test_trace_roundtrip(self, tmp_path: Path) -> None:
        """1000 tasks saved and reloaded must compare equal."""
        original = _trace(count=1000)
        target = tmp_path / "trace.json"
        original.save(target)
        loaded = TaskTrace.load(target)

        assert len(loaded) == 1000
        assert loaded == original

    def test_trace_deterministic(self) -> None:
        """The same seed must produce an identical trace."""
        first = _trace(seed=555, count=400)
        second = _trace(seed=555, count=400)

        assert first == second

    def test_trace_varies_with_seed(self) -> None:
        """A different seed must produce a different workload."""
        assert _trace(seed=555, count=400) != _trace(seed=556, count=400)

    def test_trace_is_time_ordered(self) -> None:
        """Arrivals must be non-decreasing in creation time."""
        tasks = _trace(count=500).tasks
        times = [task.created_at for task in tasks]

        assert times == sorted(times)

    def test_trace_respects_duration(self) -> None:
        """No task may arrive after the run ends."""
        config = TEST_CONFIG.evolve(duration=300.0)
        trace = generate_trace(config, RNGProvider(config.seed), _free_cells(config))

        assert all(task.created_at < 300.0 for task in trace.tasks)

    def test_materialize_gives_independent_copies(self) -> None:
        """Each policy's run must get its own task objects.

        Shared mutable tasks would let one policy's lifecycle marks leak into
        the next policy's run.
        """
        trace = _trace(count=50)
        first = trace.materialize()
        second = trace.materialize()

        first[0].mark("assigned", 12.5)

        assert first[0] is not second[0]
        assert second[0].timestamps == {}
        assert trace.tasks[0].timestamps == {}

    def test_materialize_preserves_workload(self) -> None:
        """Materializing must not perturb the workload itself."""
        trace = _trace(count=100)
        materialized = trace.materialize()

        for original, copy in zip(trace.tasks, materialized, strict=True):
            assert copy.task_id == original.task_id
            assert copy.template == original.template
            assert copy.pickup == original.pickup
            assert copy.dropoff == original.dropoff
            assert copy.service_duration == original.service_duration
            assert copy.created_at == original.created_at

    def test_service_durations_are_clamped(self) -> None:
        """No task may monopolise a scarce resource for an absurd span."""
        trace = _trace(count=800)
        mean = TEST_CONFIG.service_duration_mean

        for task in trace.tasks:
            assert 0.25 * mean <= task.service_duration <= 3.0 * mean

    def test_pickups_are_on_free_cells(self) -> None:
        """Pickups and dropoffs must be reachable."""
        cells = set(_free_cells())
        for task in _trace(count=300).tasks:
            assert task.pickup in cells
            assert task.dropoff in cells

    def test_all_templates_appear(self) -> None:
        """A long trace must exercise every template."""
        seen = {task.template for task in _trace(count=800).tasks}

        assert seen == set(TaskTemplate)


class _StubAllocator:
    """Grants one class and refuses another, to isolate partial-hold logic."""

    def __init__(self, world: World, grant: ResourceClass, refuse: ResourceClass):
        """Initialize the stub.

        Args:
            world: World supplying resource instances.
            grant: Class that will always be granted.
            refuse: Class that will always be refused.
        """
        self.world = world
        self.grant = grant
        self.refuse = refuse
        self.released: list[int] = []
        self.refusals = 0

    def declare_task(self, robot: Robot, task: Task) -> None:
        """Accept a task declaration; the stub tracks no matrices.

        Args:
            robot: The accepting robot.
            task: The accepted task.
        """

    def request(
        self, robot: Robot, resource_class: ResourceClass, task: Task
    ) -> Resource | None:
        """Grant or refuse according to the configured classes.

        Args:
            robot: The requesting robot.
            resource_class: The class requested.
            task: The task being served.

        Returns:
            A bound resource, or ``None`` when refused.
        """
        if resource_class == self.refuse:
            self.refusals += 1
            return None
        for resource in self.world.resources_of_class(resource_class):
            if not resource.is_full:
                resource.holders.add(robot.robot_id)
                robot.held_resources.add(resource.resource_id)
                return resource
        return None

    def release_all(self, robot: Robot) -> None:
        """Release everything the robot holds.

        Args:
            robot: The releasing robot.
        """
        for resource_id in sorted(robot.held_resources):
            resource = self.world.resource(resource_id)
            resource.holders.discard(robot.robot_id)
            resource.occupants.discard(robot.robot_id)
            self.released.append(resource_id)
        robot.held_resources.clear()

    def on_arrival(self, robot: Robot, resource: Resource) -> None:
        """Mark the robot as having arrived.

        Args:
            robot: The arriving robot.
            resource: The resource reached.
        """
        resource.occupants.add(robot.robot_id)


class TestPartialHold:
    """A blocked robot must keep what it already holds."""

    def test_partial_hold(self) -> None:
        """A robot holding crane and blocked on elevator must retain crane.

        Releasing on block would dissolve every deadlock before it formed and
        leave the safety checker with nothing to prevent.
        """
        world = World(
            TEST_CONFIG, RNGProvider(TEST_CONFIG.seed), wander_when_idle=False
        )
        allocator = _StubAllocator(
            world, grant=ResourceClass.CRANE, refuse=ResourceClass.ELEVATOR
        )
        manager = LifecycleManager(world, allocator)

        task = Task(
            task_id=0,
            template=TaskTemplate.HEAVY_LIFT,
            pickup=(1, 1),
            dropoff=(2, 2),
            service_duration=5.0,
            created_at=0.0,
        )
        manager.submit(task)

        robot = world.robots[0]
        for step in range(200):
            manager.update(now=step * TEST_CONFIG.dt, dt=TEST_CONFIG.dt)
            world.step(TEST_CONFIG.dt)

            if robot.current_task_id == 0 and robot.bundle_cursor >= 1:
                held_classes = {
                    world.resource(rid).resource_class for rid in robot.held_resources
                }
                assert (
                    ResourceClass.CRANE in held_classes
                ), "robot dropped the crane while blocked on the elevator"

        assert robot.state is RobotState.WAITING
        assert robot.bundle_cursor == 1
        assert allocator.refusals > 0
        assert not allocator.released, "bundle was released while still blocked"

        held_classes = {
            world.resource(rid).resource_class for rid in robot.held_resources
        }
        assert held_classes == {ResourceClass.CRANE}

    def test_holds_are_released_atomically_on_completion(self) -> None:
        """A completed task must return its whole bundle."""
        world = World(
            TEST_CONFIG, RNGProvider(TEST_CONFIG.seed), wander_when_idle=False
        )
        allocator = _StubAllocator(
            world, grant=ResourceClass.CRANE, refuse=ResourceClass.ELEVATOR
        )
        manager = LifecycleManager(world, allocator)
        manager.submit(
            Task(
                task_id=0,
                template=TaskTemplate.RECHARGE,
                pickup=(1, 1),
                dropoff=(1, 1),
                service_duration=1.0,
                created_at=0.0,
            )
        )

        robot = world.robots[0]
        for step in range(4000):
            manager.update(now=step * TEST_CONFIG.dt, dt=TEST_CONFIG.dt)
            world.step(TEST_CONFIG.dt)
            if robot.current_task_id is None and step > 10:
                break

        assert robot.current_task_id is None
        assert not robot.held_resources
        assert allocator.released, "nothing was released on completion"
        for resource in world.resources:
            assert robot.robot_id not in resource.holders
            assert robot.robot_id not in resource.occupants

    def test_resource_is_held_before_it_is_occupied(self) -> None:
        """Occupancy must lag the grant, which is the waste being measured."""
        world = World(
            TEST_CONFIG, RNGProvider(TEST_CONFIG.seed), wander_when_idle=False
        )
        allocator = _StubAllocator(
            world, grant=ResourceClass.DOCK, refuse=ResourceClass.CRANE
        )
        manager = LifecycleManager(world, allocator)
        manager.submit(
            Task(
                task_id=0,
                template=TaskTemplate.RECHARGE,
                pickup=(1, 1),
                dropoff=(1, 1),
                service_duration=1.0,
                created_at=0.0,
            )
        )

        robot = world.robots[0]
        saw_held_but_unused = False
        for step in range(4000):
            manager.update(now=step * TEST_CONFIG.dt, dt=TEST_CONFIG.dt)
            world.step(TEST_CONFIG.dt)
            for resource in world.resources:
                if resource.held_but_unused:
                    saw_held_but_unused = True
            if robot.current_task_id is None and step > 10:
                break

        assert saw_held_but_unused, "no early-binding gap was ever observed"
