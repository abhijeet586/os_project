"""Simulation entities: robots and the resources they contend for.

Entities are plain mutable dataclasses with no behaviour beyond bookkeeping.
Kinematics live in :mod:`warehouse_allocator.core.world` and allocation lives
in :mod:`warehouse_allocator.alloc`, so that an entity can never advance its
own state out of step with the rest of the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum

from warehouse_allocator.core.geometry import Coord

__all__ = ["Resource", "ResourceClass", "Robot", "RobotState"]


class ResourceClass(StrEnum):
    """A class of interchangeable warehouse resource.

    A :class:`~enum.StrEnum` so that a class name serialises to plain JSON in
    task traces and event logs without a custom encoder.

    Attributes:
        CRANE: Overhead lifting gear.
        ELEVATOR: Inter-floor transport. The class shared between the
            ``HEAVY_LIFT`` and ``FLOOR_TRANSFER`` bundles, which is what makes
            circular wait constructible.
        DOCK: Loading bay, also used for recharging.
    """

    CRANE = "crane"
    ELEVATOR = "elevator"
    DOCK = "dock"


class RobotState(Enum):
    """Lifecycle state of a robot.

    Attributes:
        IDLE: No task assigned; awaiting work.
        MOVING_TO_RESOURCE: Travelling toward a claimed resource. Time spent
            here while holding a grant is exactly the early-binding waste the
            project measures.
        WAITING: Blocked on an allocation request, stationary.
        WORKING: Performing task service at a resource.
        MOVING_TO_DROPOFF: Carrying a load to its destination.
        CHARGING: Replenishing battery at a dock.
    """

    IDLE = "idle"
    MOVING_TO_RESOURCE = "moving_to_resource"
    WAITING = "waiting"
    WORKING = "working"
    MOVING_TO_DROPOFF = "moving_to_dropoff"
    CHARGING = "charging"


@dataclass(slots=True)
class Robot:
    """A single warehouse robot.

    Position is continuous in metres while paths are discrete cells, so a
    robot may sit part-way between waypoints at any tick. Rendering
    interpolates from this continuous position; the simulation never rounds it
    back to the grid, because rounding would make travel time depend on
    timestep alignment.

    Attributes:
        robot_id: Stable unique identifier. Iteration order across robots is
            always by this id, never by dict or set order.
        position: Current continuous position as ``(x, y)`` in metres.
        velocity: Current velocity as ``(vx, vy)`` in metres per simulated
            second. Recomputed each tick from the path being followed.
        path: Remaining route as a tuple of cells, empty when not travelling.
        path_index: Index of the waypoint in ``path`` currently being moved
            toward.
        state: Current :class:`RobotState`.
        battery: Remaining charge as a fraction in ``[0.0, 1.0]``.
        current_task_id: Id of the assigned task, or ``None`` when idle.
        held_resources: Ids of resources currently granted to this robot. A
            robot holding one resource while requesting another is the
            precondition for deadlock and is therefore normal, expected state.
        bundle_cursor: How many classes of the current task's acquisition
            order have been granted so far. Lives on the robot rather than in
            a side table so that the whole task lifecycle is covered by
            :meth:`~warehouse_allocator.core.world.World.state_hash` and
            therefore by the determinism and headless-parity tests.
        visit_cursor: How many of the held resources have been visited.
        service_remaining: Simulated seconds of work left at the current
            resource, counted down while in ``WORKING``.
        destination: Cell the robot is currently routed to, or ``None``.
        next_request_at: Earliest simulated time this robot may retry a
            deferred request. Without a cooldown a blocked robot re-requests
            every tick, which floods the event log with hundreds of thousands
            of identical deferrals and buries the signal.
    """

    robot_id: int
    position: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    path: tuple[Coord, ...] = ()
    path_index: int = 0
    state: RobotState = RobotState.IDLE
    battery: float = 1.0
    current_task_id: int | None = None
    held_resources: set[int] = field(default_factory=set)
    bundle_cursor: int = 0
    visit_cursor: int = 0
    service_remaining: float = 0.0
    destination: Coord | None = None
    next_request_at: float = 0.0

    @property
    def cell(self) -> Coord:
        """Return the grid cell containing the robot.

        Returns:
            The floor of the continuous position, as ``(x, y)``.
        """
        return (int(self.position[0]), int(self.position[1]))

    @property
    def has_path(self) -> bool:
        """Report whether the robot still has waypoints to traverse.

        Returns:
            ``True`` if unvisited waypoints remain.
        """
        return self.path_index < len(self.path)

    def assign_path(self, path: tuple[Coord, ...]) -> None:
        """Set a new route for the robot to follow.

        Args:
            path: Route as returned by
                :func:`~warehouse_allocator.core.geometry.astar`, starting at
                the robot's current cell.
        """
        self.path = path
        # The first element is the cell the robot already occupies, so
        # traversal starts at the second to avoid a zero-length leg that
        # would consume a tick without moving.
        self.path_index = 1 if len(path) > 1 else len(path)

    def clear_path(self) -> None:
        """Discard the current route and stop the robot."""
        self.path = ()
        self.path_index = 0
        self.velocity = (0.0, 0.0)


@dataclass(slots=True)
class Resource:
    """A contended warehouse resource such as a crane, elevator, or dock.

    Attributes:
        resource_id: Stable unique identifier, unique across all classes.
        resource_class: Which :class:`ResourceClass` this instance belongs to.
        position: Fixed location as ``(x, y)`` in metres.
        capacity: Number of robots that may hold it simultaneously.
        holders: Ids of robots currently holding it. A robot appears here from
            the moment its request is granted, which may be long before it
            physically arrives; that gap is the early-binding waste under
            study.
        occupants: Ids of holders that have physically arrived and are using
            the resource. Always a subset of ``holders``. The difference
            between the two sets is what
            :mod:`warehouse_allocator.metrics.analysis` integrates to compute
            held-but-unused time.
    """

    resource_id: int
    resource_class: ResourceClass
    position: tuple[float, float]
    capacity: int = 1
    holders: set[int] = field(default_factory=set)
    occupants: set[int] = field(default_factory=set)

    @property
    def cell(self) -> Coord:
        """Return the grid cell containing the resource.

        Returns:
            The floor of the continuous position, as ``(x, y)``.
        """
        return (int(self.position[0]), int(self.position[1]))

    @property
    def is_full(self) -> bool:
        """Report whether the resource has no spare capacity.

        Returns:
            ``True`` if the holder count has reached capacity.
        """
        return len(self.holders) >= self.capacity

    @property
    def held_but_unused(self) -> bool:
        """Report whether the resource is claimed but nobody has arrived.

        Returns:
            ``True`` if it has holders but no occupants; the instantaneous
            form of the project's central metric.
        """
        return bool(self.holders) and not self.occupants
