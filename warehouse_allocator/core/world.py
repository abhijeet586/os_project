"""World state: obstacle layout, entity registry, and kinematics.

The world owns the grid, the robots, and the resources, and advances physical
motion one fixed timestep at a time. It contains no allocation logic: whether
a robot is *allowed* to hold a resource is decided in
:mod:`warehouse_allocator.alloc`, and task lifecycle arrives in Stage 2.

Determinism rules observed throughout:

* Entities are stored in dicts but always iterated in sorted-id order, so
  insertion order can never influence results.
* Every stochastic draw comes from a named
  :class:`~warehouse_allocator.core.rng.RNGProvider` stream.
* :meth:`World.state_hash` encodes floats by their exact IEEE-754 bytes rather
  than by a rounded decimal form, so the determinism test compares true bit
  equality rather than display equality.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Final

import numpy as np
import numpy.typing as npt

from warehouse_allocator.config import SimConfig
from warehouse_allocator.core.entities import (
    Resource,
    ResourceClass,
    Robot,
    RobotState,
)
from warehouse_allocator.core.geometry import Coord, Grid, astar
from warehouse_allocator.core.rng import RNGProvider

__all__ = ["World"]

# Racking is generated as rectangular blocks rather than scattered single
# cells because the experiment depends on euclidean distance diverging from
# true path cost. Isolated one-cell obstacles are trivially walked around and
# would leave RadiusPolicy and ETAPolicy nearly indistinguishable.
_RACK_MIN_SPAN: Final[int] = 2
_RACK_MAX_SPAN: Final[int] = 9

# Cells kept clear at the warehouse perimeter, giving a guaranteed ring
# corridor so a layout can never fragment into disconnected pockets.
_BORDER_MARGIN: Final[int] = 1

# Ceiling on rectangle placement attempts, so a density that cannot be
# achieved under the border and connectivity constraints terminates instead
# of looping forever.
_MAX_RACK_ATTEMPTS: Final[int] = 4000


class World:
    """Owns grid, entities, and physical state advancement.

    Attributes:
        config: The immutable configuration this world was built from.
        rng: The provider supplying every stochastic draw.
        grid: The occupancy grid including generated racking.
    """

    __slots__ = (
        "_free_cells",
        "_resources",
        "_robots",
        "config",
        "grid",
        "rng",
        "wander_when_idle",
    )

    def __init__(
        self,
        config: SimConfig,
        rng: RNGProvider,
        *,
        wander_when_idle: bool = True,
    ) -> None:
        """Build a world: generate the layout, then place entities.

        Args:
            config: Simulation parameters.
            rng: Seeded provider. The ``layout`` stream drives racking and the
                ``spawn`` stream drives entity placement, so changing robot
                count cannot alter the obstacle layout.
            wander_when_idle: Whether idle robots route themselves to a random
                cell. True exercises A*, the RNG, and kinematics together for
                the Stage 1 determinism tests, which run with no task layer at
                all. A full simulation passes False, because the task
                lifecycle assigns every route instead.
        """
        self.config: Final[SimConfig] = config
        self.rng: Final[RNGProvider] = rng
        self.wander_when_idle = wander_when_idle
        self.grid: Final[Grid] = Grid(config.grid_width, config.grid_height)
        self._robots: dict[int, Robot] = {}
        self._resources: dict[int, Resource] = {}

        self._generate_layout()
        self._free_cells: Final[tuple[Coord, ...]] = self._collect_free_cells()
        self._place_resources()
        self._place_robots()

    @property
    def robots(self) -> tuple[Robot, ...]:
        """Return every robot in ascending id order.

        Returns:
            Robots sorted by ``robot_id``, so callers iterating this tuple
            cannot introduce order-dependent nondeterminism.
        """
        return tuple(self._robots[key] for key in sorted(self._robots))

    @property
    def resources(self) -> tuple[Resource, ...]:
        """Return every resource in ascending id order.

        Returns:
            Resources sorted by ``resource_id``.
        """
        return tuple(self._resources[key] for key in sorted(self._resources))

    def robot(self, robot_id: int) -> Robot:
        """Look up a robot by id.

        Args:
            robot_id: Identifier to fetch.

        Returns:
            The matching :class:`~warehouse_allocator.core.entities.Robot`.

        Raises:
            KeyError: If no such robot exists.
        """
        return self._robots[robot_id]

    def resource(self, resource_id: int) -> Resource:
        """Look up a resource by id.

        Args:
            resource_id: Identifier to fetch.

        Returns:
            The matching :class:`~warehouse_allocator.core.entities.Resource`.

        Raises:
            KeyError: If no such resource exists.
        """
        return self._resources[resource_id]

    def resources_of_class(self, resource_class: ResourceClass) -> tuple[Resource, ...]:
        """Return every resource of one class, in ascending id order.

        Args:
            resource_class: Class to filter by.

        Returns:
            Matching resources sorted by ``resource_id``.
        """
        return tuple(
            resource
            for resource in self.resources
            if resource.resource_class is resource_class
        )

    def step(self, dt: float) -> None:
        """Advance physical state by one fixed timestep.

        Only kinematics and battery drain are advanced. Allocation, task
        assignment, and lease expiry are the responsibility of later layers.

        Args:
            dt: Timestep in simulated seconds. Must match the clock's ``dt``;
                a varying timestep would make the run irreproducible.
        """
        for robot in self.robots:
            if self.wander_when_idle and not robot.has_path:
                self._assign_wander_path(robot)
            self._advance_robot(robot, dt)

    def state_hash(self) -> str:
        """Return a stable hash of all mutable simulation state.

        Floats are encoded by their exact little-endian IEEE-754 bytes, not by
        a formatted decimal string, so two runs must agree bit for bit rather
        than merely to display precision. Sets are sorted before encoding
        because set iteration order is not part of the simulation's meaning.

        Returns:
            A hex digest covering the grid layout and every robot and resource
            field that can change during a run.
        """
        digest = hashlib.blake2b(digest_size=32)
        digest.update(self.grid.blocked.tobytes())

        for robot in self.robots:
            digest.update(b"R")
            digest.update(struct.pack("<q", robot.robot_id))
            digest.update(struct.pack("<dd", *robot.position))
            digest.update(struct.pack("<dd", *robot.velocity))
            digest.update(struct.pack("<q", robot.path_index))
            digest.update(robot.state.value.encode("utf-8"))
            digest.update(struct.pack("<d", robot.battery))
            digest.update(
                struct.pack(
                    "<q",
                    -1 if robot.current_task_id is None else robot.current_task_id,
                )
            )
            digest.update(struct.pack("<q", robot.bundle_cursor))
            digest.update(struct.pack("<q", robot.visit_cursor))
            digest.update(struct.pack("<d", robot.service_remaining))
            digest.update(struct.pack("<d", robot.next_request_at))
            digest.update(
                struct.pack(
                    "<qq",
                    *(robot.destination if robot.destination is not None else (-1, -1)),
                )
            )
            for cell in robot.path:
                digest.update(struct.pack("<qq", cell[0], cell[1]))
            for held in sorted(robot.held_resources):
                digest.update(struct.pack("<q", held))

        for resource in self.resources:
            digest.update(b"S")
            digest.update(struct.pack("<q", resource.resource_id))
            digest.update(resource.resource_class.value.encode("utf-8"))
            digest.update(struct.pack("<dd", *resource.position))
            digest.update(struct.pack("<q", resource.capacity))
            for holder in sorted(resource.holders):
                digest.update(struct.pack("<q", holder))
            for occupant in sorted(resource.occupants):
                digest.update(struct.pack("<q", occupant))

        return digest.hexdigest()

    def _generate_layout(self) -> None:
        """Generate racking from the ``layout`` stream, then prune pockets.

        Rectangles are placed until the requested obstacle density is reached,
        after which every free cell outside the largest connected component is
        blocked off. Pruning guarantees any robot can reach any resource, so a
        failed A* search always signals a bug rather than an unlucky layout.
        """
        stream = self.rng.stream("layout")
        height, width = self.config.grid_height, self.config.grid_width
        mask: npt.NDArray[np.bool_] = np.zeros((height, width), dtype=np.bool_)

        interior = (width - 2 * _BORDER_MARGIN) * (height - 2 * _BORDER_MARGIN)
        target = int(interior * self.config.obstacle_density)

        attempts = 0
        while int(np.count_nonzero(mask)) < target and attempts < _MAX_RACK_ATTEMPTS:
            attempts += 1
            span_x = int(stream.integers(_RACK_MIN_SPAN, _RACK_MAX_SPAN + 1))
            span_y = int(stream.integers(_RACK_MIN_SPAN, _RACK_MAX_SPAN + 1))
            max_x = width - _BORDER_MARGIN - span_x
            max_y = height - _BORDER_MARGIN - span_y
            if max_x <= _BORDER_MARGIN or max_y <= _BORDER_MARGIN:
                continue
            x0 = int(stream.integers(_BORDER_MARGIN, max_x))
            y0 = int(stream.integers(_BORDER_MARGIN, max_y))
            mask[y0 : y0 + span_y, x0 : x0 + span_x] = True

        self.grid.set_mask(mask)
        self._prune_disconnected()

    def _prune_disconnected(self) -> None:
        """Block every free cell not in the largest connected component."""
        seed = self._first_free_cell()
        if seed is None:
            raise RuntimeError("layout generation left no passable cells")
        reachable = self.grid.connected_component(seed)
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell: Coord = (x, y)
                if self.grid.is_free(cell) and cell not in reachable:
                    self.grid.set_blocked(cell, True)

    def _first_free_cell(self) -> Coord | None:
        """Return the lowest-ordered passable cell.

        Scanning in fixed row-major order rather than sampling keeps the
        connectivity seed independent of the RNG, so adding a draw elsewhere
        cannot change which component survives pruning.

        Returns:
            The first passable cell in row-major order, or ``None`` if the
            grid is fully blocked.
        """
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                if self.grid.is_free((x, y)):
                    return (x, y)
        return None

    def _collect_free_cells(self) -> tuple[Coord, ...]:
        """Enumerate passable cells in fixed row-major order.

        Returns:
            Every passable cell, ordered so that index-based sampling is
            reproducible.
        """
        return tuple(
            (x, y)
            for y in range(self.grid.height)
            for x in range(self.grid.width)
            if self.grid.is_free((x, y))
        )

    def _sample_cells(self, count: int, stream_name: str) -> tuple[Coord, ...]:
        """Draw distinct passable cells from a named RNG stream.

        Args:
            count: Number of cells required.
            stream_name: Stream to draw from.

        Returns:
            ``count`` distinct passable cells.

        Raises:
            RuntimeError: If the layout has fewer passable cells than needed.
        """
        if count > len(self._free_cells):
            raise RuntimeError(
                f"need {count} free cells but layout has {len(self._free_cells)}"
            )
        stream = self.rng.stream(stream_name)
        indices = stream.choice(len(self._free_cells), size=count, replace=False)
        return tuple(self._free_cells[int(index)] for index in indices)

    def _place_resources(self) -> None:
        """Create every configured resource at a distinct passable cell."""
        counts = self.config.resource_counts
        total = sum(count for _, count in counts)
        cells = self._sample_cells(total, "spawn")

        next_id = 0
        cursor = 0
        for class_name, count in counts:
            resource_class = ResourceClass(class_name)
            for _ in range(count):
                cell = cells[cursor]
                cursor += 1
                self._resources[next_id] = Resource(
                    resource_id=next_id,
                    resource_class=resource_class,
                    position=(cell[0] + 0.5, cell[1] + 0.5),
                )
                next_id += 1

    def _place_robots(self) -> None:
        """Create every robot at a passable cell, fully charged and idle."""
        cells = self._sample_cells(self.config.robot_count, "spawn")
        for robot_id, cell in enumerate(cells):
            self._robots[robot_id] = Robot(
                robot_id=robot_id,
                position=(cell[0] + 0.5, cell[1] + 0.5),
                state=RobotState.IDLE,
            )

    def _assign_wander_path(self, robot: Robot) -> None:
        """Route an idle robot to a random passable cell.

        Stage 1 has no tasks, so this exists to exercise A*, the RNG streams,
        and kinematics together under the determinism test. Stage 2's task
        lifecycle assigns routes instead, at which point this becomes the
        fallback for genuinely idle robots.

        Args:
            robot: The robot to route.
        """
        stream = self.rng.stream("noise")
        index = int(stream.integers(0, len(self._free_cells)))
        goal = self._free_cells[index]
        if goal == robot.cell:
            return
        path = astar(self.grid, robot.cell, goal)
        # Pruning guarantees connectivity, so a None here means the grid was
        # mutated behind the world's back rather than that the goal is remote.
        if path is None:
            return
        robot.assign_path(path)
        robot.state = RobotState.MOVING_TO_RESOURCE

    def _advance_robot(self, robot: Robot, dt: float) -> None:
        """Move one robot along its path and drain its battery.

        A single tick may cross several waypoints if the distance budget
        allows, so travel time never depends on how waypoints happen to align
        with timestep boundaries.

        Args:
            robot: The robot to advance.
            dt: Timestep in simulated seconds.
        """
        if not robot.has_path:
            robot.velocity = (0.0, 0.0)
            # Only the wander exerciser owns robot state; when a task
            # lifecycle is driving, it decides what an arrival means.
            if self.wander_when_idle and robot.state is RobotState.MOVING_TO_RESOURCE:
                robot.state = RobotState.IDLE
            return

        origin = robot.position
        budget = self.config.robot_speed * dt
        x, y = origin

        while budget > 0.0 and robot.path_index < len(robot.path):
            cell = robot.path[robot.path_index]
            target_x, target_y = cell[0] + 0.5, cell[1] + 0.5
            dx, dy = target_x - x, target_y - y
            leg = float(np.hypot(dx, dy))
            if leg <= budget:
                x, y = target_x, target_y
                robot.path_index += 1
                budget -= leg
            else:
                scale = budget / leg
                x += dx * scale
                y += dy * scale
                budget = 0.0

        robot.position = (x, y)
        robot.velocity = ((x - origin[0]) / dt, (y - origin[1]) / dt)
        robot.battery = max(0.0, robot.battery - self.config.battery_drain_rate * dt)

        # Compared field-wise rather than through ``has_path``: the guard at
        # the top of this method narrows that property, and the type checker
        # cannot see that the loop above advances ``path_index`` past it.
        if robot.path_index >= len(robot.path):
            robot.clear_path()
            if self.wander_when_idle:
                robot.state = RobotState.IDLE

    def add_robot(self) -> Robot:
        """Add a fresh idle robot at a passable cell.

        Used by the interactive fleet-size control. Draws from the ``spawn``
        stream, so an interactive session stays reproducible given the same
        sequence of control inputs.

        Returns:
            The newly created robot.
        """
        cell = self._sample_cells(1, "spawn")[0]
        robot_id = (max(self._robots) + 1) if self._robots else 0
        robot = Robot(
            robot_id=robot_id,
            position=(cell[0] + 0.5, cell[1] + 0.5),
            state=RobotState.IDLE,
        )
        self._robots[robot_id] = robot
        return robot

    def remove_idle_robot(self) -> int | None:
        """Remove the highest-numbered robot that holds nothing.

        Only idle, empty-handed robots are removable: taking a robot out while
        it holds resources would strand those resources and corrupt the
        Banker's matrices.

        Returns:
            The removed robot's id, or ``None`` if every robot is busy.
        """
        for robot_id in sorted(self._robots, reverse=True):
            robot = self._robots[robot_id]
            if robot.current_task_id is None and not robot.held_resources:
                del self._robots[robot_id]
                return robot_id
        return None

    def route_to(self, robot: Robot, goal: Coord) -> bool:
        """Route a robot to a destination cell.

        Args:
            robot: The robot to route.
            goal: Destination cell as ``(x, y)``.

        Returns:
            ``True`` if a route was assigned or the robot is already there,
            ``False`` if the destination is unreachable.
        """
        robot.destination = goal
        if robot.cell == goal:
            robot.clear_path()
            return True
        path = astar(self.grid, robot.cell, goal)
        if path is None:
            return False
        robot.assign_path(path)
        return True

    def at_destination(self, robot: Robot) -> bool:
        """Report whether a routed robot has arrived.

        Args:
            robot: The robot to test.

        Returns:
            ``True`` if the robot has a destination and no path remains.
        """
        return robot.destination is not None and not robot.has_path
