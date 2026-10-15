"""Top-down Pygame renderer.

Reads world state and draws it. It holds no simulation state of its own beyond
the previous frame's robot positions, which exist purely so motion can be
interpolated between fixed simulation steps and look smooth at any framerate.

The renderer never mutates the world. Every method takes the world as a
read-only input.

The claim lines are the point of this view. A line is drawn from each robot to
every resource it has been *granted but has not yet reached*, so early binding
becomes something you can watch happening rather than only a number in a table.
"""

from __future__ import annotations

from typing import Final

import pygame

from warehouse_allocator.core.entities import ResourceClass, RobotState
from warehouse_allocator.core.world import World

__all__ = ["Renderer"]

# Same categorical hues as the figures, so a policy or class reads the same in
# the live view as it does in the report.
CLASS_COLORS: Final[dict[ResourceClass, tuple[int, int, int]]] = {
    ResourceClass.CRANE: (42, 120, 214),
    ResourceClass.ELEVATOR: (235, 104, 52),
    ResourceClass.DOCK: (27, 175, 122),
}

STATE_COLORS: Final[dict[RobotState, tuple[int, int, int]]] = {
    RobotState.IDLE: (150, 150, 145),
    RobotState.WAITING: (235, 104, 52),
    RobotState.MOVING_TO_RESOURCE: (42, 120, 214),
    RobotState.MOVING_TO_DROPOFF: (74, 58, 167),
    RobotState.WORKING: (27, 175, 122),
    RobotState.CHARGING: (232, 123, 164),
}

BACKGROUND: Final[tuple[int, int, int]] = (252, 252, 251)
OBSTACLE: Final[tuple[int, int, int]] = (206, 205, 198)
CLAIM_LINE: Final[tuple[int, int, int]] = (235, 104, 52)
PATH_LINE: Final[tuple[int, int, int]] = (170, 190, 215)
RADIUS_LINE: Final[tuple[int, int, int]] = (200, 200, 195)


class Renderer:
    """Draws the warehouse, its robots, and their outstanding claims.

    Attributes:
        cell_size: Pixels per grid cell.
        show_paths: Whether to draw each robot's A* route.
        show_claim_lines: Whether to draw grant-but-not-arrived links.
        show_radii: Whether to draw the admission proximity radius.
    """

    __slots__ = (
        "_font",
        "_previous",
        "cell_size",
        "show_claim_lines",
        "show_paths",
        "show_radii",
        "surface",
    )

    def __init__(self, surface: pygame.Surface, cell_size: int) -> None:
        """Initialize the renderer.

        Args:
            surface: Target surface to draw onto.
            cell_size: Pixels per grid cell.
        """
        self.surface = surface
        self.cell_size = cell_size
        self.show_paths = False
        self.show_claim_lines = True
        self.show_radii = False
        self._previous: dict[int, tuple[float, float]] = {}
        self._font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 11)

    def capture(self, world: World) -> None:
        """Record current robot positions as the interpolation baseline.

        Called once per *simulation* step, not per frame. Reading positions is
        the only interaction the renderer has with the world.

        Args:
            world: The world to snapshot.
        """
        self._previous = {robot.robot_id: robot.position for robot in world.robots}

    def draw(self, world: World, alpha: float, radius: float | None = None) -> None:
        """Draw one frame.

        Args:
            world: The world to render.
            alpha: Interpolation factor in ``[0, 1]`` between the captured
                positions and the current ones. Rendering between simulation
                states is what lets the view run at any framerate without
                changing the simulation's fixed timestep.
            radius: Admission radius to visualise, when the active policy has
                one.
        """
        self.surface.fill(BACKGROUND)
        self._draw_obstacles(world)
        if self.show_radii and radius is not None:
            self._draw_radii(world, radius)
        self._draw_resources(world)
        if self.show_paths:
            self._draw_paths(world)
        if self.show_claim_lines:
            self._draw_claims(world, alpha)
        self._draw_robots(world, alpha)

    def _to_pixels(self, point: tuple[float, float]) -> tuple[int, int]:
        """Convert a world position in metres to screen pixels.

        Args:
            point: World position as ``(x, y)``.

        Returns:
            Pixel coordinates.
        """
        return (int(point[0] * self.cell_size), int(point[1] * self.cell_size))

    def _interpolated(
        self, robot_id: int, current: tuple[float, float], alpha: float
    ) -> tuple[float, float]:
        """Blend a robot's previous and current positions.

        Args:
            robot_id: The robot to interpolate.
            current: Its current position.
            alpha: Interpolation factor in ``[0, 1]``.

        Returns:
            The interpolated position.
        """
        previous = self._previous.get(robot_id, current)
        return (
            previous[0] + (current[0] - previous[0]) * alpha,
            previous[1] + (current[1] - previous[1]) * alpha,
        )

    def _draw_obstacles(self, world: World) -> None:
        """Draw the racking.

        Args:
            world: The world whose grid is drawn.
        """
        size = self.cell_size
        blocked = world.grid.blocked
        for y in range(world.grid.height):
            row = blocked[y]
            for x in range(world.grid.width):
                if row[x]:
                    pygame.draw.rect(
                        self.surface, OBSTACLE, (x * size, y * size, size, size)
                    )

    def _draw_resources(self, world: World) -> None:
        """Draw resources, shaded by occupancy.

        A resource that is held but not yet reached is drawn hollow, so the
        wasted reservation is visible at a glance.

        Args:
            world: The world whose resources are drawn.
        """
        size = self.cell_size
        for resource in world.resources:
            colour = CLASS_COLORS[resource.resource_class]
            x, y = resource.cell
            rect = pygame.Rect(x * size - size, y * size - size, size * 3, size * 3)
            if resource.occupants:
                pygame.draw.rect(self.surface, colour, rect, border_radius=3)
            elif resource.holders:
                # Held but unreached: outline only. This is the waste.
                pygame.draw.rect(self.surface, colour, rect, width=2, border_radius=3)
            else:
                faded = tuple(min(255, channel + 90) for channel in colour)
                pygame.draw.rect(self.surface, faded, rect, border_radius=3)
                pygame.draw.rect(self.surface, colour, rect, width=1, border_radius=3)

    def _draw_paths(self, world: World) -> None:
        """Draw each robot's remaining A* route.

        Args:
            world: The world whose robots are drawn.
        """
        size = self.cell_size
        for robot in world.robots:
            if not robot.has_path:
                continue
            points = [
                ((cx + 0.5) * size, (cy + 0.5) * size)
                for cx, cy in robot.path[robot.path_index :]
            ]
            if len(points) >= 2:
                pygame.draw.lines(self.surface, PATH_LINE, False, points, 1)

    def _draw_claims(self, world: World, alpha: float) -> None:
        """Draw a line from each robot to every claimed-but-unreached resource.

        Args:
            world: The world to read.
            alpha: Interpolation factor for robot positions.
        """
        for robot in world.robots:
            origin = self._to_pixels(
                self._interpolated(robot.robot_id, robot.position, alpha)
            )
            for resource_id in sorted(robot.held_resources):
                resource = world.resource(resource_id)
                if robot.robot_id in resource.occupants:
                    continue
                pygame.draw.line(
                    self.surface,
                    CLAIM_LINE,
                    origin,
                    self._to_pixels(resource.position),
                    1,
                )

    def _draw_radii(self, world: World, radius: float) -> None:
        """Draw the admission radius around each resource.

        Args:
            world: The world to read.
            radius: Radius in metres.
        """
        for resource in world.resources:
            pygame.draw.circle(
                self.surface,
                RADIUS_LINE,
                self._to_pixels(resource.position),
                int(radius * self.cell_size),
                1,
            )

    def _draw_robots(self, world: World, alpha: float) -> None:
        """Draw every robot, coloured by lifecycle state.

        Args:
            world: The world to read.
            alpha: Interpolation factor for positions.
        """
        radius = max(3, self.cell_size // 2)
        for robot in world.robots:
            position = self._to_pixels(
                self._interpolated(robot.robot_id, robot.position, alpha)
            )
            pygame.draw.circle(
                self.surface, STATE_COLORS[robot.state], position, radius
            )
            # A white ring separates overlapping robots in a crowded aisle.
            pygame.draw.circle(self.surface, (255, 255, 255), position, radius, 1)
            if robot.held_resources:
                label = self._font.render(
                    str(len(robot.held_resources)), True, (11, 11, 11)
                )
                self.surface.blit(label, (position[0] + radius, position[1] - radius))
