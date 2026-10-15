"""Live statistics panel.

Reads the simulation and draws numbers. Like the renderer it is strictly a
consumer: it never mutates state, and it derives its figures from the same
event log the offline analysis uses, so the live numbers and the reported
numbers cannot drift apart.

The tamper banner is the important part. Once a parameter has been changed
mid-run, a persistent warning is drawn and never cleared, because an
interactively tweaked run must never be mistaken for experimental data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pygame

from warehouse_allocator.metrics.events import (
    LeaseRevoked,
    RequestForceAdmitted,
    RequestRefusedUnsafe,
    TaskCompleted,
)

if TYPE_CHECKING:
    from warehouse_allocator.simulation import Simulation
    from warehouse_allocator.viz.controls import LiveControls

__all__ = ["Hud"]

PANEL_BG: Final[tuple[int, int, int]] = (247, 247, 245)
TEXT: Final[tuple[int, int, int]] = (11, 11, 11)
MUTED: Final[tuple[int, int, int]] = (82, 81, 78)
ACCENT: Final[tuple[int, int, int]] = (42, 120, 214)
WARN_BG: Final[tuple[int, int, int]] = (227, 73, 72)
SECONDS_PER_HOUR: Final[float] = 3600.0


class Hud:
    """Draws the live statistics panel and the tamper banner.

    Attributes:
        width: Panel width in pixels.
    """

    __slots__ = ("_font", "_heading", "_small", "surface", "width")

    def __init__(self, surface: pygame.Surface, width: int) -> None:
        """Initialize the panel.

        Args:
            surface: Target surface.
            width: Panel width in pixels.
        """
        self.surface = surface
        self.width = width
        self._font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 12)
        self._heading = pygame.font.SysFont(
            "menlo,dejavusansmono,monospace", 13, bold=True
        )
        self._small = pygame.font.SysFont("menlo,dejavusansmono,monospace", 10)

    def draw(
        self, simulation: Simulation, controls: LiveControls, origin_x: int
    ) -> None:
        """Draw the panel.

        Args:
            simulation: The run being displayed.
            controls: Live control state, for the policy name and speed.
            origin_x: Left edge of the panel in pixels.
        """
        height = self.surface.get_height()
        pygame.draw.rect(self.surface, PANEL_BG, (origin_x, 0, self.width, height))

        collector = simulation.collector
        now = simulation.clock.now()
        completed = collector.count_of(TaskCompleted)
        rate = completed * SECONDS_PER_HOUR / now if now > 0 else 0.0

        held = sum(1 for r in simulation.world.resources if r.holders)
        unreached = sum(1 for r in simulation.world.resources if r.held_but_unused)
        total = len(simulation.world.resources)

        y = 14
        y = self._line(origin_x, y, "WAREHOUSE FLEET ALLOCATOR", heading=True)
        y += 6
        y = self._line(origin_x, y, f"policy      {controls.policy_name}", ACCENT)
        y = self._line(origin_x, y, f"  {simulation.policy.describe()}", MUTED)
        y = self._line(origin_x, y, f"speed       {controls.speed_label}")
        y += 8

        y = self._line(origin_x, y, "-- TIME --", heading=True)
        y = self._line(origin_x, y, f"simulated   {now:8.1f} s")
        y = self._line(origin_x, y, f"step        {simulation.clock.step_index:8d}")
        y += 8

        y = self._line(origin_x, y, "-- THROUGHPUT --", heading=True)
        y = self._line(origin_x, y, f"completed   {completed:8d}")
        y = self._line(origin_x, y, f"rate        {rate:8.1f} /h", ACCENT)
        y = self._line(
            origin_x, y, f"queued      {simulation.lifecycle.queue_depth:8d}"
        )
        y += 8

        y = self._line(origin_x, y, "-- RESOURCES --", heading=True)
        for resource_class in simulation.state.classes:
            instances = simulation.world.resources_of_class(resource_class)
            busy = sum(1 for r in instances if r.holders)
            idle_held = sum(1 for r in instances if r.held_but_unused)
            y = self._line(
                origin_x,
                y,
                f"{resource_class.value:<10}{busy:2d}/{len(instances):<2d} "
                f"unreached {idle_held:2d}",
            )
        y = self._line(origin_x, y, f"held        {held:2d}/{total:<2d}")
        y = self._line(origin_x, y, f"unreached   {unreached:2d}/{total:<2d}", WARN_BG)
        y += 8

        y = self._line(origin_x, y, "-- ALLOCATOR --", heading=True)
        y = self._line(
            origin_x, y, f"defer queue {simulation.allocator.deferral_queue_depth:8d}"
        )
        y = self._line(
            origin_x, y, f"forced      {collector.count_of(RequestForceAdmitted):8d}"
        )
        y = self._line(
            origin_x, y, f"revoked     {collector.count_of(LeaseRevoked):8d}"
        )
        y = self._line(
            origin_x,
            y,
            f"unsafe      {collector.count_of(RequestRefusedUnsafe):8d}",
        )
        y += 8

        y = self._line(origin_x, y, "-- PARAMETERS --", heading=True)
        y = self._line(origin_x, y, f"radius      {controls.radius:8.1f} m")
        y = self._line(origin_x, y, f"eta thresh  {controls.eta_threshold:8.1f} s")
        y = self._line(origin_x, y, f"lease       {controls.lease_duration:8.1f} s")
        y = self._line(origin_x, y, f"max defer   {controls.max_deferrals:8d}")
        y = self._line(origin_x, y, f"robots      {controls.robot_count:8d}")
        y = self._line(origin_x, y, f"arrival     {controls.arrival_rate:8.3f} /s")

        self._draw_help(origin_x, height)
        if not collector.comparable:
            self._draw_tamper_banner()

    def _line(
        self,
        origin_x: int,
        y: int,
        text: str,
        colour: tuple[int, int, int] = TEXT,
        heading: bool = False,
    ) -> int:
        """Draw one line of panel text.

        Args:
            origin_x: Panel left edge.
            y: Vertical position.
            text: The text to draw.
            colour: Text colour.
            heading: Whether to use the bold heading face.

        Returns:
            The next vertical position.
        """
        font = self._heading if heading else self._font
        self.surface.blit(font.render(text, True, colour), (origin_x + 12, y))
        return y + (18 if heading else 15)

    def _draw_help(self, origin_x: int, height: int) -> None:
        """Draw the key bindings at the foot of the panel.

        Args:
            origin_x: Panel left edge.
            height: Surface height.
        """
        lines = [
            "SPACE pause   . step   [ ] speed",
            "P policy   R/E/L -+ params",
            "D max-deferrals   , / robots",
            "A paths  C claims  Y radii  Q quit",
        ]
        y = height - 14 * len(lines) - 12
        for line in lines:
            self.surface.blit(self._small.render(line, True, MUTED), (origin_x + 12, y))
            y += 14

    def _draw_tamper_banner(self) -> None:
        """Draw the persistent non-comparable warning.

        Once shown this is never cleared for the life of the run. A run whose
        parameters were changed by hand is not experimental data and must never
        be reported as such.
        """
        width = self.surface.get_width()
        pygame.draw.rect(self.surface, WARN_BG, (0, 0, width, 26))
        message = (
            "PARAMETER CHANGED MID-RUN - THIS RUN IS NOT COMPARABLE "
            "AND MUST NOT BE REPORTED"
        )
        label = self._heading.render(message, True, (255, 255, 255))
        self.surface.blit(label, (max(8, (width - label.get_width()) // 2), 5))
