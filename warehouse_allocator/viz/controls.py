"""Runtime-adjustable parameters and the tamper marking that guards them.

Every parameter exposed on the command line is also adjustable here while the
simulation runs, and vice versa.

**Tamper marking.** Any mid-run change emits a
:class:`~warehouse_allocator.metrics.events.ParameterChanged` event, which
permanently sets ``collector.comparable`` to ``False`` and raises a persistent
banner in the HUD. This is deliberate and one-way: interactive exploration is
useful for understanding the system, but a run whose parameters were nudged by
hand is not experimental data, and nothing should let it be reported as if it
were.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import pygame

from warehouse_allocator.alloc.policies import (
    ETAPolicy,
    LeasePolicy,
    RadiusPolicy,
    build_policy,
)
from warehouse_allocator.metrics.events import ParameterChanged

if TYPE_CHECKING:
    from warehouse_allocator.simulation import Simulation

__all__ = ["POLICY_CYCLE", "SPEEDS", "LiveControls"]

POLICY_CYCLE: Final[tuple[str, ...]] = ("null", "radius", "eta", "lease")

# Speed multipliers. 0 is pause; the last entry runs headless-fast, advancing
# many simulation steps per rendered frame.
SPEEDS: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0, 64.0)


class LiveControls:
    """Mutable runtime parameters, applied to a live simulation.

    Attributes:
        simulation: The run being controlled.
        speed_index: Index into :data:`SPEEDS`.
        single_step: Whether a single step has been requested while paused.
    """

    __slots__ = (
        "_speed_index",
        "arrival_rate",
        "eta_threshold",
        "lease_duration",
        "max_deferrals",
        "policy_name",
        "radius",
        "renderer",
        "robot_count",
        "simulation",
        "single_step",
    )

    def __init__(self, simulation: Simulation) -> None:
        """Initialize controls from the simulation's configuration.

        Args:
            simulation: The run to control.
        """
        self.simulation = simulation
        config = simulation.config
        self.policy_name = config.policy
        self.radius = config.radius
        self.eta_threshold = config.eta_threshold
        self.lease_duration = config.lease_duration
        self.max_deferrals = config.max_deferrals
        self.robot_count = config.robot_count
        self.arrival_rate = config.task_arrival_rate
        self._speed_index = SPEEDS.index(1.0)
        self.single_step = False
        self.renderer: Any | None = None

    @property
    def speed(self) -> float:
        """Return the current speed multiplier.

        Returns:
            Simulated seconds per real second, relative to real time.
        """
        return SPEEDS[self._speed_index]

    @property
    def paused(self) -> bool:
        """Report whether the simulation is paused.

        Returns:
            ``True`` when the speed multiplier is zero.
        """
        return self.speed == 0.0

    @property
    def speed_label(self) -> str:
        """Return a display label for the current speed.

        Returns:
            ``"paused"`` or a multiplier such as ``"4.0x"``.
        """
        return "paused" if self.paused else f"{self.speed:g}x"

    @property
    def active_radius(self) -> float | None:
        """Return the radius to visualise, if the active policy has one.

        Returns:
            The radius in metres, or ``None`` for policies without one.
        """
        return self.radius if self.policy_name == "radius" else None

    def _mark(self, name: str, previous: Any, current: Any) -> None:
        """Record a parameter change and mark the run non-comparable.

        Args:
            name: The parameter changed.
            previous: Its prior value.
            current: Its new value.
        """
        self.simulation.collector.record(
            ParameterChanged(
                at=self.simulation.clock.now(),
                name=name,
                previous=str(previous),
                current=str(current),
            )
        )

    def handle(self, event: pygame.event.Event) -> bool:
        """Process one keyboard event.

        Args:
            event: The Pygame event.

        Returns:
            ``False`` if the application should quit, ``True`` otherwise.
        """
        if event.type == pygame.QUIT:
            return False
        if event.type != pygame.KEYDOWN:
            return True

        key = event.key
        shift = bool(event.mod & pygame.KMOD_SHIFT)

        if key in (pygame.K_q, pygame.K_ESCAPE):
            return False
        if key == pygame.K_SPACE:
            self._toggle_pause()
        elif key == pygame.K_PERIOD:
            self.single_step = True
        elif key == pygame.K_LEFTBRACKET:
            self._speed_index = max(0, self._speed_index - 1)
        elif key == pygame.K_RIGHTBRACKET:
            self._speed_index = min(len(SPEEDS) - 1, self._speed_index + 1)
        elif key == pygame.K_p:
            self.cycle_policy()
        elif key == pygame.K_r:
            self.adjust_radius(5.0 if not shift else -5.0)
        elif key == pygame.K_e:
            self.adjust_eta(5.0 if not shift else -5.0)
        elif key == pygame.K_l:
            self.adjust_lease(30.0 if not shift else -30.0)
        elif key == pygame.K_d:
            self.adjust_max_deferrals(1 if not shift else -1)
        elif key == pygame.K_SLASH:
            self.adjust_robots(1)
        elif key == pygame.K_COMMA:
            self.adjust_robots(-1)
        elif key == pygame.K_SEMICOLON:
            self.adjust_arrival_rate(0.01 if not shift else -0.01)
        elif key == pygame.K_a:
            self.simulation_renderer_toggle("show_paths")
        elif key == pygame.K_c:
            self.simulation_renderer_toggle("show_claim_lines")
        elif key == pygame.K_y:
            self.simulation_renderer_toggle("show_radii")
        return True

    def simulation_renderer_toggle(self, attribute: str) -> None:
        """Flip a renderer display toggle.

        Display toggles change nothing about the simulation, so they are not
        tamper-marked and do not make the run non-comparable.

        Args:
            attribute: Renderer attribute name to flip.
        """
        if self.renderer is not None:
            setattr(self.renderer, attribute, not getattr(self.renderer, attribute))

    def _toggle_pause(self) -> None:
        """Toggle between paused and 1x speed."""
        if self.paused:
            self._speed_index = SPEEDS.index(1.0)
        else:
            self._speed_index = 0

    def cycle_policy(self) -> None:
        """Switch to the next admission policy, live."""
        previous = self.policy_name
        index = POLICY_CYCLE.index(previous) if previous in POLICY_CYCLE else 0
        self.policy_name = POLICY_CYCLE[(index + 1) % len(POLICY_CYCLE)]
        self.simulation.allocator.policy = build_policy(
            self.policy_name,
            radius=self.radius,
            eta_threshold=self.eta_threshold,
            lease_duration=self.lease_duration,
        )
        self._mark("policy", previous, self.policy_name)

    def adjust_radius(self, delta: float) -> None:
        """Change the admission radius.

        Args:
            delta: Change in metres.
        """
        previous = self.radius
        self.radius = max(1.0, self.radius + delta)
        policy = self.simulation.allocator.policy
        if isinstance(policy, RadiusPolicy):
            policy.radius = self.radius
        self._mark("radius", previous, self.radius)

    def adjust_eta(self, delta: float) -> None:
        """Change the ETA admission threshold.

        Args:
            delta: Change in simulated seconds.
        """
        previous = self.eta_threshold
        self.eta_threshold = max(1.0, self.eta_threshold + delta)
        policy = self.simulation.allocator.policy
        if isinstance(policy, ETAPolicy):
            policy.threshold = self.eta_threshold
        self._mark("eta_threshold", previous, self.eta_threshold)

    def adjust_lease(self, delta: float) -> None:
        """Change the lease duration.

        Args:
            delta: Change in simulated seconds.
        """
        previous = self.lease_duration
        self.lease_duration = max(5.0, self.lease_duration + delta)
        policy = self.simulation.allocator.policy
        if isinstance(policy, LeasePolicy):
            policy.duration = self.lease_duration
        self._mark("lease_duration", previous, self.lease_duration)

    def adjust_max_deferrals(self, delta: int) -> None:
        """Change the force-admission threshold.

        Args:
            delta: Change in deferral count.
        """
        previous = self.max_deferrals
        self.max_deferrals = max(0, self.max_deferrals + delta)
        self.simulation.allocator.max_deferrals = self.max_deferrals
        self._mark("max_deferrals", previous, self.max_deferrals)

    def adjust_robots(self, delta: int) -> None:
        """Add or remove a robot.

        Removal only ever takes an idle, empty-handed robot; removing a holder
        would strand its resources and corrupt the allocation matrices.

        Args:
            delta: ``+1`` to add, ``-1`` to remove.
        """
        world = self.simulation.world
        previous = len(world.robots)
        if delta > 0:
            world.add_robot()
        elif delta < 0 and previous > 1:
            if world.remove_idle_robot() is None:
                return
        else:
            return

        robot_ids = tuple(robot.robot_id for robot in world.robots)
        self.simulation.allocator.state = self.simulation.allocator.state.rebuild_for(
            robot_ids
        )
        self.robot_count = len(robot_ids)
        self._mark("robot_count", previous, self.robot_count)

    def adjust_arrival_rate(self, delta: float) -> None:
        """Change the task arrival rate for the remainder of the run.

        Args:
            delta: Change in tasks per simulated second.
        """
        previous = self.arrival_rate
        self.arrival_rate = max(0.005, self.arrival_rate + delta)
        self.simulation.rescale_future_arrivals(self.arrival_rate)
        self._mark("task_arrival_rate", previous, self.arrival_rate)
