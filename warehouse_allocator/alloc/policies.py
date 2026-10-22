"""Admission policies: the layer in front of the safety check.

Classical resource allocation assumes a process granted a resource uses it
immediately. A warehouse robot may be granted a crane while 200 metres away
from it, and the crane then sits locked and idle for the whole travel time
while nearer robots starve. That is early binding, and the admission layer
exists to attack it.

Each policy answers one question: *should this request even be considered right
now, given where the robot is?* A policy can only ever defer a request. It can
never grant one -- that is the safety checker's decision, and it is never
bypassed.

The four policies are the experiment. Building only one produces no evidence
that its rule is the right one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from warehouse_allocator.core.clock import SimClock
from warehouse_allocator.core.entities import (
    Resource,
    ResourceClass,
    Robot,
    RobotState,
)
from warehouse_allocator.core.geometry import euclidean_distance, path_cost
from warehouse_allocator.core.world import World

__all__ = [
    "AdaptivePolicy",
    "AdmissionDecision",
    "AdmissionPolicy",
    "ETAPolicy",
    "LeasePolicy",
    "NullPolicy",
    "RadiusPolicy",
    "Verdict",
    "build_policy",
]


class Verdict(Enum):
    """The two things an admission policy may decide.

    Attributes:
        ADMIT: Pass the request to the safety check.
        DEFER: Hold the request back for now.
    """

    ADMIT = "admit"
    DEFER = "defer"


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """A policy's verdict plus the reasoning behind it.

    Attributes:
        verdict: Whether to admit or defer.
        reason: Human-readable justification, recorded in the event log so a
            surprising result can be traced back to the rule that produced it.
        lease_deadline: For lease-based policies, the simulated time by which
            the robot must arrive or lose the resource. ``None`` for policies
            that do not lease.
    """

    verdict: Verdict
    reason: str
    lease_deadline: float | None = None

    @property
    def admitted(self) -> bool:
        """Report whether the policy admitted the request.

        Returns:
            ``True`` if the verdict is :attr:`Verdict.ADMIT`.
        """
        return self.verdict is Verdict.ADMIT


class AdmissionPolicy(ABC):
    """Decides whether a request should be considered now.

    Attributes:
        name: Short identifier used in configuration and reporting.
    """

    name: str = "abstract"

    @abstractmethod
    def should_admit(
        self, robot: Robot, resource: Resource, world: World, clock: SimClock
    ) -> AdmissionDecision:
        """Decide whether to admit a request.

        Args:
            robot: The requesting robot, which may already hold resources.
            resource: The specific instance the allocator would bind.
            world: The world, for geometry and pathfinding.
            clock: The simulation clock, for lease deadlines.

        Returns:
            The decision, carrying a reason for the log.
        """
        ...

    def describe(self) -> str:
        """Return a human-readable description of the policy's parameters.

        Returns:
            A short string for figure captions and HUD display.
        """
        return self.name


class NullPolicy(AdmissionPolicy):
    """Admits everything. The classical baseline.

    Reproduces textbook behaviour, in which spatial position is invisible to
    the allocator. Its failure mode is the full early-binding waste the project
    measures: a robot is granted a resource the instant it is free, however far
    away it happens to be.
    """

    name = "null"

    def should_admit(
        self, robot: Robot, resource: Resource, world: World, clock: SimClock
    ) -> AdmissionDecision:
        """Admit unconditionally.

        Args:
            robot: The requesting robot.
            resource: The candidate resource.
            world: Unused.
            clock: Unused.

        Returns:
            An unconditional admit.
        """
        return AdmissionDecision(Verdict.ADMIT, "baseline admits all")


class RadiusPolicy(AdmissionPolicy):
    """Admits only requesters within a straight-line distance.

    This is the originally proposed idea. Its failure mode is that euclidean
    distance ignores topology: a robot 20 metres away behind a wall of racking
    has a far worse arrival time than one 200 metres down a clear aisle, yet
    this policy prefers the former. :class:`ETAPolicy` exists to correct
    exactly that.

    Attributes:
        radius: Admission threshold in metres.
    """

    name = "radius"

    __slots__ = ("radius",)

    def __init__(self, radius: float) -> None:
        """Initialize the policy.

        Args:
            radius: Maximum admissible euclidean distance in metres.
        """
        self.radius = radius

    def should_admit(
        self, robot: Robot, resource: Resource, world: World, clock: SimClock
    ) -> AdmissionDecision:
        """Admit if the requester is within :attr:`radius`.

        Args:
            robot: The requesting robot.
            resource: The candidate resource.
            world: Unused; straight-line distance ignores the map, which is
                this policy's defining weakness.
            clock: Unused.

        Returns:
            The decision, with the measured distance in its reason.
        """
        distance = euclidean_distance(robot.position, resource.position)
        if distance <= self.radius:
            return AdmissionDecision(
                Verdict.ADMIT, f"distance {distance:.1f}m <= r {self.radius:.1f}m"
            )
        return AdmissionDecision(
            Verdict.DEFER, f"distance {distance:.1f}m > r {self.radius:.1f}m"
        )

    def describe(self) -> str:
        """Return the policy name and its radius.

        Returns:
            A short description for display.
        """
        return f"radius(r={self.radius:.0f}m)"


class ETAPolicy(AdmissionPolicy):
    r"""Admits only requesters whose true travel time is short enough.

    Uses **actual A\* path cost**, not straight-line distance. That
    distinction is the entire reason this policy exists alongside
    :class:`RadiusPolicy`: it sees walls, and the radius rule does not.

    Its own failure mode is cost. A* runs per candidate request, which is far
    more expensive than a distance comparison and scales badly with fleet size.

    Attributes:
        threshold: Admission threshold in simulated seconds of travel.
    """

    name = "eta"

    __slots__ = ("threshold",)

    def __init__(self, threshold: float) -> None:
        """Initialize the policy.

        Args:
            threshold: Maximum admissible travel time in simulated seconds.
        """
        self.threshold = threshold

    def should_admit(
        self, robot: Robot, resource: Resource, world: World, clock: SimClock
    ) -> AdmissionDecision:
        """Admit if A* travel time is within :attr:`threshold`.

        Args:
            robot: The requesting robot.
            resource: The candidate resource.
            world: Supplies the grid the path is computed over.
            clock: Unused.

        Returns:
            The decision, with the estimated arrival time in its reason.
        """
        cost = path_cost(world.grid, robot.cell, resource.cell)
        if cost is None:
            return AdmissionDecision(Verdict.DEFER, "no path to resource")
        eta = cost / world.config.robot_speed
        if eta <= self.threshold:
            return AdmissionDecision(
                Verdict.ADMIT, f"eta {eta:.1f}s <= {self.threshold:.1f}s"
            )
        return AdmissionDecision(
            Verdict.DEFER, f"eta {eta:.1f}s > {self.threshold:.1f}s"
        )

    def describe(self) -> str:
        """Return the policy name and its threshold.

        Returns:
            A short description for display.
        """
        return f"eta(t={self.threshold:.0f}s)"


class LeasePolicy(AdmissionPolicy):
    """Admits everyone but attaches an arrival deadline.

    The theoretically correct fix. Radius and ETA attack early binding by
    proxy, guessing in advance who will arrive promptly and refusing everyone
    else. Leasing attacks it directly: grant to anyone, then reclaim the
    resource if the holder has not actually shown up in time. No requester is
    pre-emptively excluded, and a resource can never sit claimed-but-unreached
    for longer than the lease.

    Its failure mode is revocation churn: under heavy contention a resource can
    be granted and revoked repeatedly, doing no useful work in between.

    Attributes:
        duration: Lease length in simulated seconds.
    """

    name = "lease"

    __slots__ = ("duration",)

    def __init__(self, duration: float) -> None:
        """Initialize the policy.

        Args:
            duration: Simulated seconds a holder has to arrive.
        """
        self.duration = duration

    def should_admit(
        self, robot: Robot, resource: Resource, world: World, clock: SimClock
    ) -> AdmissionDecision:
        """Admit with an arrival deadline attached.

        Args:
            robot: The requesting robot.
            resource: The candidate resource.
            world: Unused; leasing deliberately does not filter on position.
            clock: Supplies the current time the deadline is measured from.

        Returns:
            An admit carrying a lease deadline.
        """
        deadline = clock.now() + self.duration
        return AdmissionDecision(
            Verdict.ADMIT,
            f"leased until {deadline:.1f}s",
            lease_deadline=deadline,
        )

    def describe(self) -> str:
        """Return the policy name and its lease duration.

        Returns:
            A short description for display.
        """
        return f"lease(d={self.duration:.0f}s)"


class AdaptivePolicy(AdmissionPolicy):
    """Dynamic admission policy adapting to real-time warehouse contention.

    Combines fast-path immediate admission, spatial ETA filtering, and
    ETA-tailored arrival deadlines inspired by OS adaptive locks and congestion
    control:

    1. FAST-PATH (Low Contention):
       When plenty of instances of the requested class are free (>= 3 free),
       requests are admitted unconditionally. When resources are uncontested,
       early binding causes no interference, avoiding unnecessary deferrals.

    2. SPATIAL ETA FILTERING (Moderate Contention):
       When spare capacity narrows (e.g. 2 instances free), requesters must
       be within a relaxed ETA threshold (1.5x base threshold).

    3. BOTTLENECK PROTECTION WITH DYNAMIC LEASES (High Contention):
       When only 1 instance remains of a scarce bottleneck (cranes or elevators),
       the requester must pass a tightened ETA threshold. Admitted requesters
       receive an ETA-tailored lease deadline (now + eta + slack)
       preventing early-binding lockup without suffering false-revocation thrashing.

    4. HETEROGENEOUS RESOURCE-CLASS ROUTING:
       High-turnover docks with plentiful instances are exempted from short
       arrival deadlines, eliminating the revocation thrashing observed under
       static leasing.
    """

    name = "adaptive"

    __slots__ = ("_last_regime", "eta_threshold", "lease_duration", "radius")

    def __init__(
        self,
        *,
        eta_threshold: float = 30.0,
        lease_duration: float = 300.0,
        radius: float = 40.0,
    ) -> None:
        """Initialize the adaptive policy.

        Args:
            eta_threshold: Base ETA admission threshold in seconds.
            lease_duration: Base lease duration in seconds.
            radius: Base radius threshold in metres.
        """
        self.eta_threshold = eta_threshold
        self.lease_duration = lease_duration
        self.radius = radius
        self._last_regime: str = "fast_path"

    @property
    def threshold(self) -> float:
        """Alias for eta_threshold to support LiveControls.adjust_eta."""
        return self.eta_threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self.eta_threshold = value

    @property
    def duration(self) -> float:
        """Alias for lease_duration to support LiveControls.adjust_lease."""
        return self.lease_duration

    @duration.setter
    def duration(self, value: float) -> None:
        self.lease_duration = value

    def should_admit(
        self, robot: Robot, resource: Resource, world: World, clock: SimClock
    ) -> AdmissionDecision:
        """Admit or defer based on real-time contention and estimated arrival time.

        Args:
            robot: The requesting robot.
            resource: The candidate resource instance.
            world: The simulation world, supplying grid topology and entity states.
            clock: The simulation clock.

        Returns:
            An admission decision reflecting the active adaptive regime.
        """
        now = clock.now()
        r_class = resource.resource_class
        all_instances = world.resources_of_class(r_class)
        total = len(all_instances)
        busy = sum(1 for r in all_instances if r.holders)
        free = total - busy

        # Compute true A* travel time to the candidate resource.
        cost = path_cost(world.grid, robot.cell, resource.cell)
        if cost is None:
            self._last_regime = "unreachable"
            return AdmissionDecision(Verdict.DEFER, "no path to resource")

        eta = cost / world.config.robot_speed

        # Special handling for DOCK (high turnover, charging/dropoff point)
        if r_class is ResourceClass.DOCK:
            if free >= 1:
                self._last_regime = "dock_open"
                return AdmissionDecision(
                    Verdict.ADMIT, f"adaptive: dock available ({free}/{total} free)"
                )
            dock_thresh = self.eta_threshold * 1.5
            if eta <= dock_thresh:
                self._last_regime = "dock_contested"
                reason = (
                    f"adaptive: dock busy ({busy}/{total}), "
                    f"eta {eta:.1f}s <= {dock_thresh:.1f}s"
                )
                return AdmissionDecision(Verdict.ADMIT, reason)
            self._last_regime = "dock_saturated"
            reason = (
                f"adaptive: dock busy ({busy}/{total}), "
                f"eta {eta:.1f}s > {dock_thresh:.1f}s"
            )
            return AdmissionDecision(Verdict.DEFER, reason)

        # Regimes for Scarce Multi-Resource Bottlenecks (Cranes & Elevators)
        if free >= 3:
            # Regime 1: Uncontested Fast-Path
            self._last_regime = "fast_path"
            return AdmissionDecision(
                Verdict.ADMIT,
                f"adaptive: fast-path {r_class.value} ({free}/{total} free)",
            )

        if free == 2:
            # Regime 2: Moderate Contention (Relaxed ETA)
            relaxed_thresh = self.eta_threshold * 1.5
            if eta <= relaxed_thresh:
                self._last_regime = "relaxed_eta"
                reason = (
                    f"adaptive: {r_class.value} "
                    f"eta {eta:.1f}s <= {relaxed_thresh:.1f}s"
                )
                return AdmissionDecision(Verdict.ADMIT, reason)
            self._last_regime = "relaxed_defer"
            reason = (
                f"adaptive: {r_class.value} " f"eta {eta:.1f}s > {relaxed_thresh:.1f}s"
            )
            return AdmissionDecision(Verdict.DEFER, reason)

        # Regime 3: Severe Contention (free <= 1) - Last remaining instance!
        waiting_count = sum(
            1
            for r in world.robots
            if r.state is RobotState.WAITING and r.current_task_id is not None
        )
        tight_thresh = (
            max(15.0, self.eta_threshold * 0.75)
            if waiting_count >= 3
            else self.eta_threshold
        )

        if eta <= tight_thresh:
            # Admitted with an ETA-tailored dynamic lease:
            slack = max(30.0, eta)
            deadline = now + eta + slack
            dur = deadline - now
            self._last_regime = "bottleneck_lease"
            reason = (
                f"adaptive: {r_class.value} "
                f"(eta {eta:.1f}s <= {tight_thresh:.1f}s, lease {dur:.0f}s)"
            )
            return AdmissionDecision(
                Verdict.ADMIT,
                reason,
                lease_deadline=deadline,
            )

        self._last_regime = "bottleneck_defer"
        reason = (
            f"adaptive: {r_class.value} "
            f"bottleneck, eta {eta:.1f}s > {tight_thresh:.1f}s"
        )
        return AdmissionDecision(Verdict.DEFER, reason)

    def describe(self) -> str:
        """Return the policy name and its active regime.

        Returns:
            A descriptive string for display.
        """
        return f"adaptive({self._last_regime})"


def build_policy(
    name: str,
    *,
    radius: float,
    eta_threshold: float,
    lease_duration: float,
) -> AdmissionPolicy:
    """Construct a policy by name.

    Args:
        name: One of ``null``, ``radius``, ``eta``, ``lease``, ``adaptive``.
        radius: Threshold for :class:`RadiusPolicy` or :class:`AdaptivePolicy`.
        eta_threshold: Threshold for :class:`ETAPolicy` or :class:`AdaptivePolicy`.
        lease_duration: Duration for :class:`LeasePolicy` or :class:`AdaptivePolicy`.

    Returns:
        The constructed policy.

    Raises:
        ValueError: If ``name`` is not a known policy.
    """
    match name:
        case "null":
            return NullPolicy()
        case "radius":
            return RadiusPolicy(radius)
        case "eta":
            return ETAPolicy(eta_threshold)
        case "lease":
            return LeasePolicy(lease_duration)
        case "adaptive":
            return AdaptivePolicy(
                eta_threshold=eta_threshold,
                lease_duration=lease_duration,
                radius=radius,
            )
        case _:
            raise ValueError(
                f"unknown policy {name!r}; "
                "expected null, radius, eta, lease, or adaptive"
            )
