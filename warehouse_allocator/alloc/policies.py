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
from warehouse_allocator.core.entities import Resource, Robot, RobotState
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
    """Closed-loop contention feedback governor with starvation-free aging.

    Synthesizes the advantages of classical and spatial policies:
    1. Slack Regime (Contention < threshold): behaves liberally like NullPolicy,
       allowing pipeline saturation when resources are not contested.
    2. Congested Regime (Contention >= threshold): modulates admission threshold
       dynamically using true A* path cost (ETAPolicy), tightening the threshold
       as contention increases to prevent early-binding lock hoarding.
    3. Anti-Starvation Aging: discounts effective ETA proportional to wait time,
       providing a formal upper bound on deferred wait times.
    4. Feasible Slack Leases: attaches dynamic arrival deadlines scaled to actual
       travel time + slack buffer, preventing lease thrashing while bounding idle holding.

    Attributes:
        base_eta: Maximum ETA threshold when entering congested mode.
        slack_threshold: Contention ratio below which admission is liberal.
        aging_rate: Discount rate applied to ETA per second of accumulated wait.
        lease_slack: Safety margin added to ETA for feasible leases.
        enable_leases: Whether to attach arrival deadlines.
    """

    name = "adaptive"

    __slots__ = (
        "_ema_contention",
        "_first_deferred",
        "aging_rate",
        "base_eta",
        "enable_leases",
        "last_contention",
        "last_regime",
        "last_threshold",
        "lease_slack",
        "slack_threshold",
    )

    def __init__(
        self,
        base_eta: float = 45.0,
        slack_threshold: float = 0.35,
        aging_rate: float = 0.5,
        lease_slack: float = 30.0,
        enable_leases: bool = True,
    ) -> None:
        """Initialize the adaptive policy.

        Args:
            base_eta: Maximum ETA threshold in seconds under moderate congestion.
            slack_threshold: Contention ratio separating slack and congested regimes.
            aging_rate: Simulated seconds of ETA discount awarded per simulated second of wait.
            lease_slack: Grace margin in simulated seconds added to estimated arrival time.
            enable_leases: Whether to issue dynamic feasible leases upon grant.
        """
        self.base_eta = base_eta
        self.slack_threshold = slack_threshold
        self.aging_rate = aging_rate
        self.lease_slack = lease_slack
        self.enable_leases = enable_leases
        self._ema_contention = 0.0
        self._first_deferred: dict[tuple[int, ResourceClass], float] = {}
        self.last_regime = "slack"
        self.last_contention = 0.0
        self.last_threshold = float("inf")

    def should_admit(
        self, robot: Robot, resource: Resource, world: World, clock: SimClock
    ) -> AdmissionDecision:
        """Admit or defer based on closed-loop contention feedback and aged ETA.

        Args:
            robot: The requesting robot.
            resource: The candidate resource.
            world: Supplies geometry and resource occupancy.
            clock: Current simulation clock.

        Returns:
            The admission decision with dynamic lease deadline if admitted.
        """
        now = clock.now()

        # 1. True bottleneck contention: unreached lock ratio weighted by waiting queue pressure.
        # If no robots are blocked, transit reservations are benign pipeline saturation.
        # If robots are waiting while resources sit locked and empty, that is harmful early binding.
        held = sum(1 for r in world.resources if r.holders)
        unreached = sum(1 for r in world.resources if r.held_but_unused)
        waiting = sum(1 for r in world.robots if r.state is RobotState.WAITING)
        queue_pressure = waiting / max(len(world.robots), 1)
        instant_contention = queue_pressure * (unreached / max(held, 1))

        # 2. Smooth via Exponential Moving Average (EMA) to avoid high-frequency jitter
        self._ema_contention = 0.85 * self._ema_contention + 0.15 * instant_contention
        self.last_contention = self._ema_contention

        # 3. Path cost & True ETA
        cost = path_cost(world.grid, robot.cell, resource.cell)
        if cost is None:
            return AdmissionDecision(Verdict.DEFER, "no path to resource")
        true_eta = cost / world.config.robot_speed

        # 4. Anti-Starvation Aging Gradient
        req_key = (robot.robot_id, resource.resource_class)
        first_time = self._first_deferred.setdefault(req_key, now)
        wait_time = max(0.0, now - first_time)
        effective_eta = max(0.0, true_eta - (self.aging_rate * wait_time))

        # 5. Dual-Regime Decision
        if self._ema_contention < self.slack_threshold:
            self.last_regime = "slack"
            self.last_threshold = float("inf")
            self._first_deferred.pop(req_key, None)
            deadline = (now + true_eta + self.lease_slack) if (self.enable_leases and robot.state is not RobotState.WAITING) else None
            return AdmissionDecision(
                Verdict.ADMIT,
                f"slack regime (contention={self._ema_contention:.2f})",
                lease_deadline=deadline,
            )

        # Congested regime: Dynamic threshold shrinks as contention escalates
        self.last_regime = "congested"
        penalty = min(
            1.0,
            (self._ema_contention - self.slack_threshold)
            / max(0.01, 1.0 - self.slack_threshold),
        )
        dynamic_threshold = self.base_eta * (1.0 - 0.6 * penalty)
        self.last_threshold = dynamic_threshold

        if effective_eta <= dynamic_threshold:
            self._first_deferred.pop(req_key, None)
            deadline = (now + true_eta + self.lease_slack) if (self.enable_leases and robot.state is not RobotState.WAITING) else None
            reason = (
                f"congested admit: eta {true_eta:.1f}s (aged {effective_eta:.1f}s) "
                f"<= {dynamic_threshold:.1f}s"
            )
            return AdmissionDecision(
                Verdict.ADMIT,
                reason,
                lease_deadline=deadline,
            )

        return AdmissionDecision(
            Verdict.DEFER,
            f"congested defer: eta {true_eta:.1f}s (aged {effective_eta:.1f}s) "
            f"> {dynamic_threshold:.1f}s (contention={self._ema_contention:.2f})",
        )

    def describe(self) -> str:
        """Return the policy name and its current operational regime.

        Returns:
            A short description for display.
        """
        return f"adaptive(regime={self.last_regime}, rho={self.last_contention:.2f})"


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
        radius: Threshold for :class:`RadiusPolicy`.
        eta_threshold: Threshold for :class:`ETAPolicy` or base ETA for :class:`AdaptivePolicy`.
        lease_duration: Duration for :class:`LeasePolicy`.

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
                base_eta=max(45.0, eta_threshold * 1.5),
                slack_threshold=0.35,
                aging_rate=0.4,
                lease_slack=max(35.0, lease_duration / 8.0),
            )
        case _:
            raise ValueError(
                f"unknown policy {name!r}; expected null, radius, eta, lease, or adaptive"
            )
