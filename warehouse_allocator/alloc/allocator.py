"""The two-layer allocator: admission control in front of Banker's.

Request flow, in order::

    1. AdmissionPolicy.should_admit()
    2. If deferred: increment deferral_count, enqueue
    3. If deferral_count > max_deferrals: force-admit (skip policy, NOT safety)
    4. SafetyChecker.is_safe()
    5. If safe: grant. If unsafe: defer regardless of admission verdict.

Every decision emits a structured event carrying the simulated timestamp, robot
id, resource class, policy verdict, policy reason, safety verdict, and deferral
count.
"""

from __future__ import annotations

from warehouse_allocator.alloc.leases import Lease, LeaseBook
from warehouse_allocator.alloc.policies import AdmissionPolicy, LeasePolicy
from warehouse_allocator.alloc.safety import SafetyChecker
from warehouse_allocator.alloc.state import ResourceState
from warehouse_allocator.core.clock import SimClock
from warehouse_allocator.core.entities import (
    Resource,
    ResourceClass,
    Robot,
    RobotState,
)
from warehouse_allocator.core.geometry import euclidean_distance
from warehouse_allocator.core.tasks import Task
from warehouse_allocator.core.world import World
from warehouse_allocator.metrics.collector import EventCollector
from warehouse_allocator.metrics.events import (
    LeaseGranted,
    LeaseRevoked,
    RequestDeferred,
    RequestForceAdmitted,
    RequestGranted,
    RequestIssued,
    RequestRefusedUnsafe,
    ResourceOccupied,
    ResourceReleased,
    RobotStateChanged,
)

__all__ = ["Allocator"]


class Allocator:
    """Orchestrates admission, safety, granting, and lease revocation.

    Attributes:
        world: The world whose resources are allocated.
        state: The Banker's matrices.
        policy: The admission policy in front of the safety check.
        safety: The safety checker.
        leases: Live leases, when the policy issues them.
        clock: The simulation clock.
        collector: Event sink.
        max_deferrals: Deferrals tolerated before force-admission.
    """

    __slots__ = (
        "_deferrals",
        "clock",
        "collector",
        "leases",
        "max_deferrals",
        "policy",
        "safety",
        "state",
        "world",
    )

    def __init__(
        self,
        world: World,
        state: ResourceState,
        policy: AdmissionPolicy,
        clock: SimClock,
        collector: EventCollector,
        max_deferrals: int,
        safety: SafetyChecker | None = None,
    ) -> None:
        """Initialize the allocator.

        Args:
            world: World supplying resource instances and geometry.
            state: Banker's matrices for this run.
            policy: Admission policy to consult.
            clock: Simulation clock.
            collector: Append-only event sink.
            max_deferrals: Deferrals a request may accumulate before the
                admission policy is bypassed.
            safety: Safety checker; a default one is built if omitted.
        """
        self.world = world
        self.state = state
        self.policy = policy
        self.clock = clock
        self.collector = collector
        self.max_deferrals = max_deferrals
        self.safety = safety if safety is not None else SafetyChecker()
        self.leases = LeaseBook()
        self._deferrals: dict[tuple[int, ResourceClass], int] = {}

    @property
    def deferral_queue_depth(self) -> int:
        """Return how many requests are currently carrying deferrals.

        Returns:
            Number of distinct deferred request keys.
        """
        return len(self._deferrals)

    def deferral_count(self, robot_id: int, resource_class: ResourceClass) -> int:
        """Return the deferrals accumulated by one pending request.

        Args:
            robot_id: The requesting robot.
            resource_class: The class requested.

        Returns:
            Deferral count, zero if the request has never been deferred.
        """
        return self._deferrals.get((robot_id, resource_class), 0)

    def declare_task(self, robot: Robot, task: Task) -> None:
        """Declare a robot's maximum claim as it accepts a task.

        Args:
            robot: The accepting robot.
            task: The accepted task, whose bundle becomes the claim.
        """
        self.state.declare_max(robot.robot_id, task.required_bundle)

    def request(
        self, robot: Robot, resource_class: ResourceClass, task: Task
    ) -> Resource | None:
        """Run one request through admission and safety.

        Args:
            robot: The requesting robot, which may already hold resources.
            resource_class: The class being requested.
            task: The task the request serves.

        Returns:
            The bound resource if granted, otherwise ``None``.
        """
        now = self.clock.now()
        key = (robot.robot_id, resource_class)
        deferrals = self._deferrals.get(key, 0)

        self.collector.record(
            RequestIssued(
                at=now,
                robot_id=robot.robot_id,
                task_id=task.task_id,
                resource_class=resource_class,
                deferral_count=deferrals,
            )
        )

        candidate = self._select_instance(robot, resource_class)
        if candidate is None:
            self._defer(key, task, resource_class, "no free instance", "unavailable")
            return None

        forced = deferrals > self.max_deferrals
        if forced:
            # Force-admission skips the ADMISSION POLICY ONLY. The safety
            # check below is still executed, unconditionally. Bypassing it
            # would reintroduce deadlock, which is the single failure mode
            # this allocator exists to prevent. Do not "optimise" the
            # is_safe call away for forced requests.
            self.collector.record(
                RequestForceAdmitted(
                    at=now,
                    robot_id=robot.robot_id,
                    task_id=task.task_id,
                    resource_class=resource_class,
                    deferral_count=deferrals,
                )
            )
            reason = f"force-admitted after {deferrals} deferrals"
            lease_deadline = None
        else:
            decision = self.policy.should_admit(
                robot, candidate, self.world, self.clock
            )
            if not decision.admitted:
                self._defer(
                    key, task, resource_class, decision.reason, decision.verdict.value
                )
                return None
            reason = decision.reason
            lease_deadline = decision.lease_deadline

        request_vector = self.state.request_vector(resource_class)

        # Reached for every request, forced or not. See the comment above.
        if not self.state.can_grant(robot.robot_id, request_vector) or not (
            self.safety.is_safe(self.state, robot.robot_id, request_vector)
        ):
            self.collector.record(
                RequestRefusedUnsafe(
                    at=now,
                    robot_id=robot.robot_id,
                    task_id=task.task_id,
                    resource_class=resource_class,
                    forced=forced,
                )
            )
            self._defer(key, task, resource_class, "unsafe state", "refused_unsafe")
            return None

        return self._grant(
            robot,
            candidate,
            task,
            deferrals,
            forced,
            reason,
            lease_deadline=lease_deadline,
        )

    def _grant(
        self,
        robot: Robot,
        resource: Resource,
        task: Task,
        deferrals: int,
        forced: bool,
        reason: str,
        lease_deadline: float | None = None,
    ) -> Resource:
        """Commit a grant to matrices, entities, leases, and the log.

        Args:
            robot: The granted robot.
            resource: The instance being bound.
            task: The task the grant serves.
            deferrals: Deferrals this request accumulated.
            forced: Whether the admission policy was bypassed.
            reason: Justification for the log.
            lease_deadline: Optional arrival deadline for lease tracking.

        Returns:
            The bound resource.
        """
        now = self.clock.now()
        self.state.grant(
            robot.robot_id, self.state.request_vector(resource.resource_class)
        )
        resource.holders.add(robot.robot_id)
        robot.held_resources.add(resource.resource_id)
        self._deferrals.pop((robot.robot_id, resource.resource_class), None)

        self.collector.record(
            RequestGranted(
                at=now,
                robot_id=robot.robot_id,
                task_id=task.task_id,
                resource_class=resource.resource_class,
                resource_id=resource.resource_id,
                deferral_count=deferrals,
                forced=forced,
                reason=reason,
            )
        )

        deadline = lease_deadline
        if deadline is None and isinstance(self.policy, LeasePolicy):
            deadline = now + self.policy.duration

        if deadline is not None:
            self.leases.grant(
                Lease(
                    robot_id=robot.robot_id,
                    resource_id=resource.resource_id,
                    resource_class=resource.resource_class,
                    granted_at=now,
                    deadline=deadline,
                )
            )
            self.collector.record(
                LeaseGranted(
                    at=now,
                    robot_id=robot.robot_id,
                    resource_id=resource.resource_id,
                    resource_class=resource.resource_class,
                    deadline=deadline,
                )
            )
        return resource

    def _defer(
        self,
        key: tuple[int, ResourceClass],
        task: Task,
        resource_class: ResourceClass,
        reason: str,
        verdict: str,
    ) -> None:
        """Record a deferral and increment the request's counter.

        Args:
            key: The ``(robot_id, resource_class)`` request key.
            task: The task the request serves.
            resource_class: The class requested.
            reason: Justification for the log.
            verdict: The verdict label to record.
        """
        count = self._deferrals.get(key, 0) + 1
        self._deferrals[key] = count
        self.collector.record(
            RequestDeferred(
                at=self.clock.now(),
                robot_id=key[0],
                task_id=task.task_id,
                resource_class=resource_class,
                deferral_count=count,
                policy_verdict=verdict,
                reason=reason,
            )
        )

    def _select_instance(
        self, robot: Robot, resource_class: ResourceClass
    ) -> Resource | None:
        """Choose which instance of a class to offer the requester.

        The nearest instance with spare capacity, breaking ties on resource id
        so the choice is reproducible.

        Args:
            robot: The requesting robot.
            resource_class: The class requested.

        Returns:
            The chosen instance, or ``None`` if all are full.
        """
        candidates = [
            resource
            for resource in self.world.resources_of_class(resource_class)
            if not resource.is_full and robot.robot_id not in resource.holders
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                euclidean_distance(robot.position, item.position),
                item.resource_id,
            ),
        )

    def on_arrival(self, robot: Robot, resource: Resource) -> None:
        """Mark a holder as having physically reached its resource.

        This closes the early-binding window: from here the resource counts as
        used rather than merely held. It also discharges any lease, since the
        deadline it enforced has been met.

        Args:
            robot: The arriving robot.
            resource: The resource reached.
        """
        if robot.robot_id in resource.occupants:
            return
        resource.occupants.add(robot.robot_id)
        self.leases.discharge(robot.robot_id, resource.resource_id)
        self.collector.record(
            ResourceOccupied(
                at=self.clock.now(),
                robot_id=robot.robot_id,
                resource_id=resource.resource_id,
                resource_class=resource.resource_class,
            )
        )

    def release_all(self, robot: Robot) -> None:
        """Release a robot's entire bundle atomically.

        Args:
            robot: The releasing robot.
        """
        now = self.clock.now()
        for resource_id in sorted(robot.held_resources):
            resource = self.world.resource(resource_id)
            was_occupied = robot.robot_id in resource.occupants
            self.state.release(
                robot.robot_id,
                self.state.request_vector(resource.resource_class),
            )
            resource.holders.discard(robot.robot_id)
            resource.occupants.discard(robot.robot_id)
            self.collector.record(
                ResourceReleased(
                    at=now,
                    robot_id=robot.robot_id,
                    resource_id=resource_id,
                    resource_class=resource.resource_class,
                    was_occupied=was_occupied,
                )
            )
        robot.held_resources.clear()
        self.leases.discharge_robot(robot.robot_id)
        for resource_class in self.state.classes:
            self._deferrals.pop((robot.robot_id, resource_class), None)
        self.state.clear_max(robot.robot_id)

    def tick(self) -> None:
        """Expire overdue leases. Called once per simulated timestep."""
        now = self.clock.now()
        for lease in self.leases.expired(now):
            resource = self.world.resource(lease.resource_id)
            # A holder that already arrived has met the deadline; the lease is
            # simply retired rather than revoked.
            if lease.robot_id in resource.occupants:
                self.leases.discharge(lease.robot_id, lease.resource_id)
                continue
            self._revoke(lease, resource, now)

    def _revoke(self, lease: Lease, resource: Resource, now: float) -> None:
        """Return a leased resource to the pool without disturbing the rest.

        The robot keeps every other resource in its partial bundle, and its
        ``Need`` for the revoked class is restored, because it still intends to
        finish the task and will request the class again.

        Args:
            lease: The expired lease.
            resource: The resource being reclaimed.
            now: Current simulated time.
        """
        robot = self.world.robot(lease.robot_id)
        self.leases.discharge(lease.robot_id, lease.resource_id)
        self.state.release(
            lease.robot_id, self.state.request_vector(lease.resource_class)
        )
        resource.holders.discard(lease.robot_id)
        resource.occupants.discard(lease.robot_id)
        robot.held_resources.discard(lease.resource_id)

        # The robot must ask for this class again, so its acquisition cursor
        # steps back by one. Everything already granted before it stays held.
        if robot.bundle_cursor > 0:
            robot.bundle_cursor -= 1

        # A robot already travelling on a now-incomplete bundle has to stop and
        # re-acquire. Emitting the transition here keeps the event log the sole
        # authority on the state timeline.
        if robot.state is not RobotState.WAITING:
            previous = robot.state
            robot.state = RobotState.WAITING
            robot.clear_path()
            robot.destination = None
            robot.visit_cursor = 0
            robot.service_remaining = 0.0
            self.collector.record(
                RobotStateChanged(
                    at=now,
                    robot_id=robot.robot_id,
                    previous=previous,
                    current=RobotState.WAITING,
                )
            )

        self.collector.record(
            LeaseRevoked(
                at=now,
                robot_id=lease.robot_id,
                resource_id=lease.resource_id,
                resource_class=lease.resource_class,
                held_for=now - lease.granted_at,
            )
        )
