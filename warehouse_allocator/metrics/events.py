"""Immutable event types describing everything that happens in a run.

Events are the only record of a run. Every derived statistic in
:mod:`warehouse_allocator.metrics.analysis` is computed from a list of these
after the run finishes, so a new question can be answered from an old log
without re-simulating.

All events are frozen and carry ``at``, a *simulated* timestamp. Wall-clock
time never appears: it would make two runs of the same seed differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from warehouse_allocator.core.entities import ResourceClass, RobotState
from warehouse_allocator.core.tasks import TaskTemplate

__all__ = [
    "Event",
    "LeaseGranted",
    "LeaseRevoked",
    "ParameterChanged",
    "RequestDeferred",
    "RequestForceAdmitted",
    "RequestGranted",
    "RequestIssued",
    "RequestRefusedUnsafe",
    "ResourceOccupied",
    "ResourceReleased",
    "RobotStateChanged",
    "TaskCompleted",
    "TaskCreated",
]


@dataclass(frozen=True, slots=True)
class Event:
    """Base class for every simulation event.

    Attributes:
        at: Simulated time the event occurred, in seconds.
    """

    at: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event including its concrete type name.

        Returns:
            A JSON-compatible dict with a ``kind`` discriminator.
        """
        payload: dict[str, Any] = {"kind": type(self).__name__}
        for field_name in self.__slots__:
            value = getattr(self, field_name)
            payload[field_name] = (
                str(value)
                if isinstance(value, ResourceClass | RobotState | TaskTemplate)
                else value
            )
        return payload


@dataclass(frozen=True, slots=True)
class TaskCreated(Event):
    """A task arrived from the Poisson process.

    Attributes:
        task_id: The arriving task.
        template: Which template it instantiates.
    """

    task_id: int
    template: TaskTemplate


@dataclass(frozen=True, slots=True)
class RequestIssued(Event):
    """A robot asked the allocator for one resource class.

    Attributes:
        robot_id: The requesting robot.
        task_id: The task the request serves.
        resource_class: The class requested.
        deferral_count: Deferrals accumulated so far for this request.
    """

    robot_id: int
    task_id: int
    resource_class: ResourceClass
    deferral_count: int


@dataclass(frozen=True, slots=True)
class RequestDeferred(Event):
    """A request was not granted on this attempt.

    Attributes:
        robot_id: The requesting robot.
        task_id: The task the request serves.
        resource_class: The class requested.
        deferral_count: Deferrals accumulated including this one.
        policy_verdict: What the admission policy decided.
        reason: Human-readable justification for logging.
    """

    robot_id: int
    task_id: int
    resource_class: ResourceClass
    deferral_count: int
    policy_verdict: str
    reason: str


@dataclass(frozen=True, slots=True)
class RequestGranted(Event):
    """A request passed admission and the safety check.

    Attributes:
        robot_id: The granted robot.
        task_id: The task the grant serves.
        resource_class: The class granted.
        resource_id: The specific instance bound to the robot.
        deferral_count: Deferrals this request accumulated before success.
        forced: Whether the admission policy was bypassed by force-admission.
        reason: The admission policy's justification.
    """

    robot_id: int
    task_id: int
    resource_class: ResourceClass
    resource_id: int
    deferral_count: int
    forced: bool
    reason: str


@dataclass(frozen=True, slots=True)
class RequestForceAdmitted(Event):
    """A starving request bypassed the admission policy.

    The safety check is never bypassed; this event records only that the
    *admission* layer was skipped.

    Attributes:
        robot_id: The force-admitted robot.
        task_id: The task the request serves.
        resource_class: The class requested.
        deferral_count: Deferrals accumulated before forcing.
    """

    robot_id: int
    task_id: int
    resource_class: ResourceClass
    deferral_count: int


@dataclass(frozen=True, slots=True)
class RequestRefusedUnsafe(Event):
    """The Banker's safety check refused a request.

    Each of these is a deadlock that did not happen.

    Attributes:
        robot_id: The refused robot.
        task_id: The task the request serves.
        resource_class: The class requested.
        forced: Whether this request had been force-admitted past the policy.
    """

    robot_id: int
    task_id: int
    resource_class: ResourceClass
    forced: bool


@dataclass(frozen=True, slots=True)
class LeaseGranted(Event):
    """A lease with an arrival deadline was issued.

    Attributes:
        robot_id: The leaseholder.
        resource_id: The leased resource.
        resource_class: Its class.
        deadline: Simulated time by which the robot must arrive.
    """

    robot_id: int
    resource_id: int
    resource_class: ResourceClass
    deadline: float


@dataclass(frozen=True, slots=True)
class LeaseRevoked(Event):
    """A leaseholder failed to arrive and the resource returned to the pool.

    Attributes:
        robot_id: The robot that lost the resource.
        resource_id: The revoked resource.
        resource_class: Its class.
        held_for: Simulated seconds the robot held it before revocation.
    """

    robot_id: int
    resource_id: int
    resource_class: ResourceClass
    held_for: float


@dataclass(frozen=True, slots=True)
class ResourceOccupied(Event):
    """A holder physically arrived at a resource.

    The interval between the grant and this event is early-binding waste.

    Attributes:
        robot_id: The arriving robot.
        resource_id: The resource reached.
        resource_class: Its class.
    """

    robot_id: int
    resource_id: int
    resource_class: ResourceClass


@dataclass(frozen=True, slots=True)
class ResourceReleased(Event):
    """A resource returned to the pool at task completion.

    Attributes:
        robot_id: The releasing robot.
        resource_id: The released resource.
        resource_class: Its class.
        was_occupied: Whether the robot ever arrived before releasing.
    """

    robot_id: int
    resource_id: int
    resource_class: ResourceClass
    was_occupied: bool


@dataclass(frozen=True, slots=True)
class TaskCompleted(Event):
    """A task finished and its bundle was released atomically.

    Attributes:
        task_id: The completed task.
        robot_id: The robot that completed it.
        template: Which template it instantiated.
        created_at: When the task arrived.
        first_request_at: When its first resource request was issued.
        bundle_complete_at: When its final grant landed.
    """

    task_id: int
    robot_id: int
    template: TaskTemplate
    created_at: float
    first_request_at: float
    bundle_complete_at: float


@dataclass(frozen=True, slots=True)
class RobotStateChanged(Event):
    """A robot transitioned between lifecycle states.

    Attributes:
        robot_id: The transitioning robot.
        previous: The state left.
        current: The state entered.
    """

    robot_id: int
    previous: RobotState
    current: RobotState


@dataclass(frozen=True, slots=True)
class ParameterChanged(Event):
    """A parameter was altered mid-run from the interactive controls.

    Emitting this marks the run non-comparable. An interactively tweaked run
    must never be reported as experimental data.

    Attributes:
        name: The parameter changed.
        previous: Its prior value, rendered as text.
        current: Its new value, rendered as text.
    """

    name: str
    previous: str
    current: str
