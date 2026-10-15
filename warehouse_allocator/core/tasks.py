"""Task definitions and multi-resource bundle templates.

A task requires a *bundle* of resources acquired *incrementally*: the robot
requests one class, holds whatever it is granted, and keeps requesting the
remainder while holding. That holding-while-requesting is the precondition for
deadlock, and therefore the precondition for the Banker's safety check to have
anything to do.

Why each template declares an explicit acquisition ORDER
--------------------------------------------------------
It is not enough for bundles to overlap. If every robot acquired classes in one
global order (say always crane, then elevator, then dock), circular wait would
be *structurally impossible* -- that is the classic ordered-resource deadlock
*prevention* scheme. Banker's would then be dead weight over a FIFO queue,
which is exactly the failure BUILD_SPEC.md section 2.1 warns against.

So the acquisition order is per template, and the orders are deliberately
inconsistent with each other:

    HEAVY_LIFT      crane -> elevator          holds crane,    wants elevator
    FLOOR_TRANSFER  elevator -> dock           holds elevator, wants dock
    FULL_CYCLE      dock -> crane -> elevator  holds dock,     wants crane

Those three compose a genuine cycle: a HEAVY_LIFT robot holding a crane waits
on an elevator, a FLOOR_TRANSFER robot holding that elevator waits on a dock,
and a FULL_CYCLE robot holding that dock waits on the crane. The overlap on
``elevator`` between HEAVY_LIFT and FLOOR_TRANSFER is what BUILD_SPEC.md
section 2.1 asks for; the inconsistent ordering is what makes it bite.

``tests/test_allocator.py::test_circular_wait_refused`` constructs exactly this
cycle and asserts the allocator refuses the request that would close it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from warehouse_allocator.core.entities import ResourceClass
from warehouse_allocator.core.geometry import Coord

__all__ = [
    "TEMPLATE_ACQUISITION_ORDER",
    "TEMPLATE_WEIGHTS",
    "Task",
    "TaskTemplate",
    "bundle_for",
]


class TaskTemplate(StrEnum):
    """A kind of warehouse job, each with its own resource bundle.

    Attributes:
        HEAVY_LIFT: Lift a heavy pallet between floors. Crane then elevator.
        FLOOR_TRANSFER: Move a load between floors and out to a bay.
            Elevator then dock.
        FULL_CYCLE: End-to-end job touching every resource class.
        RECHARGE: Occupy a dock to recharge. The single-class template.
    """

    HEAVY_LIFT = "heavy_lift"
    FLOOR_TRANSFER = "floor_transfer"
    FULL_CYCLE = "full_cycle"
    RECHARGE = "recharge"


# Acquisition order per template. Deliberately mutually inconsistent; see the
# module docstring for why a single global order would defeat the experiment.
TEMPLATE_ACQUISITION_ORDER: Final[dict[TaskTemplate, tuple[ResourceClass, ...]]] = {
    TaskTemplate.HEAVY_LIFT: (ResourceClass.CRANE, ResourceClass.ELEVATOR),
    TaskTemplate.FLOOR_TRANSFER: (ResourceClass.ELEVATOR, ResourceClass.DOCK),
    TaskTemplate.FULL_CYCLE: (
        ResourceClass.DOCK,
        ResourceClass.CRANE,
        ResourceClass.ELEVATOR,
    ),
    TaskTemplate.RECHARGE: (ResourceClass.DOCK,),
}

# Relative arrival frequency. Multi-class templates dominate because
# single-class work cannot contend for a bundle and so contributes nothing to
# the phenomenon under study.
TEMPLATE_WEIGHTS: Final[dict[TaskTemplate, float]] = {
    TaskTemplate.HEAVY_LIFT: 0.35,
    TaskTemplate.FLOOR_TRANSFER: 0.30,
    TaskTemplate.FULL_CYCLE: 0.20,
    TaskTemplate.RECHARGE: 0.15,
}


def bundle_for(template: TaskTemplate) -> dict[ResourceClass, int]:
    """Return the resource bundle a template requires.

    Args:
        template: The template to describe.

    Returns:
        A mapping of resource class to the number of instances required.
    """
    return dict(Counter(TEMPLATE_ACQUISITION_ORDER[template]))


@dataclass(slots=True)
class Task:
    """One unit of warehouse work drawn from a template.

    The workload-defining fields are fixed when the trace is generated and are
    the only fields serialized. ``timestamps`` is filled in during a run and is
    a debugging convenience; the authoritative lifecycle record is the event
    log in :mod:`warehouse_allocator.metrics`.

    Attributes:
        task_id: Stable identifier, unique within a trace.
        template: Which :class:`TaskTemplate` this task instantiates.
        pickup: Where the load originates, as ``(x, y)`` grid cell.
        dropoff: Where the load must be delivered, as ``(x, y)`` grid cell.
        service_duration: Simulated seconds of work at the resources.
        created_at: Simulated arrival time, from the Poisson process.
        timestamps: Lifecycle marks recorded during a run, keyed by phase.
    """

    task_id: int
    template: TaskTemplate
    pickup: Coord
    dropoff: Coord
    service_duration: float
    created_at: float
    timestamps: dict[str, float] = field(default_factory=dict)

    @property
    def acquisition_order(self) -> tuple[ResourceClass, ...]:
        """Return the order in which this task's classes must be requested.

        Returns:
            The template's acquisition order, one entry per instance needed.
        """
        return TEMPLATE_ACQUISITION_ORDER[self.template]

    @property
    def required_bundle(self) -> dict[ResourceClass, int]:
        """Return the full bundle this task requires.

        Returns:
            A mapping of resource class to instance count.
        """
        return bundle_for(self.template)

    def mark(self, phase: str, when: float) -> None:
        """Record a lifecycle timestamp.

        Only the first mark for a phase is kept, so a retried phase reports
        when it was first entered rather than when it last was.

        Args:
            phase: Lifecycle phase name, for example ``"first_request"``.
            when: Simulated time of the event.
        """
        self.timestamps.setdefault(phase, when)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the workload-defining fields.

        Returns:
            A JSON-compatible dict. Runtime ``timestamps`` are excluded
            because they are a property of a run, not of the workload.
        """
        return {
            "task_id": self.task_id,
            "template": str(self.template),
            "pickup": list(self.pickup),
            "dropoff": list(self.dropoff),
            "service_duration": self.service_duration,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Reconstruct a task from :meth:`to_dict` output.

        Args:
            data: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed task, with an empty timestamp log.
        """
        return cls(
            task_id=int(data["task_id"]),
            template=TaskTemplate(data["template"]),
            pickup=(int(data["pickup"][0]), int(data["pickup"][1])),
            dropoff=(int(data["dropoff"][0]), int(data["dropoff"][1])),
            service_duration=float(data["service_duration"]),
            created_at=float(data["created_at"]),
        )

    def copy(self) -> Task:
        """Return an independent copy with a cleared timestamp log.

        Every policy in a comparison must run the identical workload, but each
        run must not inherit the previous run's lifecycle marks.

        Returns:
            A fresh task with the same workload-defining fields.
        """
        return Task(
            task_id=self.task_id,
            template=self.template,
            pickup=self.pickup,
            dropoff=self.dropoff,
            service_duration=self.service_duration,
            created_at=self.created_at,
        )
