"""Lease tracking, expiry, and revocation.

A lease binds a resource to a robot with an arrival deadline. If the robot has
not physically reached the resource by then, the lease expires and the resource
returns to the pool.

Revocation during a **partial bundle hold** is the trickiest correctness point
in the project. A robot holding a crane whose elevator lease expires must give
back the elevator while keeping the crane, and the Banker's matrices must come
out consistent: ``Available`` up by one elevator, that robot's ``Allocation``
down by one elevator and unchanged for cranes, and its ``Need`` back up by one
elevator because it still intends to finish the task and will ask again.

Getting that wrong corrupts ``Need == Max - Allocation`` and the failure
surfaces much later, deep into a run, looking like a kinematics bug.
``test_lease_revocation_partial_hold`` pins exactly this case.
"""

from __future__ import annotations

from dataclasses import dataclass

from warehouse_allocator.core.entities import ResourceClass

__all__ = ["Lease", "LeaseBook"]


@dataclass(frozen=True, slots=True)
class Lease:
    """A time-limited claim on one resource instance.

    Attributes:
        robot_id: The leaseholder.
        resource_id: The leased instance.
        resource_class: Its class, cached so revocation need not look it up.
        granted_at: When the lease was issued, in simulated seconds.
        deadline: When it expires if the holder has not arrived.
    """

    robot_id: int
    resource_id: int
    resource_class: ResourceClass
    granted_at: float
    deadline: float

    def is_expired(self, now: float) -> bool:
        """Report whether the lease has passed its deadline.

        Args:
            now: Current simulated time.

        Returns:
            ``True`` if ``now`` is at or past the deadline.
        """
        return now >= self.deadline


class LeaseBook:
    """Tracks live leases and reports which have expired.

    Leases are keyed by ``(robot_id, resource_id)`` because one robot may hold
    several leased resources at once, which is precisely the partial-bundle
    case that makes revocation delicate.
    """

    __slots__ = ("_leases",)

    def __init__(self) -> None:
        """Initialize an empty lease book."""
        self._leases: dict[tuple[int, int], Lease] = {}

    def __len__(self) -> int:
        """Return the number of live leases.

        Returns:
            Count of outstanding leases.
        """
        return len(self._leases)

    @property
    def active(self) -> tuple[Lease, ...]:
        """Return every live lease in a stable order.

        Returns:
            Leases sorted by ``(robot_id, resource_id)``, so iteration never
            depends on dict insertion order.
        """
        return tuple(self._leases[key] for key in sorted(self._leases))

    def grant(self, lease: Lease) -> None:
        """Record a new lease.

        Args:
            lease: The lease to track.
        """
        self._leases[(lease.robot_id, lease.resource_id)] = lease

    def get(self, robot_id: int, resource_id: int) -> Lease | None:
        """Look up a live lease.

        Args:
            robot_id: The holder.
            resource_id: The resource.

        Returns:
            The lease, or ``None`` if there is none.
        """
        return self._leases.get((robot_id, resource_id))

    def discharge(self, robot_id: int, resource_id: int) -> Lease | None:
        """Remove a lease that no longer applies.

        Called both when the holder arrives (the lease has done its job) and
        when the resource is released at task completion.

        Args:
            robot_id: The holder.
            resource_id: The resource.

        Returns:
            The removed lease, or ``None`` if there was none.
        """
        return self._leases.pop((robot_id, resource_id), None)

    def discharge_robot(self, robot_id: int) -> tuple[Lease, ...]:
        """Remove every lease held by one robot.

        Args:
            robot_id: The holder whose leases are being cleared.

        Returns:
            The removed leases, in stable order.
        """
        keys = sorted(key for key in self._leases if key[0] == robot_id)
        return tuple(self._leases.pop(key) for key in keys)

    def expired(self, now: float) -> tuple[Lease, ...]:
        """Return leases whose deadline has passed.

        Args:
            now: Current simulated time.

        Returns:
            Expired leases in stable order. The caller decides what to do with
            them; this method has no side effects, so a holder that has since
            arrived can be filtered out before revoking.
        """
        return tuple(lease for lease in self.active if lease.is_expired(now))
