"""The Available / Max / Allocation / Need matrices.

Standard Banker's bookkeeping, indexed by *explicit* id-to-index mappings
rather than by positional assumption. Robot ids and resource classes are both
translated through dictionaries, so nothing breaks if ids are sparse or a
resource class is added.

The invariant ``Need == Max - Allocation`` is asserted after every mutation.
That check is cheap and it is the thing that catches lease revocation during a
partial bundle hold corrupting the matrices -- the subtlest failure mode in the
project. Do not disable it for speed.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from warehouse_allocator.core.entities import ResourceClass

__all__ = ["ResourceState"]

IntMatrix = npt.NDArray[np.int64]
IntVector = npt.NDArray[np.int64]


class ResourceState:
    """Owns the four Banker's matrices for one simulation run.

    ``Max`` is declared dynamically: a robot's maximum claim is the bundle of
    whatever task it currently holds, and drops back to zero when the task
    completes. This is the standard way to apply Banker's to a workload where
    processes are not known in advance, and it keeps ``Need`` meaningful for
    idle robots (they need nothing, so they never block a safe sequence).

    Attributes:
        robot_ids: Robot ids in row order.
        classes: Resource classes in column order.
    """

    __slots__ = (
        "_allocation",
        "_available",
        "_class_index",
        "_max",
        "_need",
        "_robot_index",
        "_total",
        "classes",
        "robot_ids",
    )

    def __init__(
        self,
        robot_ids: tuple[int, ...],
        totals: dict[ResourceClass, int],
    ) -> None:
        """Initialize an empty allocation state.

        Args:
            robot_ids: Every robot that may request resources.
            totals: Instance count per resource class.
        """
        self.robot_ids: Final[tuple[int, ...]] = tuple(robot_ids)
        self.classes: Final[tuple[ResourceClass, ...]] = tuple(
            sorted(totals, key=lambda item: item.value)
        )
        self._robot_index: Final[dict[int, int]] = {
            robot_id: index for index, robot_id in enumerate(self.robot_ids)
        }
        self._class_index: Final[dict[ResourceClass, int]] = {
            cls: index for index, cls in enumerate(self.classes)
        }

        n_robots = len(self.robot_ids)
        n_classes = len(self.classes)
        self._total: IntVector = np.array(
            [totals[cls] for cls in self.classes], dtype=np.int64
        )
        self._available: IntVector = self._total.copy()
        self._max: IntMatrix = np.zeros((n_robots, n_classes), dtype=np.int64)
        self._allocation: IntMatrix = np.zeros((n_robots, n_classes), dtype=np.int64)
        self._need: IntMatrix = np.zeros((n_robots, n_classes), dtype=np.int64)

    @property
    def available(self) -> IntVector:
        """Return free instances per class.

        Returns:
            A copy of the Available vector.
        """
        return self._available.copy()

    @property
    def allocation(self) -> IntMatrix:
        """Return currently held instances per robot per class.

        Returns:
            A copy of the Allocation matrix.
        """
        return self._allocation.copy()

    @property
    def max_claim(self) -> IntMatrix:
        """Return the declared maximum claim per robot per class.

        Returns:
            A copy of the Max matrix.
        """
        return self._max.copy()

    @property
    def need(self) -> IntMatrix:
        """Return the outstanding need per robot per class.

        Returns:
            A copy of the Need matrix.
        """
        return self._need.copy()

    @property
    def total(self) -> IntVector:
        """Return the total instance count per class.

        Returns:
            A copy of the totals vector.
        """
        return self._total.copy()

    def robot_row(self, robot_id: int) -> int:
        """Translate a robot id to its matrix row.

        Args:
            robot_id: The robot to locate.

        Returns:
            The row index.

        Raises:
            KeyError: If the robot is not registered.
        """
        return self._robot_index[robot_id]

    def class_column(self, resource_class: ResourceClass) -> int:
        """Translate a resource class to its matrix column.

        Args:
            resource_class: The class to locate.

        Returns:
            The column index.

        Raises:
            KeyError: If the class is not registered.
        """
        return self._class_index[resource_class]

    def request_vector(
        self, resource_class: ResourceClass, count: int = 1
    ) -> IntVector:
        """Build a request vector for a single class.

        Args:
            resource_class: The class being requested.
            count: How many instances.

        Returns:
            A zero vector with ``count`` in the requested class's column.
        """
        vector = np.zeros(len(self.classes), dtype=np.int64)
        vector[self._class_index[resource_class]] = count
        return vector

    def declare_max(self, robot_id: int, bundle: dict[ResourceClass, int]) -> None:
        """Declare a robot's maximum claim as it accepts a task.

        Args:
            robot_id: The accepting robot.
            bundle: Instances per class the task will require.

        Raises:
            ValueError: If the robot still holds resources, or the bundle
                exceeds the total instances that exist.
        """
        row = self._robot_index[robot_id]
        if int(self._allocation[row].sum()) != 0:
            raise ValueError(f"robot {robot_id} declared a new max while still holding")
        claim = np.zeros(len(self.classes), dtype=np.int64)
        for resource_class, count in bundle.items():
            claim[self._class_index[resource_class]] = count
        if np.any(claim > self._total):
            raise ValueError(
                f"robot {robot_id} declared a claim exceeding total resources"
            )
        self._max[row] = claim
        self._need[row] = claim
        self._check_invariants()

    def clear_max(self, robot_id: int) -> None:
        """Reset a robot's claim to zero when its task ends.

        Args:
            robot_id: The robot to reset.

        Raises:
            ValueError: If the robot still holds resources.
        """
        row = self._robot_index[robot_id]
        if int(self._allocation[row].sum()) != 0:
            raise ValueError(f"robot {robot_id} cleared its max while still holding")
        self._max[row] = 0
        self._need[row] = 0
        self._check_invariants()

    def can_grant(self, robot_id: int, request: IntVector) -> bool:
        """Report whether a request is within need and availability.

        This is the cheap precondition, not the safety check. A request may
        pass here and still be refused as unsafe.

        Args:
            robot_id: The requesting robot.
            request: Instances requested per class.

        Returns:
            ``True`` if the request is within the robot's declared need and
            currently available.
        """
        row = self._robot_index[robot_id]
        return bool(
            np.all(request <= self._need[row]) and np.all(request <= self._available)
        )

    def grant(self, robot_id: int, request: IntVector) -> None:
        """Commit a grant to the matrices.

        Args:
            robot_id: The granted robot.
            request: Instances granted per class.

        Raises:
            ValueError: If the grant would exceed need or availability.
        """
        if not self.can_grant(robot_id, request):
            raise ValueError(f"grant to robot {robot_id} exceeds need or availability")
        row = self._robot_index[robot_id]
        self._available -= request
        self._allocation[row] += request
        self._need[row] -= request
        self._check_invariants()

    def release(self, robot_id: int, request: IntVector) -> None:
        """Return held instances to the pool without changing the claim.

        Used both by normal release and by lease revocation. Revocation must
        *not* reduce ``Max``: the robot still intends to complete its task and
        will request the class again, so its need is restored rather than
        forgotten.

        Args:
            robot_id: The releasing robot.
            request: Instances returned per class.

        Raises:
            ValueError: If the robot does not hold that much.
        """
        row = self._robot_index[robot_id]
        if np.any(request > self._allocation[row]):
            raise ValueError(f"robot {robot_id} cannot release more than it holds")
        self._available += request
        self._allocation[row] -= request
        self._need[row] += request
        self._check_invariants()

    def rebuild_for(self, robot_ids: tuple[int, ...]) -> ResourceState:
        """Return a copy resized to a new fleet, preserving surviving rows.

        Used by the interactive robot-count slider. Only safe because the
        controls add fresh robots (holding nothing) and remove idle ones
        (holding nothing), so no allocation is ever silently dropped.

        Args:
            robot_ids: The new fleet.

        Returns:
            A new state carrying over each surviving robot's claim and
            allocation.

        Raises:
            ValueError: If a departing robot still holds resources.
        """
        totals = {
            resource_class: int(self._total[index])
            for index, resource_class in enumerate(self.classes)
        }
        rebuilt = ResourceState(robot_ids=robot_ids, totals=totals)

        surviving = set(robot_ids)
        for robot_id in self.robot_ids:
            if robot_id in surviving:
                continue
            if int(self._allocation[self._robot_index[robot_id]].sum()) != 0:
                raise ValueError(
                    f"robot {robot_id} cannot be removed while holding resources"
                )

        for robot_id in robot_ids:
            if robot_id not in self._robot_index:
                continue
            source = self._robot_index[robot_id]
            target = rebuilt._robot_index[robot_id]
            rebuilt._max[target] = self._max[source]
            rebuilt._allocation[target] = self._allocation[source]
            rebuilt._need[target] = self._need[source]

        rebuilt._available = self._total - rebuilt._allocation.sum(axis=0)
        rebuilt._check_invariants()
        return rebuilt

    def held_by(self, robot_id: int) -> IntVector:
        """Return what one robot currently holds.

        Args:
            robot_id: The robot to inspect.

        Returns:
            A copy of that robot's Allocation row.
        """
        row: IntVector = self._allocation[self._robot_index[robot_id]].copy()
        return row

    def _check_invariants(self) -> None:
        """Assert the Banker's invariants after a mutation.

        Raises:
            AssertionError: If any invariant is violated. The message names
                which one, because these fire deep into long runs and the
                first question is always which invariant broke.
        """
        if not np.array_equal(self._need, self._max - self._allocation):
            raise AssertionError("invariant violated: Need != Max - Allocation")
        if np.any(self._allocation > self._max):
            raise AssertionError("invariant violated: Allocation > Max")
        if np.any(self._allocation < 0) or np.any(self._need < 0):
            raise AssertionError("invariant violated: negative entry")
        if np.any(self._available < 0):
            raise AssertionError("invariant violated: negative Available")
        if not np.array_equal(
            self._available + self._allocation.sum(axis=0), self._total
        ):
            raise AssertionError(
                "invariant violated: Available + sum(Allocation) != Total"
            )

    def __repr__(self) -> str:
        """Return a debug representation of the pool occupancy.

        Returns:
            A string naming available versus total per class.
        """
        parts = [
            f"{cls.value}={self._available[index]}/{self._total[index]}"
            for index, cls in enumerate(self.classes)
        ]
        return f"ResourceState({', '.join(parts)})"
