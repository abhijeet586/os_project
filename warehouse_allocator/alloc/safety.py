"""The Banker's safety algorithm, unmodified.

This module is deliberately boring. It is the textbook algorithm, and the
project's contribution is the admission layer in front of it, not any change
to it. Readers who know the algorithm should recognise it immediately; readers
who do not should be able to follow it from the docstring below.
"""

from __future__ import annotations

import numpy as np

from warehouse_allocator.alloc.state import IntMatrix, IntVector, ResourceState

__all__ = ["SafetyChecker"]


class SafetyChecker:
    """Decides whether granting a request leaves the system in a safe state.

    A state is *safe* if there exists an ordering of the robots -- a safe
    sequence -- in which every robot can finish, one after another, using only
    the resources currently free plus everything released by the robots ahead
    of it. If such an order exists, deadlock is avoidable no matter how
    execution actually interleaves. If none exists, the state is unsafe and the
    request must be refused, even though nothing is deadlocked *yet*.

    The algorithm, in the standard work-vector and finish-vector form:

    1. **Tentatively grant.** Compute the state that would result::

           Available'      = Available - Request
           Allocation'[i]  = Allocation[i] + Request
           Need'[i]        = Need[i] - Request

    2. **Initialise.** ``Work = Available'`` is what is free right now.
       ``Finish[r] = False`` for every robot: none has been shown to complete.

    3. **Find a robot that can finish.** Scan for any ``r`` with
       ``Finish[r] == False`` and ``Need'[r] <= Work`` elementwise -- a robot
       whose entire remaining need can be met from what is free. Such a robot
       is guaranteed to be able to run to completion.

    4. **Let it finish and reclaim.** ``Work += Allocation'[r]`` (it returns
       everything it held on completion) and ``Finish[r] = True``. Repeat
       step 3.

    5. **Verdict.** If every robot ends with ``Finish[r] == True``, a safe
       sequence exists and the state is safe. If a pass finds no runnable
       robot while some remain unfinished, those robots can only be waiting on
       each other, and the state is unsafe.

    Idle robots hold nothing and need nothing, so they satisfy step 3
    immediately and never obstruct a safe sequence.

    The scan in step 3 runs in ascending row order, which is ascending robot
    id. Any order yields the same *verdict*, but a fixed order makes the
    algorithm's behaviour reproducible when debugging.
    """

    __slots__ = ()

    def is_safe(self, state: ResourceState, robot_id: int, request: IntVector) -> bool:
        """Report whether granting a request would leave a safe state.

        Pure: ``state`` is never modified. The tentative grant is applied to
        local copies only.

        Args:
            state: The current allocation state.
            robot_id: The robot making the request.
            request: Instances requested per class, in ``state.classes`` order.

        Returns:
            ``True`` if a safe sequence exists after the grant.
        """
        row = state.robot_row(robot_id)
        need = state.need
        allocation = state.allocation
        available = state.available

        # A request beyond the declared maximum is an error in the caller, and
        # a request beyond what exists cannot be granted now regardless of
        # safety. Both are unsafe to grant.
        if np.any(request > need[row]) or np.any(request > available):
            return False

        available = available - request
        allocation[row] = allocation[row] + request
        need[row] = need[row] - request

        return self._has_safe_sequence(available, allocation, need)

    def safe_sequence(
        self, state: ResourceState, robot_id: int, request: IntVector
    ) -> tuple[int, ...] | None:
        """Return the safe sequence a grant would admit, for explanation.

        Used by documentation and diagnostics; the allocator itself only needs
        the boolean.

        Args:
            state: The current allocation state.
            robot_id: The robot making the request.
            request: Instances requested per class.

        Returns:
            Robot ids in a viable completion order, or ``None`` if unsafe.
        """
        row = state.robot_row(robot_id)
        need = state.need
        allocation = state.allocation
        available = state.available

        if np.any(request > need[row]) or np.any(request > available):
            return None

        available = available - request
        allocation[row] = allocation[row] + request
        need[row] = need[row] - request

        order = self._build_sequence(available, allocation, need)
        if order is None:
            return None
        return tuple(state.robot_ids[index] for index in order)

    def _has_safe_sequence(
        self,
        available: IntVector,
        allocation: IntMatrix,
        need: IntMatrix,
    ) -> bool:
        """Report whether a safe sequence exists over the given matrices.

        Args:
            available: Free instances per class after the tentative grant.
            allocation: Held instances per robot per class.
            need: Outstanding need per robot per class.

        Returns:
            ``True`` if every robot can be finished in some order.
        """
        return self._build_sequence(available, allocation, need) is not None

    @staticmethod
    def _build_sequence(
        available: IntVector,
        allocation: IntMatrix,
        need: IntMatrix,
    ) -> list[int] | None:
        """Construct a safe sequence, or report that none exists.

        Args:
            available: Free instances per class after the tentative grant.
            allocation: Held instances per robot per class.
            need: Outstanding need per robot per class.

        Returns:
            Row indices in completion order, or ``None`` if the state is
            unsafe.
        """
        n_robots = allocation.shape[0]
        work = available.copy()
        finish = np.zeros(n_robots, dtype=bool)
        order: list[int] = []

        progressed = True
        while progressed:
            progressed = False
            for row in range(n_robots):
                if finish[row]:
                    continue
                if bool(np.all(need[row] <= work)):
                    work = work + allocation[row]
                    finish[row] = True
                    order.append(row)
                    progressed = True

        if not bool(np.all(finish)):
            return None
        return order
