"""Stage 3 verification: the Banker's safety algorithm.

The textbook matrices used here are the standard OS-course example (Silberschatz
et al., *Operating System Concepts*), so the expected verdicts are independently
checkable rather than self-referential.
"""

from __future__ import annotations

import numpy as np
import pytest

from warehouse_allocator.alloc.safety import SafetyChecker
from warehouse_allocator.alloc.state import ResourceState
from warehouse_allocator.core.entities import ResourceClass

# The classic five-process, three-resource example. Totals are A=10, B=5, C=7.
# Classes are sorted by value in ResourceState, giving column order
# crane, dock, elevator -- which stands in for A, B, C respectively.
CLASS_A = ResourceClass.CRANE
CLASS_B = ResourceClass.DOCK
CLASS_C = ResourceClass.ELEVATOR

TEXTBOOK_MAX = [
    [7, 5, 3],
    [3, 2, 2],
    [9, 0, 2],
    [2, 2, 2],
    [4, 3, 3],
]
TEXTBOOK_ALLOCATION = [
    [0, 1, 0],
    [2, 0, 0],
    [3, 0, 2],
    [2, 1, 1],
    [0, 0, 2],
]
TEXTBOOK_TOTALS = {CLASS_A: 10, CLASS_B: 5, CLASS_C: 7}


def _textbook_state(
    allocation: list[list[int]] | None = None,
) -> ResourceState:
    """Build the textbook safe state.

    Args:
        allocation: Optional Allocation matrix override.

    Returns:
        A :class:`ResourceState` populated to the textbook example, with
        column order matching ``[CLASS_A, CLASS_B, CLASS_C]``.
    """
    state = ResourceState(robot_ids=(0, 1, 2, 3, 4), totals=TEXTBOOK_TOTALS)
    assert state.classes == (
        CLASS_A,
        CLASS_B,
        CLASS_C,
    ), f"column order changed: {state.classes}"

    rows = allocation if allocation is not None else TEXTBOOK_ALLOCATION
    for robot_id, (max_row, alloc_row) in enumerate(
        zip(TEXTBOOK_MAX, rows, strict=True)
    ):
        state.declare_max(
            robot_id,
            {CLASS_A: max_row[0], CLASS_B: max_row[1], CLASS_C: max_row[2]},
        )
        for column, count in enumerate(alloc_row):
            if count:
                vector = np.zeros(3, dtype=np.int64)
                vector[column] = count
                state.grant(robot_id, vector)
    return state


class TestKnownStates:
    """Verdicts on matrices with independently known answers."""

    def test_safety_known_safe(self) -> None:
        """The textbook safe state must be reported safe.

        Available is (3, 3, 2) and the safe sequence P1, P3, P4, P2, P0
        exists.
        """
        state = _textbook_state()
        assert list(state.available) == [3, 3, 2]

        zero = np.zeros(3, dtype=np.int64)
        assert SafetyChecker().is_safe(state, robot_id=0, request=zero) is True

    def test_safety_known_safe_sequence(self) -> None:
        """A concrete safe sequence must be recoverable."""
        state = _textbook_state()
        sequence = SafetyChecker().safe_sequence(
            state, robot_id=0, request=np.zeros(3, dtype=np.int64)
        )

        assert sequence is not None
        assert set(sequence) == {0, 1, 2, 3, 4}
        assert sequence[0] == 1

    def test_safety_known_unsafe(self) -> None:
        """A known unsafe matrix must be reported unsafe.

        Granting P0 its full remaining need of (7, 4, 3) from an Available of
        (3, 3, 2) is impossible; the smaller probe below is the real test.
        """
        state = _textbook_state()
        # P4 requesting (3, 3, 0) leaves Available (0, 0, 2), after which no
        # process's remaining need can be met. This is the standard unsafe
        # continuation of the same example.
        request = np.array([3, 3, 0], dtype=np.int64)

        assert SafetyChecker().is_safe(state, robot_id=4, request=request) is False

    def test_textbook_request_p1_is_safe(self) -> None:
        """P1 requesting (1, 0, 2) is the textbook's granted example."""
        state = _textbook_state()
        request = np.array([1, 0, 2], dtype=np.int64)

        assert SafetyChecker().is_safe(state, robot_id=1, request=request) is True

    def test_request_beyond_need_refused(self) -> None:
        """A request exceeding the declared maximum must be refused."""
        state = _textbook_state()
        request = np.array([0, 3, 0], dtype=np.int64)

        assert SafetyChecker().is_safe(state, robot_id=0, request=request) is False

    def test_request_beyond_available_refused(self) -> None:
        """A request exceeding what is free must be refused."""
        state = _textbook_state()
        request = np.array([7, 0, 0], dtype=np.int64)

        assert SafetyChecker().is_safe(state, robot_id=0, request=request) is False

    def test_is_safe_is_pure(self) -> None:
        """The checker must never mutate the state it inspects."""
        state = _textbook_state()
        before = (
            state.available.copy(),
            state.allocation.copy(),
            state.need.copy(),
            state.max_claim.copy(),
        )

        SafetyChecker().is_safe(
            state, robot_id=1, request=np.array([1, 0, 2], dtype=np.int64)
        )

        assert np.array_equal(state.available, before[0])
        assert np.array_equal(state.allocation, before[1])
        assert np.array_equal(state.need, before[2])
        assert np.array_equal(state.max_claim, before[3])

    def test_all_idle_is_safe(self) -> None:
        """A fleet holding nothing and needing nothing is trivially safe."""
        state = ResourceState(robot_ids=(0, 1, 2), totals=TEXTBOOK_TOTALS)

        assert SafetyChecker().is_safe(state, 0, np.zeros(3, dtype=np.int64)) is True

    def test_full_allocation_is_still_safe(self) -> None:
        """Robots holding exactly their max need nothing and can all finish."""
        state = ResourceState(robot_ids=(0, 1), totals={CLASS_A: 2})
        for robot_id in (0, 1):
            state.declare_max(robot_id, {CLASS_A: 1})
            state.grant(robot_id, np.array([1], dtype=np.int64))

        assert list(state.available) == [0]
        assert SafetyChecker().is_safe(state, 0, np.zeros(1, dtype=np.int64)) is True


class TestUnsafeConstruction:
    """A deliberately deadlock-prone state must be caught."""

    def test_grant_completing_deadlock_is_unsafe(self) -> None:
        """The grant that would close a hold-and-wait ring must be refused.

        Two robots, two single-instance classes. Each holds one and needs the
        other. Granting either the second resource completes the cycle.
        """
        totals = {CLASS_A: 1, CLASS_B: 1}
        state = ResourceState(robot_ids=(0, 1), totals=totals)
        state.declare_max(0, {CLASS_A: 1, CLASS_B: 1})
        state.declare_max(1, {CLASS_A: 1, CLASS_B: 1})

        state.grant(0, np.array([1, 0], dtype=np.int64))
        state.grant(1, np.array([0, 1], dtype=np.int64))

        assert list(state.available) == [0, 0]
        assert (
            SafetyChecker().is_safe(state, 0, np.array([0, 1], dtype=np.int64)) is False
        )
        assert (
            SafetyChecker().is_safe(state, 1, np.array([1, 0], dtype=np.int64)) is False
        )

    def test_second_grant_prevented_before_deadlock(self) -> None:
        """With two instances, the first robot can be fully served safely."""
        totals = {CLASS_A: 1, CLASS_B: 1}
        state = ResourceState(robot_ids=(0, 1), totals=totals)
        state.declare_max(0, {CLASS_A: 1, CLASS_B: 1})
        state.declare_max(1, {CLASS_A: 1, CLASS_B: 1})
        state.grant(0, np.array([1, 0], dtype=np.int64))

        # Robot 0 can finish, so completing its bundle is safe.
        assert (
            SafetyChecker().is_safe(state, 0, np.array([0, 1], dtype=np.int64)) is True
        )
        # Giving robot 1 the other class instead strands both.
        assert (
            SafetyChecker().is_safe(state, 1, np.array([0, 1], dtype=np.int64)) is False
        )


class TestMatrixInvariants:
    """`Need == Max - Allocation` must survive arbitrary valid sequences."""

    def test_matrix_invariants(self) -> None:
        """Random valid operation sequences must preserve every invariant."""
        rng = np.random.default_rng(20260831)
        totals = {CLASS_A: 3, CLASS_B: 2, CLASS_C: 4}
        robot_ids = tuple(range(6))

        for _ in range(60):
            state = ResourceState(robot_ids=robot_ids, totals=totals)
            declared: dict[int, dict[ResourceClass, int]] = {}

            for _ in range(120):
                robot_id = int(rng.integers(0, len(robot_ids)))
                holding = int(state.held_by(robot_id).sum())
                action = int(rng.integers(0, 3))

                if robot_id not in declared and holding == 0:
                    bundle = {
                        CLASS_A: int(rng.integers(0, 2)),
                        CLASS_B: int(rng.integers(0, 2)),
                        CLASS_C: int(rng.integers(0, 2)),
                    }
                    state.declare_max(robot_id, bundle)
                    declared[robot_id] = bundle
                elif action == 0 and robot_id in declared:
                    column = int(rng.integers(0, 3))
                    vector = np.zeros(3, dtype=np.int64)
                    vector[column] = 1
                    if state.can_grant(robot_id, vector):
                        state.grant(robot_id, vector)
                elif action == 1 and holding > 0:
                    held = state.held_by(robot_id)
                    columns = np.flatnonzero(held)
                    column = int(columns[int(rng.integers(0, len(columns)))])
                    vector = np.zeros(3, dtype=np.int64)
                    vector[column] = 1
                    state.release(robot_id, vector)
                elif action == 2 and robot_id in declared and holding == 0:
                    state.clear_max(robot_id)
                    declared.pop(robot_id)

                assert np.array_equal(state.need, state.max_claim - state.allocation)
                assert np.all(state.allocation <= state.max_claim)
                assert np.all(state.allocation >= 0)
                assert np.all(state.need >= 0)
                assert np.all(state.available >= 0)
                assert np.array_equal(
                    state.available + state.allocation.sum(axis=0), state.total
                )

    def test_release_more_than_held_rejected(self) -> None:
        """Over-releasing must raise rather than corrupt the matrices."""
        state = ResourceState(robot_ids=(0,), totals={CLASS_A: 2})
        state.declare_max(0, {CLASS_A: 2})
        state.grant(0, np.array([1], dtype=np.int64))

        with pytest.raises(ValueError, match="cannot release more than it holds"):
            state.release(0, np.array([2], dtype=np.int64))

    def test_declare_while_holding_rejected(self) -> None:
        """Redeclaring a max mid-hold would silently break the invariant."""
        state = ResourceState(robot_ids=(0,), totals={CLASS_A: 2})
        state.declare_max(0, {CLASS_A: 2})
        state.grant(0, np.array([1], dtype=np.int64))

        with pytest.raises(ValueError, match="still holding"):
            state.declare_max(0, {CLASS_A: 1})

    def test_claim_beyond_totals_rejected(self) -> None:
        """A claim larger than the world can ever satisfy must be refused."""
        state = ResourceState(robot_ids=(0,), totals={CLASS_A: 2})

        with pytest.raises(ValueError, match="exceeding total resources"):
            state.declare_max(0, {CLASS_A: 3})

    def test_explicit_index_mapping(self) -> None:
        """Rows and columns must resolve through explicit id mappings."""
        state = ResourceState(robot_ids=(7, 3, 11), totals=TEXTBOOK_TOTALS)

        assert state.robot_row(7) == 0
        assert state.robot_row(3) == 1
        assert state.robot_row(11) == 2
        assert state.class_column(state.classes[1]) == 1
        with pytest.raises(KeyError):
            state.robot_row(999)
