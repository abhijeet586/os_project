"""Grid geometry, distance metrics, and deterministic A* pathfinding.

This module is the single highest-risk correctness point in the project. If
A* is nondeterministic, every downstream policy comparison is noise, because
two runs of the same seed will route robots differently and produce different
travel times, different contention, and different throughput.

Determinism contract, non-negotiable:

1. The open set is a binary heap keyed by
   ``(f_score, h_score, tie_break_counter, node)``. The trailing counter makes
   every key unique, so ``heapq`` never falls back to comparing anything
   ambiguous and equal-cost frontiers pop in a fixed order.
2. Neighbour expansion order is the fixed constant tuple
   :data:`NEIGHBOR_OFFSETS`, never a set and never a runtime-built collection.
3. Sets and dicts are used only for membership tests and keyed lookups. They
   are never iterated anywhere the iteration order could reach the output.

The grid is 4-connected. Diagonal movement is excluded deliberately: with unit
step costs the Manhattan heuristic is then both admissible and consistent, and
open aisles produce large numbers of genuinely equal-cost paths, which is
exactly the condition that exposes unstable tie-breaking. A test walks 100
identical searches across such a map and asserts one single path comes back.
"""

from __future__ import annotations

import heapq
import math
from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "NEIGHBOR_OFFSETS",
    "Coord",
    "Grid",
    "astar",
    "euclidean_distance",
    "manhattan_distance",
    "path_cost",
]

Coord = tuple[int, int]
"""An integer grid cell as ``(x, y)``, with ``x`` the column and ``y`` the row."""

# Fixed expansion order: north, east, south, west. Declared as a module-level
# constant tuple so the order can never vary between runs, processes, or
# platforms the way a set's iteration order can.
NEIGHBOR_OFFSETS: Final[tuple[Coord, ...]] = ((0, -1), (1, 0), (0, 1), (-1, 0))


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return the straight-line distance between two points.

    This is the metric ``RadiusPolicy`` admits on, and its blindness to
    obstacles is precisely the weakness ``ETAPolicy`` exists to correct.

    Args:
        a: First point as ``(x, y)`` in metres.
        b: Second point as ``(x, y)`` in metres.

    Returns:
        Euclidean distance in metres.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def manhattan_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return the L1 distance between two points.

    On a 4-connected unit-cost grid this is an admissible and consistent A*
    heuristic: it never overestimates, because no single step reduces the
    summed axis distance by more than one.

    Args:
        a: First point as ``(x, y)``.
        b: Second point as ``(x, y)``.

    Returns:
        Manhattan distance in metres.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Grid:
    """A rectangular occupancy grid of passable and blocked cells.

    One cell is one metre, so grid coordinates and metric positions share a
    scale and no conversion factor is needed anywhere in the simulation.

    Attributes:
        width: Number of columns.
        height: Number of rows.
    """

    __slots__ = ("_blocked", "_height", "_width")

    def __init__(self, width: int, height: int) -> None:
        """Create an obstacle-free grid.

        Args:
            width: Number of columns; must be positive.
            height: Number of rows; must be positive.

        Raises:
            ValueError: If either dimension is not positive.
        """
        if width <= 0 or height <= 0:
            raise ValueError("grid dimensions must be positive")
        self._width = width
        self._height = height
        self._blocked: npt.NDArray[np.bool_] = np.zeros((height, width), dtype=np.bool_)

    @property
    def width(self) -> int:
        """Return the number of columns.

        Returns:
            Grid width in cells.
        """
        return self._width

    @property
    def height(self) -> int:
        """Return the number of rows.

        Returns:
            Grid height in cells.
        """
        return self._height

    @property
    def blocked(self) -> npt.NDArray[np.bool_]:
        """Return the raw obstacle mask, indexed ``[y, x]``.

        Returns:
            A read-only view of the boolean obstacle array. The view is
            marked non-writeable so callers cannot desynchronise the grid
            from paths already computed against it.
        """
        view = self._blocked.view()
        view.flags.writeable = False
        return view

    def in_bounds(self, cell: Coord) -> bool:
        """Report whether a cell lies inside the grid.

        Args:
            cell: Cell to test as ``(x, y)``.

        Returns:
            ``True`` if within bounds.
        """
        x, y = cell
        return 0 <= x < self._width and 0 <= y < self._height

    def is_free(self, cell: Coord) -> bool:
        """Report whether a cell is in bounds and passable.

        Args:
            cell: Cell to test as ``(x, y)``.

        Returns:
            ``True`` if the cell can be occupied or traversed.
        """
        if not self.in_bounds(cell):
            return False
        x, y = cell
        return not bool(self._blocked[y, x])

    def set_blocked(self, cell: Coord, blocked: bool = True) -> None:
        """Mark a single cell as an obstacle or clear it.

        Args:
            cell: Cell to modify as ``(x, y)``.
            blocked: ``True`` to place an obstacle, ``False`` to clear.

        Raises:
            ValueError: If the cell is out of bounds.
        """
        if not self.in_bounds(cell):
            raise ValueError(f"cell {cell} is out of bounds")
        x, y = cell
        self._blocked[y, x] = blocked

    def set_mask(self, mask: npt.NDArray[np.bool_]) -> None:
        """Replace the entire obstacle layout.

        Args:
            mask: Boolean array of shape ``(height, width)``; ``True`` marks
                an obstacle.

        Raises:
            ValueError: If ``mask`` has the wrong shape.
        """
        if mask.shape != (self._height, self._width):
            raise ValueError(
                f"mask shape {mask.shape} does not match grid "
                f"{(self._height, self._width)}"
            )
        self._blocked = mask.astype(np.bool_, copy=True)

    def neighbors(self, cell: Coord) -> tuple[Coord, ...]:
        """Return the passable neighbours of a cell in fixed order.

        Args:
            cell: Cell whose neighbours to enumerate as ``(x, y)``.

        Returns:
            A tuple of passable neighbours, always ordered by
            :data:`NEIGHBOR_OFFSETS`.
        """
        x, y = cell
        result: list[Coord] = []
        for dx, dy in NEIGHBOR_OFFSETS:
            candidate: Coord = (x + dx, y + dy)
            if self.is_free(candidate):
                result.append(candidate)
        return tuple(result)

    def free_cell_count(self) -> int:
        """Return the number of passable cells.

        Returns:
            Count of cells not marked as obstacles.
        """
        return int(self._blocked.size - np.count_nonzero(self._blocked))

    def connected_component(self, start: Coord) -> frozenset[Coord]:
        """Return every cell reachable from ``start`` by 4-connected moves.

        Used to guarantee the warehouse has no marooned pockets, so that any
        robot can reach any resource and an unreachable-goal A* result always
        indicates a bug rather than an unlucky layout.

        Args:
            start: Seed cell, which must itself be passable.

        Returns:
            The frozen set of reachable cells, empty if ``start`` is blocked.
        """
        if not self.is_free(start):
            return frozenset()
        seen: set[Coord] = {start}
        # An explicit stack rather than recursion: a 120x80 grid can chain
        # deeper than CPython's default recursion limit.
        stack: list[Coord] = [start]
        while stack:
            current = stack.pop()
            for neighbor in self.neighbors(current):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return frozenset(seen)

    def __repr__(self) -> str:
        """Return a debug representation with dimensions and obstacle count.

        Returns:
            A string of the form ``Grid(120x80, blocked=1152)``.
        """
        blocked = int(np.count_nonzero(self._blocked))
        return f"Grid({self._width}x{self._height}, blocked={blocked})"


def astar(grid: Grid, start: Coord, goal: Coord) -> tuple[Coord, ...] | None:
    """Find a shortest 4-connected path, deterministically.

    Ties between equal-cost frontier nodes are broken in a fixed, explicit
    order rather than by whatever order a set or dict happens to yield. The
    heap key is ``(f, h, counter, node)``:

    * ``f`` orders by total estimated cost, the standard A* criterion.
    * ``h`` breaks ``f`` ties toward the goal, which is a well-known A*
      speedup and, more importantly here, a *stable* preference.
    * ``counter`` is a strictly increasing push index. It makes every key
      unique, so ``heapq`` never needs to compare the ``node`` tuples to
      resolve a tie, and equal-cost pushes pop in first-in-first-out order.
    * ``node`` is present only so the popped entry carries its cell; by the
      time it is reached the key is already unique.

    Without the counter, two frontier nodes with identical ``f`` and ``h``
    would be ordered by their coordinate tuples. That is still deterministic,
    but it silently biases search toward low coordinates and, worse, invites a
    future edit to swap in a non-comparable node type and reintroduce
    instability. The counter makes the guarantee explicit and local.

    Args:
        grid: The grid to search.
        start: Starting cell as ``(x, y)``; must be passable.
        goal: Target cell as ``(x, y)``; must be passable.

    Returns:
        The path as a tuple of cells from ``start`` to ``goal`` inclusive, or
        ``None`` if either endpoint is impassable or no route exists. A path
        from a cell to itself is the single-element tuple ``(start,)``.
    """
    if not grid.is_free(start) or not grid.is_free(goal):
        return None
    if start == goal:
        return (start,)

    counter = 0
    start_h = manhattan_distance(start, goal)
    open_heap: list[tuple[float, float, int, Coord]] = [
        (start_h, start_h, counter, start)
    ]
    came_from: dict[Coord, Coord] = {}
    g_score: dict[Coord, float] = {start: 0.0}
    # Membership-only set. Never iterated, so its ordering cannot influence
    # the result; it exists purely to skip stale heap entries cheaply.
    closed: set[Coord] = set()

    while open_heap:
        _, _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(came_from, current)
        closed.add(current)

        current_g = g_score[current]
        for neighbor in grid.neighbors(current):
            if neighbor in closed:
                continue
            tentative_g = current_g + 1.0
            known_g = g_score.get(neighbor)
            if known_g is not None and tentative_g >= known_g:
                continue
            g_score[neighbor] = tentative_g
            came_from[neighbor] = current
            h = manhattan_distance(neighbor, goal)
            counter += 1
            heapq.heappush(open_heap, (tentative_g + h, h, counter, neighbor))

    return None


def _reconstruct(came_from: dict[Coord, Coord], goal: Coord) -> tuple[Coord, ...]:
    """Walk predecessor links back from the goal to build the path.

    ``came_from`` is only ever read by key, never iterated, so dict ordering
    cannot influence the reconstructed path.

    Args:
        came_from: Mapping of each visited cell to its predecessor.
        goal: The cell to walk back from.

    Returns:
        The path in start-to-goal order.
    """
    path: list[Coord] = [goal]
    node = goal
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return tuple(path)


def path_cost(grid: Grid, start: Coord, goal: Coord) -> float | None:
    """Return the true traversal cost between two cells.

    This is what ``ETAPolicy`` admits on. It differs from
    :func:`euclidean_distance` exactly where obstacles force a detour, which
    is the whole reason the two policies are worth comparing.

    Args:
        grid: The grid to search.
        start: Starting cell as ``(x, y)``.
        goal: Target cell as ``(x, y)``.

    Returns:
        Path length in metres (steps, since each step is one metre), or
        ``None`` if no route exists.
    """
    path = astar(grid, start, goal)
    if path is None:
        return None
    return float(len(path) - 1)
