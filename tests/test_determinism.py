"""Stage 1 verification: the determinism contract.

Every result this project produces rests on the guarantee asserted here. If
any test in this module is flaky, it is broken, and every downstream policy
comparison is noise rather than evidence.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
from pathlib import Path

import numpy as np
import pytest

from warehouse_allocator.config import SimConfig
from warehouse_allocator.core.clock import SimClock
from warehouse_allocator.core.geometry import Grid, astar, path_cost
from warehouse_allocator.core.rng import REQUIRED_STREAMS, RNGProvider
from warehouse_allocator.core.world import World

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "warehouse_allocator"

# Deliberately small so 1000 steps of a determinism run stay fast while still
# containing enough racking for paths to require real detours.
TEST_CONFIG = SimConfig(
    grid_width=40,
    grid_height=30,
    obstacle_density=0.15,
    robot_count=8,
    dt=0.05,
    duration=60.0,
    seed=1234,
)


def _run(steps: int, seed: int) -> tuple[str, list[str]]:
    """Run the world forward and record its hash trajectory.

    Args:
        steps: Number of fixed timesteps to advance.
        seed: Master seed for the run.

    Returns:
        The final state hash and the hash recorded every 100 steps, so a
        divergence can be localised in time rather than only detected at the
        end.
    """
    config = TEST_CONFIG.evolve(seed=seed)
    world = World(config, RNGProvider(seed))
    clock = SimClock(config.dt)
    trajectory: list[str] = []
    for index in range(steps):
        world.step(clock.dt)
        clock.tick()
        if index % 100 == 0:
            trajectory.append(world.state_hash())
    return world.state_hash(), trajectory


class TestIdenticalSeeds:
    """Same seed must reproduce the same run, exactly."""

    def test_identical_seeds_identical_state(self) -> None:
        """1000 steps twice under one seed must agree at every checkpoint."""
        first_hash, first_traj = _run(steps=1000, seed=1234)
        second_hash, second_traj = _run(steps=1000, seed=1234)

        assert first_traj == second_traj, "state diverged mid-run"
        assert first_hash == second_hash, "final state differs across runs"

    def test_different_seeds_diverge(self) -> None:
        """A different seed must actually change the run.

        Without this, a ``state_hash`` that ignored the state it is meant to
        cover would pass the equality test above trivially.
        """
        first_hash, _ = _run(steps=200, seed=1234)
        second_hash, _ = _run(steps=200, seed=9876)

        assert first_hash != second_hash

    def test_state_actually_advances(self) -> None:
        """The hash must change as the world runs.

        Guards the same failure mode from the other side: a frozen world
        would satisfy every equality assertion while simulating nothing.
        """
        world = World(TEST_CONFIG, RNGProvider(TEST_CONFIG.seed))
        initial = world.state_hash()
        for _ in range(50):
            world.step(TEST_CONFIG.dt)

        assert world.state_hash() != initial


class TestStreamIsolation:
    """Named RNG streams must be mutually independent."""

    def test_stream_isolation(self) -> None:
        """Draining an unrelated stream must not shift ``task_gen``."""
        baseline_provider = RNGProvider(seed=7)
        baseline = baseline_provider.stream("task_gen").random(100).tolist()

        interfered_provider = RNGProvider(seed=7)
        interfered_provider.stream("noise").random(100)
        interfered_provider.stream("layout").integers(0, 1000, size=100)
        observed = interfered_provider.stream("task_gen").random(100).tolist()

        assert observed == baseline

    def test_stream_order_independent(self) -> None:
        """Requesting streams in a different order must not change draws.

        This is the property call-order ``SeedSequence.spawn`` fails to
        provide and the reason this project derives streams from a
        name-keyed spawn key instead.
        """
        forward = RNGProvider(seed=99)
        for name in REQUIRED_STREAMS:
            forward.stream(name)
        forward_values = forward.stream("task_gen").random(50).tolist()

        reversed_provider = RNGProvider(seed=99)
        for name in reversed(REQUIRED_STREAMS):
            reversed_provider.stream(name)
        reversed_values = reversed_provider.stream("task_gen").random(50).tolist()

        assert forward_values == reversed_values

    def test_new_consumer_does_not_shift_existing(self) -> None:
        """Introducing a brand-new stream must not disturb existing ones."""
        before = RNGProvider(seed=5).stream("task_gen").random(50).tolist()

        with_newcomer = RNGProvider(seed=5)
        with_newcomer.stream("a_stream_added_later").random(500)
        after = with_newcomer.stream("task_gen").random(50).tolist()

        assert after == before

    def test_streams_are_distinct(self) -> None:
        """Two different stream names must not yield the same sequence."""
        provider = RNGProvider(seed=11)
        task_values = provider.stream("task_gen").random(50).tolist()
        noise_values = provider.stream("noise").random(50).tolist()

        assert task_values != noise_values

    def test_stream_is_cached(self) -> None:
        """Repeated lookups must continue one sequence, not restart it."""
        provider = RNGProvider(seed=3)
        first = provider.stream("task_gen").random(10).tolist()
        second = provider.stream("task_gen").random(10).tolist()

        assert first != second

    @pytest.mark.parametrize("name", REQUIRED_STREAMS)
    def test_required_streams_available(self, name: str) -> None:
        """Every stream the simulation relies on must be mintable."""
        assert RNGProvider(seed=1).stream(name) is not None


class TestAStarDeterminism:
    """A* must return one identical path for one identical query."""

    @staticmethod
    def _open_grid() -> Grid:
        """Build an obstacle-free grid rich in equal-cost paths.

        Returns:
            A 20x20 open grid. On a 4-connected unit-cost grid, any corner-to-
            corner route has a very large number of equally short solutions,
            which is exactly the condition unstable tie-breaking would expose.
        """
        return Grid(20, 20)

    def test_astar_deterministic(self) -> None:
        """100 identical searches over an open map must agree exactly."""
        grid = self._open_grid()
        paths = {astar(grid, (0, 0), (19, 19)) for _ in range(100)}

        assert len(paths) == 1, f"A* returned {len(paths)} distinct paths"

    def test_astar_deterministic_with_obstacles(self) -> None:
        """Determinism must hold when racking forces a detour."""
        grid = Grid(20, 20)
        for y in range(0, 15):
            grid.set_blocked((10, y))

        paths = {astar(grid, (0, 0), (19, 19)) for _ in range(100)}

        assert len(paths) == 1
        only = paths.pop()
        assert only is not None
        assert (10, 0) not in only

    def test_astar_returns_shortest_path(self) -> None:
        """Determinism is worthless if the path is not actually shortest."""
        grid = self._open_grid()
        path = astar(grid, (0, 0), (5, 7))

        assert path is not None
        assert len(path) - 1 == 12
        assert path[0] == (0, 0)
        assert path[-1] == (5, 7)

    def test_astar_path_is_contiguous(self) -> None:
        """Consecutive path cells must be 4-connected neighbours."""
        grid = Grid(20, 20)
        for y in range(0, 15):
            grid.set_blocked((10, y))
        path = astar(grid, (0, 0), (19, 19))

        assert path is not None
        for current, following in itertools.pairwise(path):
            step = abs(current[0] - following[0]) + abs(current[1] - following[1])
            assert step == 1
            assert grid.is_free(following)

    def test_astar_unreachable_returns_none(self) -> None:
        """A fully walled-off goal must yield ``None``, not a partial path."""
        grid = Grid(10, 10)
        for y in range(10):
            grid.set_blocked((5, y))

        assert astar(grid, (0, 0), (9, 9)) is None

    def test_astar_blocked_endpoint_returns_none(self) -> None:
        """A blocked start or goal must be rejected outright."""
        grid = Grid(10, 10)
        grid.set_blocked((9, 9))

        assert astar(grid, (0, 0), (9, 9)) is None
        assert astar(grid, (9, 9), (0, 0)) is None

    def test_astar_same_cell(self) -> None:
        """A path from a cell to itself is that single cell."""
        assert astar(Grid(10, 10), (3, 3), (3, 3)) == ((3, 3),)

    def test_path_cost_matches_detour(self) -> None:
        """Path cost must reflect the true detour, not straight-line distance.

        This divergence is the entire justification for ``ETAPolicy``
        existing alongside ``RadiusPolicy``.
        """
        grid = Grid(20, 20)
        for y in range(0, 19):
            grid.set_blocked((10, y))

        cost = path_cost(grid, (9, 0), (11, 0))

        assert cost is not None
        assert cost > 2.0


class TestNoBareRandom:
    """All randomness must flow through the seeded provider."""

    @staticmethod
    def _source_files() -> list[Path]:
        """Collect every Python source file in the package.

        Returns:
            Paths to all package modules.
        """
        return sorted(PACKAGE_ROOT.rglob("*.py"))

    def test_no_bare_random(self) -> None:
        """No package module may import the stdlib ``random`` module.

        Parsed with :mod:`ast` rather than grepped, so the word ``random``
        appearing in a docstring or in ``numpy.random`` cannot produce a false
        positive, and an obfuscated ``import  random`` cannot slip past.
        """
        offenders: list[str] = []
        for path in self._source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "random" or alias.name.startswith("random."):
                            offenders.append(f"{path.name}:{node.lineno}")
                elif isinstance(node, ast.ImportFrom) and node.module == "random":
                    offenders.append(f"{path.name}:{node.lineno}")

        assert not offenders, f"bare random imports found: {offenders}"

    def test_package_has_sources(self) -> None:
        """Guard against the scan above passing on an empty file list."""
        assert len(self._source_files()) >= 6


class TestClock:
    """The clock must be the only source of time, and must not drift."""

    def test_clock_advances_by_dt(self) -> None:
        """Each tick advances simulated time by exactly ``dt``."""
        clock = SimClock(0.05)

        assert clock.now() == 0.0
        assert clock.step_index == 0
        clock.tick()
        assert clock.step_index == 1
        assert clock.now() == pytest.approx(0.05)

    def test_clock_does_not_drift(self) -> None:
        """72,000 ticks must land exactly on one simulated hour.

        Computed as ``step_index * dt`` rather than accumulated, because
        repeated float addition of 0.05 visibly drifts at this length.
        """
        clock = SimClock(0.05)
        for _ in range(72_000):
            clock.tick()

        assert clock.now() == pytest.approx(3600.0, abs=1e-9)

    def test_clock_rejects_nonpositive_dt(self) -> None:
        """A zero or negative timestep is not a runnable configuration."""
        with pytest.raises(ValueError):
            SimClock(0.0)
        with pytest.raises(ValueError):
            SimClock(-0.05)


class TestWorldConstruction:
    """The generated layout must be usable and reproducible."""

    def test_layout_reproducible(self) -> None:
        """One seed must always produce the same racking."""
        first = World(TEST_CONFIG, RNGProvider(TEST_CONFIG.seed))
        second = World(TEST_CONFIG, RNGProvider(TEST_CONFIG.seed))

        assert np.array_equal(first.grid.blocked, second.grid.blocked)

    def test_layout_is_fully_connected(self) -> None:
        """Every passable cell must be reachable from every other.

        Pruning enforces this so that a ``None`` from A* during a run always
        indicates a bug, never an unlucky layout.
        """
        world = World(TEST_CONFIG, RNGProvider(TEST_CONFIG.seed))
        start = world.robots[0].cell
        component = world.grid.connected_component(start)

        assert len(component) == world.grid.free_cell_count()

    def test_entities_placed_on_free_cells(self) -> None:
        """No robot or resource may spawn inside racking."""
        world = World(TEST_CONFIG, RNGProvider(TEST_CONFIG.seed))

        for robot in world.robots:
            assert world.grid.is_free(robot.cell)
        for resource in world.resources:
            assert world.grid.is_free(resource.cell)

    def test_entity_counts_match_config(self) -> None:
        """The world must contain exactly what the config asked for."""
        world = World(TEST_CONFIG, RNGProvider(TEST_CONFIG.seed))

        assert len(world.robots) == TEST_CONFIG.robot_count
        expected = sum(count for _, count in TEST_CONFIG.resource_counts)
        assert len(world.resources) == expected

    def test_robot_ids_are_sorted(self) -> None:
        """Iteration order must be by id, never by insertion or hash."""
        world = World(TEST_CONFIG, RNGProvider(TEST_CONFIG.seed))
        ids = [robot.robot_id for robot in world.robots]

        assert ids == sorted(ids)

    def test_robots_do_not_tunnel_through_racking(self) -> None:
        """A robot must never occupy a blocked cell mid-run.

        The config validator caps ``robot_speed * dt`` at one cell for exactly
        this reason; this asserts the cap actually holds in motion.
        """
        world = World(TEST_CONFIG, RNGProvider(TEST_CONFIG.seed))
        for _ in range(500):
            world.step(TEST_CONFIG.dt)
            for robot in world.robots:
                assert world.grid.is_free(
                    robot.cell
                ), f"robot {robot.robot_id} entered blocked cell {robot.cell}"

    def test_grid_mask_is_read_only(self) -> None:
        """The exposed obstacle mask must not be mutable by callers."""
        world = World(TEST_CONFIG, RNGProvider(TEST_CONFIG.seed))

        with pytest.raises(ValueError):
            world.grid.blocked[0, 0] = True


class TestConfigProvenance:
    """A config must round-trip so results can be regenerated from it."""

    def test_roundtrip_via_dict(self) -> None:
        """``from_dict(to_dict(c))`` must reproduce the config."""
        assert SimConfig.from_dict(TEST_CONFIG.to_dict()) == TEST_CONFIG

    def test_roundtrip_via_json(self) -> None:
        """``from_json(to_json(c))`` must reproduce the config."""
        assert SimConfig.from_json(TEST_CONFIG.to_json()) == TEST_CONFIG

    def test_unknown_field_rejected(self) -> None:
        """A stale provenance record must fail loudly, not silently default."""
        payload = TEST_CONFIG.to_dict()
        payload["retired_parameter"] = 3

        with pytest.raises(ValueError, match="unknown SimConfig fields"):
            SimConfig.from_dict(payload)

    def test_config_is_frozen(self) -> None:
        """A run must not be able to mutate its own parameters."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            TEST_CONFIG.seed = 5  # type: ignore[misc]

    def test_resource_counts_map_is_a_copy(self) -> None:
        """Mutating the returned map must not affect the config."""
        world_counts = TEST_CONFIG.resource_counts_map
        world_counts["crane"] = 999

        assert TEST_CONFIG.resource_counts_map["crane"] != 999

    def test_tunnelling_config_rejected(self) -> None:
        """A speed that crosses more than one cell per tick is invalid."""
        with pytest.raises(ValueError, match="one cell per tick"):
            TEST_CONFIG.evolve(robot_speed=100.0, dt=0.05)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("grid_width", 0),
            ("obstacle_density", 1.0),
            ("robot_count", 0),
            ("dt", 0.0),
            ("duration", -1.0),
            ("max_deferrals", -1),
            ("task_arrival_rate", 0.0),
            ("robot_speed", 0.0),
            ("battery_drain_rate", -0.1),
        ],
    )
    def test_invalid_parameters_rejected(self, field_name: str, value: float) -> None:
        """Out-of-domain parameters must raise rather than run silently."""
        with pytest.raises(ValueError):
            TEST_CONFIG.evolve(**{field_name: value})
