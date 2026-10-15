"""Stage 5 verification: rendering must not touch the simulation.

Two independent guarantees are asserted here:

1. *Behavioural*: a rendered run and a headless run of the same seed produce
   identical final state and identical event logs.
2. *Structural*: ``core`` and ``alloc`` do not import Pygame, transitively.

The second is what makes the first hard to break by accident.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from warehouse_allocator.config import SimConfig
from warehouse_allocator.simulation import Simulation, build_trace

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "warehouse_allocator"

PARITY_CONFIG = SimConfig(
    grid_width=50,
    grid_height=36,
    robot_count=8,
    resource_counts=(("crane", 3), ("elevator", 3), ("dock", 4)),
    duration=120.0,
    seed=42,
    policy="lease",
    task_arrival_rate=0.4,
)

STEPS = 1200


def _headless() -> tuple[str, str, int]:
    """Run the simulation with no renderer attached.

    Returns:
        The final state hash, the event log signature, and the event count.
    """
    trace = build_trace(PARITY_CONFIG)
    simulation = Simulation(PARITY_CONFIG, trace=trace)
    for _ in range(STEPS):
        simulation.step()
    return (
        simulation.state_hash(),
        simulation.collector.signature(),
        len(simulation.collector),
    )


def _rendered() -> tuple[str, str, int]:
    """Run the same simulation with a live renderer drawing every step.

    Uses SDL's dummy video driver so the test runs on a machine with no
    display, which is the normal case in CI.

    Returns:
        The final state hash, the event log signature, and the event count.
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")

    from warehouse_allocator.viz.hud import Hud
    from warehouse_allocator.viz.renderer import Renderer

    pygame.init()
    try:
        surface = pygame.Surface((640, 480))
        renderer = Renderer(surface, cell_size=8)
        hud = Hud(surface, 300)

        trace = build_trace(PARITY_CONFIG)
        simulation = Simulation(PARITY_CONFIG, trace=trace)
        from warehouse_allocator.viz.controls import LiveControls

        controls = LiveControls(simulation)
        controls.renderer = renderer

        for index in range(STEPS):
            renderer.capture(simulation.world)
            simulation.step()
            # Render at a deliberately irregular cadence with varying
            # interpolation, since a renderer that perturbed state would most
            # likely do so through exactly these paths.
            if index % 3 == 0:
                renderer.draw(simulation.world, alpha=(index % 7) / 7.0, radius=25.0)
                hud.draw(simulation, controls, 300)
        return (
            simulation.state_hash(),
            simulation.collector.signature(),
            len(simulation.collector),
        )
    finally:
        pygame.quit()


class TestHeadlessParity:
    """Rendering must be observationally invisible to the simulation."""

    def test_headless_matches_rendered(self) -> None:
        """Same seed, rendered and headless, must agree exactly."""
        headless_hash, headless_signature, headless_count = _headless()
        rendered_hash, rendered_signature, rendered_count = _rendered()

        assert headless_count == rendered_count, "event counts differ"
        assert headless_signature == rendered_signature, "event logs differ"
        assert headless_hash == rendered_hash, "final world state differs"

    def test_headless_is_reproducible(self) -> None:
        """The headless baseline must itself be stable."""
        assert _headless() == _headless()

    def test_run_is_not_trivially_empty(self) -> None:
        """Parity would be vacuous if the run recorded nothing."""
        _, _, count = _headless()

        assert count > 100, f"only {count} events recorded; parity is vacuous"


class TestNoPygameInCore:
    """The simulation layers must not depend on the rendering library."""

    FORBIDDEN = ("pygame", "matplotlib")
    SIMULATION_PACKAGES = ("core", "alloc")

    @staticmethod
    def _imports(path: Path) -> set[str]:
        """Return every module name imported by a source file.

        Args:
            path: Source file to parse.

        Returns:
            Top-level module names imported.
        """
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".")[0])
        return found

    def _local_targets(self, path: Path) -> set[str]:
        """Return the in-package modules a file imports.

        Args:
            path: Source file to parse.

        Returns:
            Dotted module paths inside ``warehouse_allocator``.
        """
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("warehouse_allocator"):
                    targets.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("warehouse_allocator"):
                        targets.add(alias.name)
        return targets

    def _module_path(self, dotted: str) -> Path | None:
        """Resolve a dotted module name to a source file.

        Args:
            dotted: Module path such as ``warehouse_allocator.core.world``.

        Returns:
            The file, or ``None`` if it does not exist.
        """
        relative = (
            dotted.replace("warehouse_allocator", "").strip(".").replace(".", "/")
        )
        candidate = PACKAGE_ROOT / f"{relative}.py"
        if candidate.exists():
            return candidate
        package = PACKAGE_ROOT / relative / "__init__.py"
        return package if package.exists() else None

    @pytest.mark.parametrize("subpackage", SIMULATION_PACKAGES)
    def test_no_pygame_in_core(self, subpackage: str) -> None:
        """Walk the transitive import graph and assert Pygame never appears.

        Transitive, not just direct: a direct-import check would pass happily
        while ``core`` pulled in Pygame through an intermediate module.
        """
        roots = sorted((PACKAGE_ROOT / subpackage).rglob("*.py"))
        assert roots, f"no sources found in {subpackage}/"

        seen: set[Path] = set()
        queue = list(roots)
        offenders: list[str] = []

        while queue:
            path = queue.pop()
            if path in seen:
                continue
            seen.add(path)

            for name in self._imports(path):
                if name in self.FORBIDDEN:
                    offenders.append(f"{path.relative_to(PACKAGE_ROOT)} imports {name}")
            for dotted in self._local_targets(path):
                resolved = self._module_path(dotted)
                if resolved is not None and resolved not in seen:
                    queue.append(resolved)

        assert not offenders, f"{subpackage}/ reaches a rendering library: {offenders}"

    def test_simulation_module_is_clean(self) -> None:
        """The orchestrator must also stay free of rendering imports."""
        found = self._imports(PACKAGE_ROOT / "simulation.py")

        assert not found & set(self.FORBIDDEN)

    def test_guard_detects_a_violation(self) -> None:
        """The scanner must actually catch a Pygame import.

        Without this, a broken scanner would report success forever.
        """
        offender = PACKAGE_ROOT / "viz" / "renderer.py"

        assert "pygame" in self._imports(offender)

    def test_viz_is_where_pygame_lives(self) -> None:
        """Pygame must appear in viz/, confirming the boundary is real."""
        sources = sorted((PACKAGE_ROOT / "viz").rglob("*.py"))
        importers = [path.name for path in sources if "pygame" in self._imports(path)]

        assert importers, "no viz module imports pygame"
