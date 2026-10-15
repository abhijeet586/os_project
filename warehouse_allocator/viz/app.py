"""The interactive application loop.

The loop separates *simulation* stepping from *rendering* completely. The
simulation advances in fixed ``dt`` increments driven by an accumulator; the
renderer draws whenever there is time to, interpolating between the last two
simulation states. Frame rate therefore has no influence whatsoever on the
simulated result, which is what allows a rendered run and a headless run of the
same seed to finish bit-identical.
"""

from __future__ import annotations

import pygame

from warehouse_allocator.simulation import Simulation
from warehouse_allocator.viz.controls import LiveControls
from warehouse_allocator.viz.hud import Hud
from warehouse_allocator.viz.renderer import Renderer

__all__ = ["run_app"]

HUD_WIDTH = 300
TARGET_FPS = 60
# Cap on simulation steps per frame, so a slow machine or a high speed
# multiplier degrades the frame rate rather than freezing the window.
MAX_STEPS_PER_FRAME = 4000


def run_app(simulation: Simulation, cell_size: int = 8) -> Simulation:
    """Run the simulation with rendering and live controls.

    Args:
        simulation: The run to display.
        cell_size: Pixels per grid cell.

    Returns:
        The simulation, once the window closes or the duration elapses.
    """
    pygame.init()
    pygame.display.set_caption("Autonomous Warehouse Fleet Allocator")

    world = simulation.world
    width = world.grid.width * cell_size + HUD_WIDTH
    height = max(world.grid.height * cell_size, 620)
    surface = pygame.display.set_mode((width, height))

    renderer = Renderer(surface, cell_size)
    hud = Hud(surface, HUD_WIDTH)
    controls = LiveControls(simulation)
    controls.renderer = renderer
    clock = pygame.time.Clock()

    dt = simulation.config.dt
    accumulator = 0.0
    running = True

    renderer.capture(world)
    while running and simulation.clock.now() < simulation.config.duration:
        frame_seconds = clock.tick(TARGET_FPS) / 1000.0

        for event in pygame.event.get():
            if not controls.handle(event):
                running = False

        if controls.single_step:
            renderer.capture(world)
            simulation.step()
            controls.single_step = False
            accumulator = 0.0
        elif not controls.paused:
            accumulator += frame_seconds * controls.speed
            steps = 0
            while accumulator >= dt and steps < MAX_STEPS_PER_FRAME:
                renderer.capture(world)
                simulation.step()
                accumulator -= dt
                steps += 1
                if simulation.clock.now() >= simulation.config.duration:
                    break

        alpha = min(1.0, accumulator / dt) if dt > 0 else 1.0
        renderer.draw(world, alpha, controls.active_radius)
        hud.draw(simulation, controls, world.grid.width * cell_size)
        pygame.display.flip()

    pygame.quit()
    return simulation
