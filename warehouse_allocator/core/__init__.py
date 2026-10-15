"""Deterministic simulation primitives.

This subpackage contains everything the simulation needs to advance state:
seeded randomness, the fixed-timestep clock, grid geometry with deterministic
A*, entity definitions, and the world that owns them.

Nothing here may import Pygame, matplotlib, or any other rendering or
presentation library. Rendering consumes this state; it never participates
in producing it.
"""
