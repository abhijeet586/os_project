"""Autonomous warehouse fleet allocator.

A deterministic 2D simulation measuring how spatial admission control affects
resource allocation throughput in a warehouse robot fleet.

The package is split into layers that may only depend downward:

* :mod:`warehouse_allocator.core` -- simulation primitives (RNG, clock,
  geometry, entities, world). Depends on nothing else in the package.
* :mod:`warehouse_allocator.alloc` -- admission policies and the Banker's
  safety checker. Consumes ``core``.
* :mod:`warehouse_allocator.metrics` -- event-sourced measurement.
* :mod:`warehouse_allocator.viz` -- Pygame rendering. Consumes simulation
  state and never participates in it.

Neither ``core`` nor ``alloc`` may import Pygame; a test enforces this.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
