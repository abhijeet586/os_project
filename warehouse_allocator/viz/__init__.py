"""Pygame rendering and live controls.

Rendering **consumes** simulation state and never participates in producing it.
Pygame is imported only inside this subpackage; ``core`` and ``alloc`` must
never import it, and ``tests/test_headless_parity.py`` walks the import graph
to enforce that.

The practical consequence is that a rendered run and a headless run of the same
seed produce bit-identical results, which is also asserted by test.
"""
