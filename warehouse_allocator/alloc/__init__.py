"""Two-layer resource allocation.

The structure is::

    request -> [AdmissionPolicy] -> [SafetyChecker (standard Banker's)] -> grant/defer

This is **not** a modified Banker's algorithm. The safety algorithm in
:mod:`warehouse_allocator.alloc.safety` is the textbook one, unmodified. The
contribution is the admission layer in front of it, which decides whether a
request should even be *considered* based on how far the requester is from the
resource. Calling this a Banker's variant misdescribes it.

Nothing here may import Pygame; a test enforces it.
"""
