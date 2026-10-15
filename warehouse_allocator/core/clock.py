"""Fixed-timestep simulation clock.

The clock is the only source of time in the simulation package. Wall-clock
time is never consulted: a run that depends on how fast the host machine is
cannot be reproduced, and reproducibility is the property every result here
rests on.

Simulated time is derived as ``step_index * dt`` rather than accumulated by
repeated addition. Repeated float addition drifts, and after the 72,000 ticks
of a one-hour run at the default ``dt`` that drift is large enough to move
event timestamps into a different bucket.
"""

from __future__ import annotations

__all__ = ["SimClock"]


class SimClock:
    """Advances simulated time in fixed increments.

    Attributes:
        dt: The fixed timestep in simulated seconds.
    """

    __slots__ = ("_dt", "_step_index")

    def __init__(self, dt: float) -> None:
        """Initialize the clock at time zero.

        Args:
            dt: Fixed timestep in simulated seconds.

        Raises:
            ValueError: If ``dt`` is not positive.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self._dt = dt
        self._step_index = 0

    @property
    def dt(self) -> float:
        """Return the fixed timestep in simulated seconds.

        Returns:
            The timestep this clock advances by.
        """
        return self._dt

    @property
    def step_index(self) -> int:
        """Return the number of ticks elapsed.

        Returns:
            Zero immediately after construction, incrementing by one per
            :meth:`tick`.
        """
        return self._step_index

    def now(self) -> float:
        """Return the current simulated time.

        Returns:
            Elapsed simulated seconds, computed as ``step_index * dt`` to
            avoid the accumulated float drift of repeated addition.
        """
        return self._step_index * self._dt

    def tick(self) -> float:
        """Advance the clock by exactly one timestep.

        Returns:
            The simulated time *after* advancing.
        """
        self._step_index += 1
        return self.now()

    def reset(self) -> None:
        """Return the clock to time zero."""
        self._step_index = 0

    def __repr__(self) -> str:
        """Return a debug representation of the clock's position.

        Returns:
            A string of the form ``SimClock(dt=0.05, step=120, now=6.0)``.
        """
        return f"SimClock(dt={self._dt}, step={self._step_index}, " f"now={self.now()})"
