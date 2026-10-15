"""Append-only event recorder.

The collector does exactly one thing: append events. It computes **no**
derived statistics during the run.

That restraint is deliberate. Deriving statistics on the hot path both slows
the simulation and quietly bakes in whichever questions were thought of first;
answering a new one then means re-simulating. With an append-only log, every
statistic in :mod:`warehouse_allocator.metrics.analysis` is recomputable from
an old run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from warehouse_allocator.metrics.events import Event, ParameterChanged

__all__ = ["EventCollector"]

E = TypeVar("E", bound=Event)


class EventCollector:
    """Records simulation events in arrival order.

    Attributes:
        comparable: Whether this run may be reported as experimental data.
            Set ``False`` permanently by any mid-run parameter change.
    """

    __slots__ = ("_events", "comparable")

    def __init__(self) -> None:
        """Initialize an empty, comparable collector."""
        self._events: list[Event] = []
        self.comparable = True

    def __len__(self) -> int:
        """Return the number of recorded events.

        Returns:
            Event count.
        """
        return len(self._events)

    @property
    def events(self) -> tuple[Event, ...]:
        """Return every recorded event in order.

        Returns:
            An immutable snapshot of the log.
        """
        return tuple(self._events)

    def record(self, event: Event) -> None:
        """Append one event.

        Args:
            event: The event to record. A :class:`ParameterChanged` also marks
                the run non-comparable, permanently.
        """
        self._events.append(event)
        if isinstance(event, ParameterChanged):
            # Tamper marking is one-way: a run touched by hand can never be
            # promoted back to reportable data.
            self.comparable = False

    def of_type(self, event_type: type[E]) -> tuple[E, ...]:
        """Return every recorded event of one concrete type.

        Args:
            event_type: The event class to filter for.

        Returns:
            Matching events in recorded order.
        """
        return tuple(event for event in self._events if isinstance(event, event_type))

    def count_of(self, event_type: type[Event]) -> int:
        """Count recorded events of one concrete type.

        Args:
            event_type: The event class to count.

        Returns:
            Number of matching events.
        """
        return sum(1 for event in self._events if isinstance(event, event_type))

    def save(self, path: str | Path) -> None:
        """Write the event log to newline-delimited JSON.

        Args:
            path: Destination file path. Parent directories are created.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for event in self._events:
                handle.write(json.dumps(event.to_dict(), sort_keys=True))
                handle.write("\n")

    def signature(self) -> str:
        """Return a stable digest of the whole log.

        Used by the headless-versus-rendered parity test, which must assert
        the event logs are identical and not merely similar in length.

        Returns:
            A BLAKE2b hex digest over every event in order.
        """
        import hashlib

        digest = hashlib.blake2b(digest_size=32)
        for event in self._events:
            digest.update(json.dumps(event.to_dict(), sort_keys=True).encode("utf-8"))
        return digest.hexdigest()
