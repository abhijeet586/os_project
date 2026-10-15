"""Seeded random number generation with independent named sub-streams.

Every stochastic draw in the simulation goes through this module. Bare
``random`` module calls are forbidden package-wide and a test enforces it.

The design goal is stronger than "same seed reproduces the same run". It is
that *adding a new consumer of randomness never shifts the draw sequence of
existing consumers*. Without that, appending one new feature reshuffles every
prior result and makes cross-version comparison meaningless.

Deviation from BUILD_SPEC.md, deliberate:
    The spec asks for sub-streams derived by ``SeedSequence.spawn``. Call-order
    ``spawn()`` does not deliver the property the spec asks it to deliver:
    children are handed out in call order, so introducing a ``stream()`` call
    earlier in program flow shifts every stream created after it. This module
    instead derives each stream from a *name-keyed* ``spawn_key``, which is the
    same underlying mechanism ``spawn`` uses but indexed by a stable hash of
    the stream name rather than by a call counter. Registration order then has
    no effect at all.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np

__all__ = ["REQUIRED_STREAMS", "RNGProvider"]

# Streams the simulation is expected to use. Listed for discoverability and
# asserted by tests; the provider will mint any name on demand.
REQUIRED_STREAMS: Final[tuple[str, ...]] = (
    "spawn",
    "task_gen",
    "noise",
    "layout",
)

# Width of the name-derived spawn key. 8 bytes keeps the key comfortably
# inside the integer domain SeedSequence accepts while making accidental
# collisions between stream names practically impossible.
_NAME_DIGEST_BYTES: Final[int] = 8


def _name_key(name: str) -> int:
    """Map a stream name to a stable integer spawn key.

    ``hash()`` is unusable here: Python randomises string hashing per process
    unless ``PYTHONHASHSEED`` is pinned, which would make stream derivation
    depend on an environment variable rather than on the seed.

    Args:
        name: Stream name, for example ``"task_gen"``.

    Returns:
        A non-negative integer derived from a BLAKE2b digest of the name,
        stable across processes, platforms, and Python versions.
    """
    digest = hashlib.blake2b(
        name.encode("utf-8"), digest_size=_NAME_DIGEST_BYTES
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


class RNGProvider:
    """Owns the master seed and mints independent named generators.

    Each named stream is an independent generator derived from the master
    seed and the stream's name. Two streams never share state, so heavy
    consumption by one leaves the others untouched.

    Streams are cached: repeated calls to :meth:`stream` with the same name
    return the *same* generator object, so a consumer that draws in a loop
    advances a single sequence rather than restarting it.

    Attributes:
        seed: The master seed this provider was constructed with.
    """

    __slots__ = ("_seed", "_streams")

    def __init__(self, seed: int) -> None:
        """Initialize the provider.

        Args:
            seed: Master seed for the entire run. Recorded in run provenance
                so any result can be regenerated.
        """
        self._seed: Final[int] = seed
        self._streams: dict[str, np.random.Generator] = {}

    @property
    def seed(self) -> int:
        """Return the master seed.

        Returns:
            The seed passed at construction.
        """
        return self._seed

    def stream(self, name: str) -> np.random.Generator:
        """Return the generator for a named sub-stream.

        Args:
            name: Stream name. Any string is accepted; see
                :data:`REQUIRED_STREAMS` for the ones the simulation uses.

        Returns:
            The cached :class:`numpy.random.Generator` for ``name``, created
            on first request.

        Raises:
            ValueError: If ``name`` is empty.
        """
        if not name:
            raise ValueError("stream name must be a non-empty string")
        existing = self._streams.get(name)
        if existing is not None:
            return existing
        sequence = np.random.SeedSequence(
            entropy=self._seed, spawn_key=(_name_key(name),)
        )
        generator = np.random.default_rng(sequence)
        self._streams[name] = generator
        return generator

    def fresh(self, name: str) -> np.random.Generator:
        """Return a new generator for ``name``, ignoring and clearing cache.

        Useful for regenerating a reproducible artefact (such as a task trace)
        without having to rebuild the whole provider.

        Args:
            name: Stream name to reset.

        Returns:
            A generator positioned at the start of that stream's sequence.
        """
        self._streams.pop(name, None)
        return self.stream(name)

    def __repr__(self) -> str:
        """Return a debug representation naming the seed and live streams.

        Returns:
            A string of the form ``RNGProvider(seed=42, streams=[...])``.
        """
        names = sorted(self._streams)
        return f"RNGProvider(seed={self._seed}, streams={names})"
