"""Poisson task arrivals and replayable task traces.

The workload is generated **once** from the seed and then replayed against
every admission policy. Policies must never see different workloads: a
comparison on different workloads is not a comparison, it is two unrelated
measurements printed next to each other.

:meth:`TaskTrace.materialize` hands each run its own fresh copies of the task
objects, so one policy's lifecycle bookkeeping cannot leak into the next
policy's run while the workload itself stays bit-identical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from warehouse_allocator.config import SimConfig
from warehouse_allocator.core.geometry import Coord
from warehouse_allocator.core.rng import RNGProvider
from warehouse_allocator.core.tasks import (
    TEMPLATE_WEIGHTS,
    Task,
    TaskTemplate,
)

__all__ = ["TaskTrace", "generate_trace"]

# Service durations are drawn from an exponential and then clamped. The raw
# exponential has unbounded tail: a single 400-second draw would monopolise a
# scarce crane for a ninth of the run and swamp the policy effect being
# measured.
_SERVICE_MIN_FACTOR = 0.25
_SERVICE_MAX_FACTOR = 3.0


@dataclass(frozen=True, slots=True)
class TaskTrace:
    """An immutable, replayable workload.

    Attributes:
        seed: The master seed the trace was generated from.
        duration: Simulated seconds the trace spans.
        arrival_rate: Poisson rate in tasks per simulated second.
        tasks: Tasks in ascending ``created_at`` order.
    """

    seed: int
    duration: float
    arrival_rate: float
    tasks: tuple[Task, ...]

    def __len__(self) -> int:
        """Return the number of tasks in the trace.

        Returns:
            Task count.
        """
        return len(self.tasks)

    def materialize(self) -> list[Task]:
        """Return fresh task copies for one run.

        Returns:
            Independent :class:`~warehouse_allocator.core.tasks.Task` objects
            with empty timestamp logs, in arrival order.
        """
        return [task.copy() for task in self.tasks]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace to JSON-compatible types.

        Returns:
            A dict carrying the generation parameters and every task.
        """
        return {
            "seed": self.seed,
            "duration": self.duration,
            "arrival_rate": self.arrival_rate,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskTrace:
        """Reconstruct a trace from :meth:`to_dict` output.

        Args:
            data: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed trace.
        """
        return cls(
            seed=int(data["seed"]),
            duration=float(data["duration"]),
            arrival_rate=float(data["arrival_rate"]),
            tasks=tuple(Task.from_dict(item) for item in data["tasks"]),
        )

    def save(self, path: str | Path) -> None:
        """Write the trace to a JSON file.

        Args:
            path: Destination file path. Parent directories are created.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> TaskTrace:
        """Read a trace from a JSON file.

        Args:
            path: Source file path.

        Returns:
            The loaded trace.
        """
        parsed: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(parsed)


def generate_trace(
    config: SimConfig,
    rng: RNGProvider,
    free_cells: tuple[Coord, ...],
) -> TaskTrace:
    """Generate a complete workload ahead of the run.

    Arrivals follow a Poisson process, realised as exponential inter-arrival
    gaps. Generating the whole trace up front rather than sampling during the
    run is what makes replay across policies exact: no policy can perturb the
    arrival sequence by consuming draws at a different rate.

    Args:
        config: Simulation parameters supplying duration and arrival rate.
        rng: Seeded provider; only the ``task_gen`` stream is consumed.
        free_cells: Passable cells that pickups and dropoffs may occupy.

    Returns:
        The generated :class:`TaskTrace`.

    Raises:
        ValueError: If ``free_cells`` is empty.
    """
    if not free_cells:
        raise ValueError("cannot place tasks on a layout with no free cells")

    stream = rng.fresh("task_gen")
    templates = tuple(TEMPLATE_WEIGHTS)
    weights = np.array(
        [TEMPLATE_WEIGHTS[template] for template in templates], dtype=np.float64
    )
    weights /= weights.sum()

    mean_service = config.service_duration_mean
    lower = mean_service * _SERVICE_MIN_FACTOR
    upper = mean_service * _SERVICE_MAX_FACTOR

    tasks: list[Task] = []
    now = 0.0
    task_id = 0
    while True:
        now += float(stream.exponential(1.0 / config.task_arrival_rate))
        if now >= config.duration:
            break
        template = templates[int(stream.choice(len(templates), p=weights))]
        pickup = free_cells[int(stream.integers(0, len(free_cells)))]
        dropoff = free_cells[int(stream.integers(0, len(free_cells)))]
        service = float(np.clip(stream.exponential(mean_service), lower, upper))
        tasks.append(
            Task(
                task_id=task_id,
                template=TaskTemplate(template),
                pickup=pickup,
                dropoff=dropoff,
                service_duration=service,
                created_at=now,
            )
        )
        task_id += 1

    if config.burst_interval > 0.0 and config.burst_size > 0:
        burst_stream = rng.fresh("burst_gen")
        burst_time = config.burst_interval
        while burst_time < config.duration:
            for _ in range(config.burst_size):
                template = templates[int(burst_stream.choice(len(templates), p=weights))]
                pickup = free_cells[int(burst_stream.integers(0, len(free_cells)))]
                dropoff = free_cells[int(burst_stream.integers(0, len(free_cells)))]
                service = float(np.clip(burst_stream.exponential(mean_service), lower, upper))
                tasks.append(
                    Task(
                        task_id=task_id,
                        template=TaskTemplate(template),
                        pickup=pickup,
                        dropoff=dropoff,
                        service_duration=service,
                        created_at=burst_time,
                    )
                )
                task_id += 1
            burst_time += config.burst_interval
        tasks.sort(key=lambda t: t.created_at)
        for idx, task in enumerate(tasks):
            task.task_id = idx

    return TaskTrace(
        seed=config.seed,
        duration=config.duration,
        arrival_rate=config.task_arrival_rate,
        tasks=tuple(tasks),
    )
