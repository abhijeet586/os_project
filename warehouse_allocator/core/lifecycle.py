"""Robot task lifecycle: accept, acquire incrementally, serve, release.

The lifecycle a robot walks per task:

1. Accept a task from the queue.
2. Request the bundle's classes **one at a time**, in the template's
   acquisition order, **holding whatever has already been granted** while
   requesting the remainder.
3. Travel to the pickup, then to each held resource in turn, occupying each on
   arrival and performing its share of the service.
4. Travel to the dropoff.
5. Release the entire bundle atomically.

Step 2 is the load-bearing one. A robot that released its holdings whenever it
blocked could never participate in a deadlock, the Banker's safety check would
never fire, and the project would have no subject. So a blocked robot sits in
``WAITING`` holding everything it has already been granted, and retries the
next class on a later tick.

Note also what step 3 costs. Every resource in the bundle is held from the
moment it is granted, but the robot can only be physically present at one of
them at a time. Everything it holds and has not yet reached is locked and
idle -- that is early binding, and measuring it is the point of the project.

This module depends only on ``core``. The allocator is injected as a
:class:`AllocatorProtocol`, so the simulation layer can wire in the real
two-layer allocator without ``core`` ever importing ``alloc``.
"""

from __future__ import annotations

from typing import Protocol

from warehouse_allocator.core.entities import (
    Resource,
    ResourceClass,
    Robot,
    RobotState,
)
from warehouse_allocator.core.tasks import Task
from warehouse_allocator.core.world import World

__all__ = ["AllocatorProtocol", "LifecycleManager"]


class AllocatorProtocol(Protocol):
    """The allocator surface the lifecycle depends on.

    Declared structurally so ``core`` never imports ``alloc``.
    """

    def request(
        self, robot: Robot, resource_class: ResourceClass, task: Task
    ) -> Resource | None:
        """Request one instance of a resource class for a robot.

        Args:
            robot: The requesting robot, which may already hold resources.
            resource_class: The class being requested.
            task: The task the request serves.

        Returns:
            The bound resource if granted, or ``None`` if deferred or refused.
        """
        ...

    def declare_task(self, robot: Robot, task: Task) -> None:
        """Declare a robot's maximum claim as it accepts a task.

        Args:
            robot: The accepting robot.
            task: The accepted task, whose bundle becomes the claim.
        """
        ...

    def release_all(self, robot: Robot) -> None:
        """Release every resource a robot holds, atomically.

        Args:
            robot: The robot whose bundle is being returned.
        """
        ...

    def on_arrival(self, robot: Robot, resource: Resource) -> None:
        """Note that a holder has physically reached a resource.

        Args:
            robot: The arriving robot.
            resource: The resource it has reached.
        """
        ...


class LifecycleObserver(Protocol):
    """Hooks the metrics layer implements to observe lifecycle transitions."""

    def task_started(self, robot: Robot, task: Task, when: float) -> None:
        """Record a robot accepting a task.

        Args:
            robot: The accepting robot.
            task: The accepted task.
            when: Simulated time.
        """
        ...

    def task_completed(self, robot: Robot, task: Task, when: float) -> None:
        """Record a robot completing a task.

        Args:
            robot: The completing robot.
            task: The completed task.
            when: Simulated time.
        """
        ...

    def state_changed(self, robot: Robot, previous: RobotState, when: float) -> None:
        """Record a robot state transition.

        Args:
            robot: The robot whose state changed.
            previous: The state being left.
            when: Simulated time.
        """
        ...


class LifecycleManager:
    """Drives every robot's task state machine.

    Attributes:
        world: The world whose robots are driven.
        allocator: The injected allocator.
        observer: Optional metrics observer.
    """

    __slots__ = (
        "_pending",
        "_tasks",
        "allocator",
        "observer",
        "retry_interval",
        "world",
    )

    def __init__(
        self,
        world: World,
        allocator: AllocatorProtocol,
        observer: LifecycleObserver | None = None,
        retry_interval: float = 0.5,
    ) -> None:
        """Initialize the lifecycle manager.

        Args:
            world: World holding the robots and resources.
            allocator: Allocator to route requests through.
            observer: Optional observer notified of lifecycle transitions.
            retry_interval: Simulated seconds a blocked robot waits before
                re-issuing a deferred request.
        """
        self.world = world
        self.allocator = allocator
        self.observer = observer
        self.retry_interval = retry_interval
        self._tasks: dict[int, Task] = {}
        self._pending: list[Task] = []

    @property
    def queue_depth(self) -> int:
        """Return the number of tasks waiting for a robot.

        Returns:
            Count of unassigned tasks.
        """
        return len(self._pending)

    def task(self, task_id: int) -> Task:
        """Look up a task by id.

        Args:
            task_id: Identifier to fetch.

        Returns:
            The matching task.

        Raises:
            KeyError: If no such task is registered.
        """
        return self._tasks[task_id]

    def submit(self, task: Task) -> None:
        """Add an arrived task to the queue.

        Args:
            task: The task to enqueue.
        """
        self._tasks[task.task_id] = task
        self._pending.append(task)

    def update(self, now: float, dt: float) -> None:
        """Advance every robot's lifecycle by one tick.

        Robots are processed in ascending id order so that competition for a
        scarce resource resolves identically on every run.

        Args:
            now: Current simulated time.
            dt: Timestep in simulated seconds.
        """
        for robot in self.world.robots:
            self._update_robot(robot, now, dt)

    def _update_robot(self, robot: Robot, now: float, dt: float) -> None:
        """Advance one robot's lifecycle.

        Args:
            robot: The robot to advance.
            now: Current simulated time.
            dt: Timestep in simulated seconds.
        """
        if robot.current_task_id is None:
            self._try_accept(robot, now)
            return

        task = self._tasks[robot.current_task_id]
        if robot.state is RobotState.WAITING:
            self._try_acquire(robot, task, now)
        elif robot.state is RobotState.MOVING_TO_RESOURCE:
            self._advance_to_resource(robot, task, now)
        elif robot.state is RobotState.WORKING:
            self._advance_service(robot, task, now, dt)
        elif robot.state is RobotState.MOVING_TO_DROPOFF:
            self._advance_to_dropoff(robot, task, now)

    def _set_state(self, robot: Robot, state: RobotState, now: float) -> None:
        """Transition a robot and notify the observer.

        Args:
            robot: The robot to transition.
            state: The new state.
            now: Current simulated time.
        """
        if robot.state is state:
            return
        previous = robot.state
        robot.state = state
        if self.observer is not None:
            self.observer.state_changed(robot, previous, now)

    def _try_accept(self, robot: Robot, now: float) -> None:
        """Assign a queued task to an idle robot.

        Args:
            robot: The idle robot.
            now: Current simulated time.
        """
        if not self._pending:
            return
        task = self._pending.pop(0)
        robot.current_task_id = task.task_id
        robot.bundle_cursor = 0
        robot.visit_cursor = 0
        robot.service_remaining = 0.0
        robot.next_request_at = now
        self.allocator.declare_task(robot, task)
        task.mark("assigned", now)
        if self.observer is not None:
            self.observer.task_started(robot, task, now)
        self._set_state(robot, RobotState.WAITING, now)

    def _try_acquire(self, robot: Robot, task: Task, now: float) -> None:
        """Request the next class in the bundle, holding what is already won.

        A refusal leaves the robot in ``WAITING`` still holding its partial
        bundle. That is deliberate: releasing on block would dissolve every
        deadlock before it could form.

        Args:
            robot: The requesting robot.
            task: Its current task.
            now: Current simulated time.
        """
        order = task.acquisition_order
        if robot.bundle_cursor >= len(order):
            self._begin_travel(robot, task, now)
            return

        # A blocked robot that re-asks every tick produces one deferral event
        # per tick per robot, which swamps the log without changing any
        # outcome. The cooldown keeps retries frequent enough to be responsive
        # and rare enough to stay readable.
        if now < robot.next_request_at:
            return

        resource_class = order[robot.bundle_cursor]
        task.mark("first_request", now)
        granted = self.allocator.request(robot, resource_class, task)
        if granted is None:
            robot.next_request_at = now + self.retry_interval
            return

        robot.bundle_cursor += 1
        if robot.bundle_cursor >= len(order):
            task.mark("bundle_complete", now)
            self._begin_travel(robot, task, now)

    def _begin_travel(self, robot: Robot, task: Task, now: float) -> None:
        """Start the pickup leg once the full bundle is held.

        Args:
            robot: The robot to route.
            task: Its current task.
            now: Current simulated time.
        """
        robot.visit_cursor = 0
        if not self.world.route_to(robot, task.pickup):
            # Connectivity is guaranteed by layout pruning, so an unreachable
            # pickup means the task was built against a different layout.
            self._finish_task(robot, task, now)
            return
        self._set_state(robot, RobotState.MOVING_TO_RESOURCE, now)

    def _visit_targets(self, robot: Robot) -> tuple[Resource, ...]:
        """Return the held resources to visit, in a stable order.

        Args:
            robot: The holding robot.

        Returns:
            Held resources sorted by id, so the visit route never depends on
            set iteration order.
        """
        return tuple(
            self.world.resource(resource_id)
            for resource_id in sorted(robot.held_resources)
        )

    def _advance_to_resource(self, robot: Robot, task: Task, now: float) -> None:
        """Handle arrival at the pickup or at a held resource.

        Args:
            robot: The travelling robot.
            task: Its current task.
            now: Current simulated time.
        """
        if not self.world.at_destination(robot):
            return

        targets = self._visit_targets(robot)
        if robot.visit_cursor >= len(targets):
            self._begin_dropoff(robot, task, now)
            return

        resource = targets[robot.visit_cursor]
        if robot.cell != resource.cell:
            if not self.world.route_to(robot, resource.cell):
                self._begin_dropoff(robot, task, now)
            return

        # Arrival is what converts a held resource into a used one. Everything
        # before this instant, since the grant, was early-binding waste.
        self.allocator.on_arrival(robot, resource)
        task.mark(f"arrived_{resource.resource_id}", now)
        robot.service_remaining = task.service_duration / max(len(targets), 1)
        self._set_state(robot, RobotState.WORKING, now)

    def _advance_service(self, robot: Robot, task: Task, now: float, dt: float) -> None:
        """Count down service at the current resource.

        Args:
            robot: The working robot.
            task: Its current task.
            now: Current simulated time.
            dt: Timestep in simulated seconds.
        """
        robot.service_remaining -= dt
        if robot.service_remaining > 0.0:
            return

        robot.service_remaining = 0.0
        robot.visit_cursor += 1
        targets = self._visit_targets(robot)
        if robot.visit_cursor >= len(targets):
            self._begin_dropoff(robot, task, now)
            return

        following = targets[robot.visit_cursor]
        if not self.world.route_to(robot, following.cell):
            self._begin_dropoff(robot, task, now)
            return
        self._set_state(robot, RobotState.MOVING_TO_RESOURCE, now)

    def _begin_dropoff(self, robot: Robot, task: Task, now: float) -> None:
        """Start the final delivery leg.

        Args:
            robot: The robot to route.
            task: Its current task.
            now: Current simulated time.
        """
        if not self.world.route_to(robot, task.dropoff):
            self._finish_task(robot, task, now)
            return
        self._set_state(robot, RobotState.MOVING_TO_DROPOFF, now)

    def _advance_to_dropoff(self, robot: Robot, task: Task, now: float) -> None:
        """Complete the task once the dropoff is reached.

        Args:
            robot: The travelling robot.
            task: Its current task.
            now: Current simulated time.
        """
        if self.world.at_destination(robot):
            self._finish_task(robot, task, now)

    def _finish_task(self, robot: Robot, task: Task, now: float) -> None:
        """Release the bundle atomically and return the robot to idle.

        Args:
            robot: The completing robot.
            task: The completed task.
            now: Current simulated time.
        """
        self.allocator.release_all(robot)
        task.mark("completed", now)
        if self.observer is not None:
            self.observer.task_completed(robot, task, now)
        robot.current_task_id = None
        robot.bundle_cursor = 0
        robot.visit_cursor = 0
        robot.service_remaining = 0.0
        robot.destination = None
        robot.clear_path()
        self._set_state(robot, RobotState.IDLE, now)
