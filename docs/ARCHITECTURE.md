# Architecture

## Module map

```
warehouse_allocator/
  config.py            SimConfig: frozen, validated, JSON-serializable
  simulation.py        Wires the layers; owns the per-tick operation order
  __main__.py          CLI entry point

  core/                Simulation primitives. Depends on nothing else here.
    rng.py             RNGProvider, name-keyed independent sub-streams
    clock.py           SimClock, fixed timestep, drift-free
    geometry.py        Grid, distances, deterministic A*
    entities.py        Robot, Resource, RobotState, ResourceClass
    tasks.py           Task, bundle templates, acquisition orders
    task_generator.py  Poisson arrivals, replayable TaskTrace
    lifecycle.py       Robot task state machine
    world.py           Layout, entity registry, kinematics, state_hash

  alloc/               The two-layer allocator. Consumes core.
    state.py           Available / Max / Allocation / Need matrices
    safety.py          Banker's safety algorithm (pure, unmodified)
    policies.py        AdmissionPolicy ABC + the four implementations
    leases.py          Lease grant, expiry, revocation bookkeeping
    allocator.py       Orchestration, deferrals, force-admission

  metrics/             Measurement. Consumes core; alloc emits into it.
    events.py          Immutable event types
    collector.py       Append-only recorder
    analysis.py        Derived statistics (post-run only)
    plots.py           Matplotlib figures

  viz/                 Rendering. Consumes everything, is consumed by nothing.
    renderer.py        Pygame grid, entities, claim lines
    hud.py             Live statistics panel, tamper banner
    controls.py        Runtime parameter adjustment
    app.py             The interactive loop

  experiments/         Headless harness. Consumes everything except viz.
    runner.py          Multi-policy, multi-seed runs, CIs, CSVs
    sweep.py           Parameter sweeps
```

## Dependency rule

Dependencies point **downward only**:

```
        viz  ─────────┐
                      ▼
  experiments ──► simulation ──► alloc ──► metrics.events
                      │            │
                      └──────► core ◄──────┘
```

- `core` imports nothing else in the package.
- `alloc` imports `core` and `metrics.events` (inert value types).
- `viz` imports everything; **nothing imports `viz`.**
- `core/lifecycle.py` needs an allocator but must not import `alloc`, which
  would be a cycle. It declares `AllocatorProtocol`, a structural `Protocol`,
  and the real allocator is injected. This is why the lifecycle can be tested
  against a stub with no allocation machinery at all.

---

## Data flow through one tick

The order is fixed in `Simulation.step()` and is **part of the determinism
contract** — changing it changes results, so it lives in one place and is
documented rather than emerging from call order:

```
1. Admit arrivals    tasks whose created_at has been reached enter the queue
2. Expire leases     so a resource freed this tick is grantable this tick
3. Lifecycle         each robot, in ascending id order:
                       IDLE     -> accept task, declare Max, -> WAITING
                       WAITING  -> request next class in the acquisition order
                                   (holding everything already granted)
                       MOVING   -> on arrival, mark occupied, -> WORKING
                       WORKING  -> count down service, then next resource
                       DROPOFF  -> on arrival, release bundle atomically
4. Kinematics        world.step(dt): advance positions, drain battery
5. Clock             tick last, so events in a tick carry its start time
```

Robots are always iterated in ascending id order, so competition for a scarce
resource resolves identically on every run.

---

## The determinism contract

Every number this project reports rests on this. It is enforced by test, not by
convention.

**The guarantee.** The same seed produces a bit-identical run: identical final
`state_hash()`, identical event log, across reruns, across processes, and
between headless and rendered execution.

**What makes it hold:**

| Mechanism | Where | Why |
|---|---|---|
| All randomness through `RNGProvider` | `core/rng.py` | No bare `random` anywhere; asserted by AST scan, not grep |
| Name-keyed RNG sub-streams | `core/rng.py` | Adding a new consumer never shifts existing draw sequences |
| Stable name hashing (BLAKE2b, not `hash()`) | `core/rng.py` | Python randomises string hashing per process |
| A\* heap key `(f, h, counter, node)` | `core/geometry.py` | Unique keys; equal-cost frontiers pop FIFO, never by set order |
| Fixed neighbour tuple | `core/geometry.py` | Never a set, never runtime-built |
| Sets used for membership only | throughout | Never iterated where order could reach output |
| Entities iterated by sorted id | `core/world.py` | Insertion order cannot leak in |
| `now = step_index * dt` | `core/clock.py` | Repeated float addition drifts over 72,000 ticks |
| `state_hash` packs raw IEEE-754 bytes | `core/world.py` | Compares true bit equality, not display equality |
| One trace per seed, replayed | `experiments/runner.py` | Policies never see different workloads |

**The single highest-risk point is A\* tie-breaking.** Resolved by set iteration
order it would be nondeterministic; the explicit push counter in the heap key
makes ties resolve first-in-first-out. Verified by
`test_astar_deterministic` (100 identical searches over a map with many
equal-cost paths collapse to exactly one path) and cross-checked across five
`PYTHONHASHSEED` values in separate processes.

---

## The rendering boundary

**Rule: the simulation package never imports Pygame.** Rendering consumes
simulation state; it never participates in producing it.

Enforced by `tests/test_headless_parity.py::TestNoPygameInCore`, which walks the
**transitive** import graph of `core/` and `alloc/` and asserts no rendering
library is reachable. Transitive matters: a direct-import check would pass
happily while `core` pulled in Pygame through an intermediate module. A
companion test asserts the scanner actually detects a real Pygame import, so a
broken scanner cannot report success forever.

The behavioural counterpart, `test_headless_matches_rendered`, runs the same
seed twice — once headless, once with a live renderer and HUD drawing at an
irregular cadence with varying interpolation — and asserts identical final
`state_hash()` and identical event log signature.

The two tests are complementary: the structural one makes the behavioural one
hard to break by accident.

**How rendering stays out of the way:** the simulation advances in fixed `dt`
increments driven by an accumulator, while the renderer draws whenever there is
time, interpolating between the last two simulation states. Frame rate therefore
has no influence on the result. The renderer's only interaction with the world
is `capture()`, which reads positions.

---

## The measurement boundary

`metrics/collector.py` is **append-only** and computes no derived statistics
during a run. Everything in `analysis.py` is reconstructed afterwards from the
event log.

Two reasons. It keeps the simulation hot path free of measurement bias. And it
means a new question can be answered from an old run without re-simulating —
the per-class held-but-unused breakdown, for instance, is derived purely from
`RequestGranted` / `ResourceOccupied` / `ResourceReleased` / `LeaseRevoked`
timestamps, and could be recomputed from a saved log years later.

`ParameterChanged` is the one event with a side effect: it permanently sets
`collector.comparable = False`. Interactive tweaking is useful for intuition,
but such a run must never be reportable as data.

---

## Testing map

| File | Enforces |
|---|---|
| `test_determinism.py` | Seeds, RNG stream isolation, A\* determinism, no bare `random`, clock drift, layout connectivity, config provenance |
| `test_tasks.py` | Bundle overlap, order inconsistency, trace round-trip and replay, partial holds |
| `test_safety.py` | Textbook safe/unsafe verdicts, purity, matrix invariants under random operation sequences |
| `test_allocator.py` | Circular wait refusal, force-admission, **force-admission still safety-checked**, lease revocation during partial hold, policy rules |
| `test_headless_parity.py` | Headless/rendered parity, transitive Pygame exclusion |
