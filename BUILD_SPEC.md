# BUILD_SPEC.md

# Autonomous Warehouse Fleet Allocator
## Executable build specification for Claude Code

---

## 0. HOW TO USE THIS FILE

Place this file at the root of an empty repository. Then run:

```
claude
> Read BUILD_SPEC.md and execute Stage 1. Stop when Stage 1 verification passes.
```

Then, one at a time:

```
> Execute Stage 2. Stop when Stage 2 verification passes.
```

Do not ask Claude Code to execute all stages in one command. Each stage has a
verification gate that must pass before the next begins. Stage 1's determinism
guarantee is load-bearing for every result the project produces.

### Rules for the agent

1. Execute exactly one stage per invocation. Stop at the verification gate.
2. Before writing code for a stage, restate the stage's deliverables and the
   verification criteria in your own words. If anything is ambiguous, ask
   before writing.
3. Run the verification commands yourself. Paste the real output. Do not claim
   a test passes without showing it pass.
4. If a verification fails, fix it and re-run before reporting. Do not proceed
   to the next stage with a failing gate.
5. At the end of each stage, append a dated entry to `PROGRESS.md` listing
   files created, verification output, and any deviation from this spec with
   the reason.
6. If you believe a requirement in this spec is wrong, say so and explain why
   before implementing. Do not silently substitute your own design.

---

## 1. WHAT THIS PROJECT IS

A 2D simulation measuring how spatial admission control affects resource
allocation throughput in a warehouse robot fleet.

The problem being studied: in classical resource allocation, a process that is
granted a resource uses it immediately. In a physical warehouse, a robot may be
granted a crane while 200 metres away from it. The crane sits locked and idle
for the entire travel time while nearby robots starve. This is an early binding
problem.

The experiment: compare four admission policies on identical workloads and
measure the resulting throughput, wait distribution, and specifically the
held-but-unused resource time that early binding wastes.

This is a research simulation producing defensible numbers, not a visual demo.
Every design decision below serves reproducibility and measurement.

---

## 2. DESIGN CORRECTIONS (MANDATORY)

The originating design document contained six errors. These corrections
override it. Do not implement the original versions.

### 2.1 Multi-resource task bundles are required

Deadlock requires a process to hold one resource while blocking on another. If
robots only ever hold a single resource, deadlock is structurally impossible
and the Banker's safety algorithm is dead weight over a simple FIFO queue.

Therefore: tasks must require **bundles** of resources acquired **incrementally**.
A robot requests one resource class, holds what it is granted, and continues
requesting the rest while holding. At least three of the four task templates
must require two or more distinct resource classes, and the bundles must
overlap so that circular wait is constructible.

### 2.2 Correct naming

This is **not** a modified Banker's Algorithm. It is a two-layer allocator:

```
request -> [AdmissionPolicy] -> [SafetyChecker (standard Banker's)] -> grant/defer
```

The Banker's safety algorithm is used unmodified. The novelty is the admission
layer in front of it. Name the classes `AdmissionPolicy` and `SafetyChecker`.
Any documentation calling this a Banker's variant is wrong and will be
dismissed by anyone who knows the algorithm.

### 2.3 Four pluggable admission policies, not one

Implement all four behind a single interface:

| Policy | Rule | Rationale | Known failure mode |
|---|---|---|---|
| `NullPolicy` | Admit everything | Baseline. Reproduces classical behaviour. | Full early-binding waste. |
| `RadiusPolicy` | Admit if euclidean distance <= r | The originally proposed idea. | Straight-line distance ignores obstacles. A robot 20m away behind a wall has worse ETA than one 200m down a clear aisle. |
| `ETAPolicy` | Admit if A* path cost to resource <= threshold | Fixes radius's blindness to topology. | Path cost computation is expensive at scale. |
| `LeasePolicy` | Admit anyone, grant a lease with an arrival deadline, auto-revoke and return to pool on expiry | The theoretically correct fix. Attacks early binding directly rather than by proxy. | Revocation churn under heavy contention. |

The comparison between these four **is** the experimental result. Building only
the radius policy produces no evidence that the radius idea is correct.

### 2.4 Starvation mitigation is mandatory

Any admission filter starves agents in low-traffic regions. Every deferred
request carries a `deferral_count`. After `max_deferrals` (default 5) the
request is force-admitted, bypassing the admission policy but **never** the
safety check. Every force-admission is logged and reported as a first-class
metric.

### 2.5 Throughput is the primary metric

Mean wait time is trivially gamed: reject aggressively and rejected robots stop
counting as waiting. The headline metric is
`tasks_completed_per_simulated_hour`.

Wait time is a secondary diagnostic, reported as p50 / p95 / p99 / max, never
as a bare mean. Wait is measured from **first request** to **final grant**,
inclusive of all deferrals, so that rejection cannot hide as improvement.

### 2.6 Fixed timestep, decoupled from rendering

Simulation advances in fixed `dt` increments. Rendering interpolates between
simulation states and may run at any framerate or be disabled entirely.

Requirement: results must be **bit-identical** across runs with the same seed,
and identical between headless and rendered runs. This is asserted by test.

The most common source of nondeterminism in this class of project is A*
tie-breaking resolved by set iteration order. Break ties on a stable explicit
key. Verify this specifically.

---

## 3. ENGINEERING STANDARDS

Applies to every stage.

- Python 3.11+.
- Type hints on every signature. `mypy --strict` passes clean.
- Google-style docstrings on every module, class, and public method.
- Inline comments explain **why**, never restate **what**. A comment that
  paraphrases the line below it is noise; delete it.
- No global mutable state. Policies and collaborators are injected.
- Every stochastic draw goes through the seeded RNG provider. No bare `random`
  module calls anywhere in the codebase. Grep for it before each verification.
- **The simulation package must never import Pygame.** Rendering consumes
  simulation state. It never participates in it. Enforce with a test that
  walks the import graph.
- pytest for all tests. Tests live in `tests/`, mirroring the package layout.
- `ruff` for linting, `black` for formatting.

---

## 4. TARGET STRUCTURE

```
warehouse_allocator/
  __init__.py
  __main__.py                CLI entry point
  config.py                  SimConfig frozen dataclass
  core/
    __init__.py
    rng.py                   Seeded RNG provider with named sub-streams
    clock.py                 Fixed-timestep simulation clock
    geometry.py              Grid, distances, deterministic A*
    entities.py              Robot, Resource
    tasks.py                 Task, bundle templates
    task_generator.py        Poisson arrivals, TaskTrace save/load
    world.py                 Grid state, entity registry, kinematics
  alloc/
    __init__.py
    state.py                 Available / Max / Allocation / Need matrices
    safety.py                Banker's safety algorithm (pure)
    policies.py              AdmissionPolicy ABC + four implementations
    leases.py                Lease grant, expiry, revocation
    allocator.py             Orchestrator, deferral queue, force-admission
  metrics/
    __init__.py
    events.py                Immutable event types
    collector.py             Event-sourced recorder
    analysis.py              Derived statistics
    plots.py                 Matplotlib figures
  viz/
    __init__.py
    renderer.py              Pygame grid and entity rendering
    hud.py                   Live statistics panel
    controls.py              Runtime-adjustable parameters
  experiments/
    __init__.py
    runner.py                Multi-policy, multi-seed headless runs
    sweep.py                 Parameter sweeps
tests/
docs/
results/
```

---

## STAGE 1: FOUNDATION AND DETERMINISM

Build the deterministic simulation core. No allocation logic. No rendering.

### Deliverables

**`config.py`**
Frozen dataclass `SimConfig`. Every tunable parameter with a docstring line
explaining its effect and a defensible default:
grid dimensions, obstacle density, robot count, resource counts per class,
`dt` (default 0.05 simulated seconds), sim duration, seed, policy name, policy
parameters (`radius`, `eta_threshold`, `lease_duration`), `max_deferrals`
(default 5), task arrival rate, robot speed, battery drain rate.
Include `from_dict`, `to_dict`, and `to_json` for run provenance.

**`core/rng.py`**
`RNGProvider` wrapping `numpy.random.default_rng(seed)`. Exposes named
sub-streams via `provider.stream("task_gen")`, each derived by
`SeedSequence.spawn` so that adding a new consumer never shifts the sequence of
existing ones. Required streams: `spawn`, `task_gen`, `noise`, `layout`.
This design is what makes results stable across code changes, not just across
reruns of identical code.

**`core/clock.py`**
`SimClock`: fixed `dt`, `tick()`, `now()` in simulated seconds, `step_index`.
No wall-clock time anywhere in the simulation package.

**`core/geometry.py`**
Grid coordinates, euclidean and manhattan distance, deterministic A* with
obstacle support.

A* determinism requirements, non-negotiable:
- Open set is a heap keyed by `(f_score, h_score, tie_break_counter, node)`.
- Neighbour expansion order is a fixed constant tuple, not a set.
- Never iterate a `set` or an unordered `dict` in a way that affects output.
Write the tie-breaking rationale as a comment. This is the single highest-risk
correctness point in the entire project.

**`core/entities.py`**
`Robot`: id, position, velocity, current path, `RobotState` enum
(`IDLE`, `MOVING_TO_RESOURCE`, `WAITING`, `WORKING`, `MOVING_TO_DROPOFF`,
`CHARGING`), battery level, current task, set of held resources.
`Resource`: id, resource class, position, capacity, current holders.

**`core/world.py`**
Grid, obstacle layout generated from the `layout` RNG stream, entity registry,
`step(dt)` advancing kinematics only. Provide `state_hash()` returning a stable
hash of all mutable simulation state, used by determinism tests.

### Verification gate

```bash
pytest tests/test_determinism.py -v
mypy --strict warehouse_allocator/
```

Required tests:
1. `test_identical_seeds_identical_state`: run 1000 steps twice with the same
   seed, assert identical `state_hash()`.
2. `test_stream_isolation`: draw 100 values from an unused stream, assert the
   `task_gen` sequence is unchanged.
3. `test_astar_deterministic`: run A* 100 times on a map with multiple
   equal-cost paths, assert identical path returned every time.
4. `test_no_bare_random`: grep the package source, assert no `import random`.

---

## STAGE 2: TASKS AND RESOURCE BUNDLES

Build the workload. This is what makes the allocator non-trivial.

### Deliverables

**`core/tasks.py`**
`Task`: id, required bundle (`dict[ResourceClass, int]`), pickup coordinate,
dropoff coordinate, service duration, creation timestamp, lifecycle timestamp
log.

Four task templates minimum. At least three must require 2+ distinct resource
classes, and the bundles must overlap. Suggested:

| Template | Bundle |
|---|---|
| `HEAVY_LIFT` | crane 1, elevator 1 |
| `FLOOR_TRANSFER` | elevator 1, dock 1 |
| `FULL_CYCLE` | crane 1, elevator 1, dock 1 |
| `RECHARGE` | dock 1 |

The overlap between `HEAVY_LIFT` and `FLOOR_TRANSFER` on `elevator` is what
makes circular wait constructible. Document this in a comment.

**`core/task_generator.py`**
Poisson arrival process driven by the `task_gen` stream. Generates a complete
`TaskTrace` ahead of the run. `TaskTrace.save(path)` and `TaskTrace.load(path)`
to JSON.

**Trace replay is mandatory.** The trace is generated once from the seed and
replayed identically across every policy. Policies must never see different
workloads. A comparison on different workloads is not a comparison.

**Robot task lifecycle**, implemented in `world.py` or a dedicated
`core/lifecycle.py`:
1. Accept task from queue.
2. Request resource classes in bundle **one at a time**, holding what is
   granted while requesting the remainder.
3. Travel to resource, occupy, perform service.
4. Travel to dropoff.
5. Release the entire bundle atomically.

Step 2 holding-while-requesting is the precondition for deadlock. If a robot
releases what it holds when blocked, there is no deadlock to avoid and the
entire project has no subject.

### Verification gate

```bash
pytest tests/test_tasks.py -v
```

Required tests:
1. `test_trace_roundtrip`: generate 1000 tasks, save, load, assert equality.
2. `test_trace_deterministic`: same seed produces identical trace.
3. `test_partial_hold`: construct a state where a robot holds resource class A
   while blocked requesting class B. Assert it holds A throughout.
4. `test_bundles_overlap`: assert at least two templates share a resource class.

---

## STAGE 3: ALLOCATOR CORE

The heart of the project.

### Deliverables

**`alloc/state.py`**
`ResourceState` managing NumPy matrices `Available`, `Max`, `Allocation`,
`Need`, indexed by explicit id-to-index mappings, never by positional
assumption. Assert the invariant `Need == Max - Allocation` after every
mutation. Assert `Allocation <= Max` elementwise. Assert no negative entries.

**`alloc/safety.py`**
`SafetyChecker.is_safe(state, robot_id, request) -> bool` implementing the
standard Banker's safety algorithm. Pure function, no side effects.

Include a docstring that walks the algorithm step by step with the standard
work-vector and finish-vector formulation. This is the piece a reader will
scrutinize hardest. Make it explicit and readable.

**`alloc/policies.py`**
```python
class AdmissionPolicy(ABC):
    @abstractmethod
    def should_admit(
        self, robot: Robot, resource: Resource, world: World, clock: SimClock
    ) -> AdmissionDecision: ...
```
`AdmissionDecision` carries a verdict enum and a reason string for logging.
Implement `NullPolicy`, `RadiusPolicy`, `ETAPolicy`, `LeasePolicy` per section 2.3.
`ETAPolicy` uses **actual A* path cost**, not straight-line distance. That
distinction is the entire reason it exists.

**`alloc/leases.py`**
Lease grant with arrival deadline. Expiry checked every tick. On expiry:
revoke, return the resource to the pool, update the Banker's matrices
correctly, increment the revocation metric.

Revocation during a **partial bundle hold** is the trickiest correctness point
in the project. A robot holding a crane whose elevator lease expires must
return the elevator without corrupting `Allocation` or losing the crane. Write
a dedicated test for exactly this case.

**`alloc/allocator.py`**
Request flow:
```
1. AdmissionPolicy.should_admit()
2. If deferred: increment deferral_count, enqueue
3. If deferral_count > max_deferrals: force-admit (skip policy, NOT safety)
4. SafetyChecker.is_safe()
5. If safe: grant. If unsafe: defer regardless of admission verdict.
```

**The safety check is never bypassed. Not for force-admission, not for priority
override, not for low battery, not for anything.** Bypassing it reintroduces
deadlock, which is the failure mode the entire allocator exists to prevent.
State this in a comment at the bypass site so no future edit removes it by
accident.

Every decision emits a structured log record: timestamp, robot id, resource
class, policy verdict, policy reason, safety verdict, deferral count.

### Verification gate

```bash
pytest tests/test_allocator.py tests/test_safety.py -v
mypy --strict warehouse_allocator/
```

Required tests:
1. `test_safety_known_safe`: standard OS textbook safe-state matrix returns True.
2. `test_safety_known_unsafe`: known unsafe matrix returns False.
3. `test_circular_wait_refused`: construct a genuine circular wait, assert the
   allocator refuses the request that would complete the cycle.
4. `test_force_admit_after_max_deferrals`: a request deferred `max_deferrals`
   times is admitted on the next attempt.
5. `test_force_admit_still_safety_checked`: a force-admitted request that would
   create an unsafe state is still refused.
6. `test_lease_revocation_partial_hold`: robot holds A, lease on B expires,
   assert B returns to pool, A is retained, matrices remain consistent.
7. `test_matrix_invariants`: property-based test asserting
   `Need == Max - Allocation` holds after random valid operation sequences.

---

## STAGE 4: METRICS AND EXPERIMENT HARNESS

Build measurement before visuals. The numbers are the deliverable.

### Deliverables

**`metrics/events.py`**
Immutable event dataclasses: `TaskCreated`, `RequestIssued`, `RequestDeferred`,
`RequestGranted`, `RequestForceAdmitted`, `RequestRefusedUnsafe`,
`LeaseGranted`, `LeaseRevoked`, `ResourceOccupied`, `ResourceReleased`,
`TaskCompleted`, `RobotStateChanged`, `ParameterChanged`. All carry simulated
timestamps.

**`metrics/collector.py`**
Event-sourced. Append only. **Never compute derived statistics during the run.**
Everything is derived afterward from the event log. This keeps the simulation
hot path free of measurement bias and lets you re-derive new metrics from old
runs without re-simulating.

**`metrics/analysis.py`**

Primary:
- `tasks_completed_per_simulated_hour`

Secondary diagnostics:
- Wait time first-request to final-grant: p50, p95, p99, max. Percentiles are
  mandatory. A bare mean hides exactly the tail damage admission filtering causes.
- Robot state time breakdown as fractions: moving, waiting, working, idle.
- Per resource class: fraction of time **held**, and fraction of time
  **held-but-unused** (granted but the holder has not yet arrived).
  **The gap between these two is the central finding of the project.** It is
  the direct measurement of early-binding waste. Give it a named function and
  a prominent place in the output.
- Starvation: force-admission count, distribution of deferral counts.
- Lease revocation count and revocation rate, where applicable.
- Deadlock avoidance interventions: count of `RequestRefusedUnsafe`.

**`experiments/runner.py`**
Runs one replayed trace against all four policies, headless, sequentially.
Accepts a seed list. Aggregates across seeds with mean and 95% confidence
intervals. Minimum 5 seeds for any reported result. Single-seed results prove
nothing and must not appear in `RESULTS.md`.
Emits one CSV per run plus a combined summary CSV in `results/`.
Every CSV carries the full `SimConfig` JSON in a header comment for provenance.

**`experiments/sweep.py`**
Parameter sweep over `radius`, `eta_threshold`, `lease_duration`. Outputs the
tradeoff curve. There will be an optimum; reporting one arbitrary parameter
value is weak evidence.

**`metrics/plots.py`**
Matplotlib. All figures saved to `results/` as PNG and PDF at 300 dpi, with
axis labels including units:
1. Grouped bar: throughput by policy, with confidence interval error bars.
2. Box plot: wait time distribution by policy.
3. Line: parameter sweep curves.
4. Stacked area: robot state fractions over simulated time.
5. Grouped bar: held versus held-but-unused time per resource class per policy.
   This is the money figure. Make it clear.

### Verification gate

```bash
python -m warehouse_allocator.experiments.runner --seeds 1,2,3,4,5 --duration 3600
```

Must produce the summary CSV and all five figures. Report the actual numbers in
your stage summary, including which policy won on throughput and by how much.

---

## STAGE 5: VISUALIZATION AND LIVE CONTROLS

Pygame consumes simulation state. It never mutates it.

### Deliverables

**`viz/renderer.py`**
Top-down grid with obstacles. Resources colour-coded by class, shaded by
occupancy. Robots colour-coded by state. **Lines drawn from each robot to any
resource it has claimed but not yet reached**, so early binding is visible on
screen rather than only in the metrics. Interpolate positions between
simulation steps so motion is smooth at any framerate.

**`viz/hud.py`**
Live panel: simulated time, tasks completed, current throughput rate, per-class
resource utilization, active policy name, deferral queue depth, force-admission
count, lease revocation count.

**`viz/controls.py`**
All adjustable at runtime:
- Policy switch, cycling all four live
- `radius`, `eta_threshold`, `lease_duration` sliders
- `max_deferrals` slider
- Robot count and task arrival rate sliders
- Speed multiplier, including pause, single-step, fast-forward-headless
- Toggles: show A* paths, show claim lines, show proximity radii

**Tamper marking.** Any mid-run parameter change emits `ParameterChanged` and
marks the run `comparable=False` in its metrics header. Display a persistent
warning banner once this happens. Interactively-tweaked runs must never be
reportable as experimental data.

### Verification gate

```bash
pytest tests/test_headless_parity.py -v
python -m warehouse_allocator --policy lease --seed 42 --duration 600
```

Required tests:
1. `test_headless_matches_rendered`: same seed, headless and rendered, assert
   identical final `state_hash()` and identical event log.
2. `test_no_pygame_in_core`: walk the import graph of `core/` and `alloc/`,
   assert Pygame appears nowhere.

---

## STAGE 6: DOCUMENTATION AND PACKAGING

### Deliverables

**`README.md`**
What the project is. The early-binding problem in plain language with the
200-metre crane example. Install. How to run the sim. How to run experiments.
How to read the figures.

**`docs/ALGORITHM.md`**
The two-layer allocator explained in full. Banker's safety algorithm walked
through with a worked numeric example showing a safe and an unsafe state.
Each admission policy with its theoretical justification and its failure mode.
An explicit section stating this is admission control in front of Banker's,
**not** a modified Banker's, and why the distinction matters.

**`docs/RESULTS.md`**
Actual findings from actual runs. Numbers, not claims.

Requirement: **include at least one result where the proposed approach loses.**
If radius filtering underperforms leasing, state it with the numbers. If
admission filtering reduces throughput at low contention, state that. A results
document in which the proposed method wins on every metric is not a result, it
is a marketing page, and any competent reader will treat it as one.

If genuinely no adverse finding appeared, say so explicitly and explain what
you tested that could have produced one.

**`docs/ARCHITECTURE.md`**
Module map. Data flow. The determinism contract. The rule that simulation never
imports rendering, and the test that enforces it.

**Packaging**
- `requirements.txt`, pinned versions
- `pyproject.toml`
- `Makefile`: `make test`, `make run`, `make experiment`, `make sweep`,
  `make lint`, `make figures`
- CLI:
  `python -m warehouse_allocator --policy lease --seed 42 --headless --duration 3600`

### Verification gate

```bash
make lint && make test && make experiment && make figures
```

All green. All figures regenerated. `RESULTS.md` reflects the regenerated numbers.

---

## FINAL CHECKLIST

Verify each and report pass or fail with evidence. Do not mark the project
complete with any item failing.

1. Same seed produces identical results across reruns.
2. Headless and rendered runs produce identical results.
3. `mypy --strict` passes clean.
4. All pytest tests pass.
5. A deadlock is provably constructible in the workload, and the safety checker
   provably prevents it. Show the test output.
6. All four policies ran on an identical replayed trace.
7. Results aggregated over at least 5 seeds with confidence intervals.
8. Throughput, not wait time, is the reported headline metric.
9. Held-but-unused resource time is measured and reported per policy.
10. Starvation events are counted and reported, not hidden.
11. Every runtime-adjustable parameter is exposed in both UI and CLI.
12. `RESULTS.md` reports at least one finding contradicting the original
    hypothesis, or explicitly states none was found and why that is credible.

---

## KNOWN FAILURE MODES TO WATCH FOR

Listed because they are the ones that will actually happen.

- **Nondeterministic A*.** Caused by set iteration or unstable heap tie-breaks.
  Symptom: Stage 1 determinism test passes intermittently. If it is flaky, it
  is broken. Fix it before Stage 2 or every downstream comparison is noise.
- **Matrix corruption on lease revocation during partial hold.** Symptom:
  `Need != Max - Allocation` assertion fires deep into a long run. The
  invariant assertion exists to catch this early; do not disable it for speed.
- **Metrics computed during the run.** Symptom: you cannot answer a new
  question without re-simulating. Keep the collector append-only.
- **Policies compared on different workloads.** Symptom: results shift when you
  change unrelated code. The trace must be generated once and replayed.
- **Reporting mean wait time.** It will show the filter winning while
  throughput drops. Percentiles or nothing.
- **Silently dropping the safety check under force-admission.** Symptom: the
  simulation deadlocks and you spend a day debugging kinematics. The comment at
  the bypass site exists to prevent this.
