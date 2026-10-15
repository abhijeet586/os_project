# PROGRESS.md

Build log for the Autonomous Warehouse Fleet Allocator. One entry per stage,
appended at the stage's verification gate.

---

## Stage 1 — Foundation and Determinism (2026-08-31)

**Status: PASSED.** Both gate commands green.

### Environment

The machine had only Python 3.9.6 (Xcode's bundled interpreter) and no package
manager. `uv` 0.12.7 was installed to `~/.local/bin` and used to provision
CPython **3.12.14** into `.venv/`, satisfying the spec's 3.11+ requirement.

| Component | Version |
|---|---|
| Python | 3.12.14 |
| numpy | 2.5.2 |
| pytest | 9.1.1 |
| mypy | 2.3.1 |
| ruff | 0.16.5 |
| black | 26.5.1 |

### Files created

| File | Purpose |
|---|---|
| `warehouse_allocator/__init__.py` | Package root, layer dependency rules |
| `warehouse_allocator/config.py` | Frozen `SimConfig` + provenance serialization |
| `warehouse_allocator/core/__init__.py` | Core subpackage docstring |
| `warehouse_allocator/core/rng.py` | `RNGProvider`, name-keyed sub-streams |
| `warehouse_allocator/core/clock.py` | `SimClock`, fixed timestep, drift-free |
| `warehouse_allocator/core/geometry.py` | `Grid`, distances, deterministic A* |
| `warehouse_allocator/core/entities.py` | `Robot`, `Resource`, state enums |
| `warehouse_allocator/core/world.py` | Layout generation, registry, kinematics, `state_hash` |
| `tests/__init__.py` | Test package marker |
| `tests/test_determinism.py` | 47 tests covering the determinism contract |
| `pyproject.toml` | Build metadata and mypy/ruff/black/pytest config |

Empty package directories were also created for `alloc/`, `metrics/`, `viz/`,
and `experiments/`, plus `docs/` and `results/`, matching the target structure.

### Verification output

```
$ .venv/bin/python -m pytest tests/test_determinism.py -v
============================= test session starts ==============================
platform darwin -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
collected 47 items
...
============================== 47 passed in 0.09s ==============================

$ .venv/bin/python -m mypy --strict warehouse_allocator/
Success: no issues found in 8 source files
```

The four tests the spec names by requirement, all passing:

1. `test_identical_seeds_identical_state` — 1000 steps twice at seed 1234,
   compared at every 100-step checkpoint as well as at the end.
2. `test_stream_isolation` — 100 draws from `noise` and `layout` leave the
   `task_gen` sequence bit-identical.
3. `test_astar_deterministic` — 100 searches over a 20x20 open grid (which has
   an enormous number of equal-cost corner-to-corner routes) collapse to
   exactly one distinct path. Repeated with a wall forcing a detour.
4. `test_no_bare_random` — no package module imports the stdlib `random`.

### Additional verification beyond the gate

The spec warns that intermittent A* determinism is the project's top failure
mode, so two checks were run that the gate does not require:

* **Flakiness.** The suite was run 10 consecutive times: `47 passed` every
  time, no variation.
* **Cross-process determinism.** The same 1000-step run and the same A* query
  were executed in five separate processes under `PYTHONHASHSEED` values
  `0, 1, 42, 12345, 99999`. All five produced the identical state hash
  (`ebff17ebfcd627dc893b8627ed10a245...`) and the identical path. This matters
  because Python randomises string hashing per process by default; the
  in-process test alone would not have caught a dependence on it.

A work-verification probe confirmed the simulation is not trivially idle:
over 1000 steps, all 8 robots moved, total net displacement 167.1 m, and
battery fell by exactly 0.075 (50 simulated seconds × 0.0015 drain rate).

### Deviations from the spec, with reasons

1. **RNG sub-stream derivation.** The spec asks for streams derived by
   `SeedSequence.spawn`, justified as making "adding a new consumer never
   shift the sequence of existing ones." Call-order `spawn()` does not provide
   that property: children are issued in call order, so inserting a
   `stream()` call earlier in program flow shifts every stream created after
   it. `core/rng.py` instead derives each stream from a *name-keyed*
   `SeedSequence(entropy=seed, spawn_key=(blake2b(name),))` — the same
   underlying mechanism, indexed by a stable hash of the stream name rather
   than by a call counter. Registration order then has no effect at all.
   `test_stream_order_independent` and
   `test_new_consumer_does_not_shift_existing` assert the property the spec
   actually wanted. A stable BLAKE2b digest is used rather than `hash()`
   because Python randomises string hashing per process.

2. **`test_no_bare_random` parses instead of grepping.** The spec says to grep
   for `import random`. A grep produces false positives on `numpy.random` and
   on the word appearing in prose, and false negatives on unusual spacing.
   The test walks the `ast` of every package module and inspects real import
   nodes.

3. **`pyproject.toml` created in Stage 1.** It is listed as a Stage 6
   deliverable, but mypy, ruff, black, and pytest all need configuration to
   run the Stage 1 gate meaningfully. Only tool configuration and build
   metadata were written; `requirements.txt` and the `Makefile` remain for
   Stage 6.

### Design decisions worth recording

* **One grid cell = one metre.** Grid coordinates and metric positions share a
  scale, so no conversion factor appears anywhere in the simulation.
* **4-connected grid.** With unit step costs the Manhattan heuristic is both
  admissible and consistent, and open aisles then contain very large numbers
  of genuinely equal-cost paths — the exact condition that would expose
  unstable tie-breaking. Diagonal movement would have weakened the test.
* **A* tie-break key is `(f, h, counter, node)`.** The strictly increasing
  push counter makes every heap key unique, so `heapq` never compares node
  tuples to resolve a tie and equal-cost pushes pop first-in-first-out. The
  rationale is written out in the `astar` docstring as the spec requires.
* **Racking is rectangular blocks, not scattered cells.** The experiment
  depends on euclidean distance diverging from true path cost; isolated
  one-cell obstacles are walked around trivially and would leave
  `RadiusPolicy` and `ETAPolicy` nearly indistinguishable.
* **Disconnected pockets are pruned.** After layout generation, every free
  cell outside the largest connected component is blocked. A `None` from A*
  during a run therefore always signals a bug, never an unlucky layout.
* **`state_hash` encodes floats as raw IEEE-754 bytes** via `struct.pack`,
  not as formatted decimals, so the determinism test compares true bit
  equality rather than display equality. Sets are sorted before encoding.
* **Config rejects `robot_speed * dt > 1.0`.** A robot crossing more than one
  cell per tick could tunnel through one-cell-thick racking and silently
  invalidate every path-cost measurement. `test_robots_do_not_tunnel_through_racking`
  asserts the cap holds in motion.
* **Simulated time is `step_index * dt`, never accumulated.** Repeated float
  addition of 0.05 drifts visibly over the 72,000 ticks of a one-hour run;
  `test_clock_does_not_drift` pins this.
* **Stage 1 wander behaviour.** With no tasks yet, idle robots are routed to a
  random passable cell from the `noise` stream. This exists to exercise A*,
  the RNG, and kinematics together under the determinism test; Stage 2's task
  lifecycle takes over route assignment.

### Not built in this stage, by design

No allocation logic, no task or bundle types, no metrics, no rendering. Those
are Stages 2 through 5.

### Stop point

Stage 1's verification gate passes. Awaiting instruction to execute Stage 2.

---

## Stages 2-6 (2026-08-31)

Executed in one session at the user's explicit instruction, overriding the
spec's one-stage-per-invocation rule. Every stage's verification gate was still
run in order, and no stage was started before the previous gate passed.

### Stage 2 - Tasks and resource bundles: PASSED

Files: `core/tasks.py`, `core/task_generator.py`, `core/lifecycle.py`,
`tests/test_tasks.py`. `core/entities.py` and `core/world.py` extended.

```
$ pytest tests/test_tasks.py -q
21 passed
```

**Key design point, and the one most easily got wrong.** It is not enough for
bundles to overlap. If every robot acquired classes in one global order,
circular wait would be *structurally impossible* - that is ordered-resource
deadlock **prevention**, and it would make the Banker's checker dead weight,
which is exactly the failure section 2.1 warns against. So each template
declares its own acquisition order and the orders deliberately contradict each
other:

| Template | Order | Holds | Waits for |
|---|---|---|---|
| HEAVY_LIFT | crane -> elevator | crane | elevator |
| FLOOR_TRANSFER | elevator -> dock | elevator | dock |
| FULL_CYCLE | dock -> crane -> elevator | dock | crane |

`test_acquisition_orders_are_inconsistent` asserts no global ordering can
explain the templates, so the property cannot be silently lost by a later edit.

Deviation: `Task` carries a mutable `timestamps` log as the spec asks, but the
authoritative lifecycle record is the event log. `TaskTrace.materialize()`
hands each policy fresh copies so one run's marks cannot leak into the next
while the workload stays bit-identical.

### Stage 3 - Allocator core: PASSED

Files: `alloc/state.py`, `alloc/safety.py`, `alloc/policies.py`,
`alloc/leases.py`, `alloc/allocator.py`, `simulation.py`,
`metrics/events.py`, `metrics/collector.py`, `tests/test_safety.py`,
`tests/test_allocator.py`.

```
$ pytest tests/test_allocator.py tests/test_safety.py -q
39 passed
$ mypy --strict warehouse_allocator/
Success: no issues found in 21 source files
```

All seven required tests present and passing, including
`test_force_admit_still_safety_checked` (a force-admitted request that would
close a deadlock cycle is still refused) and
`test_lease_revocation_partial_hold` (elevator returns, crane retained, `Need`
restored, invariants hold).

Deviation: `metrics/events.py` and `metrics/collector.py` were written during
Stage 3 rather than Stage 4, because Stage 3 requires every decision to emit a
structured record and inventing a throwaway record type first would have been
wasted work.

### Stage 4 - Metrics and experiment harness: PASSED

Files: `metrics/analysis.py`, `metrics/plots.py`, `experiments/runner.py`,
`experiments/sweep.py`.

**A real defect was found and fixed at this gate.** The first full run produced
21 completions out of 635 tasks and 760,287 deferral events. Diagnosis rather
than guesswork: every deferral carried the reason `no free instance`, all 12
resources were held by 6 robots while 22 robots waited, and null/radius/eta
produced byte-identical results. Availability, not admission, was the binding
constraint, so the admission layer was never consulted and the experiment had
no subject. Two fixes:

1. Resource counts rebalanced (crane 6, elevator 6, dock 8 against 24 robots)
   so contention comes from the admission layer.
2. A per-robot retry cooldown (`next_request_at`, 0.5 s) so a blocked robot no
   longer re-requests every tick. Event volume fell ~30x with no change to any
   outcome.

Post-fix the four policies differentiate. Grid, arrival rate and lease duration
were retuned on measured evidence, not taste.

Deviation: figures are rendered via the data-visualisation method (form first,
colour last). The four-policy categorical palette was **validated with the
checker**, not eyeballed: worst adjacent CVD separation 9.1 dE, worst
normal-vision separation 22.9 dE, both clear of their gates. The contrast WARN
is discharged by the visible value labels and legends every figure carries.

### Stage 5 - Visualization and live controls: PASSED

Files: `viz/renderer.py`, `viz/hud.py`, `viz/controls.py`, `viz/app.py`,
`__main__.py`, `tests/test_headless_parity.py`.

```
$ pytest tests/test_headless_parity.py -q
8 passed
$ python -m warehouse_allocator --policy lease --seed 42 --duration 600
(runs; windowed path exercised under SDL_VIDEODRIVER=dummy, no display here)
```

`test_no_pygame_in_core` walks the **transitive** import graph, not just direct
imports - a direct check would pass while `core` reached Pygame through an
intermediate module. A companion test asserts the scanner really does detect a
Pygame import, so a broken scanner cannot report success forever.

Robot-count and arrival-rate controls required `ResourceState.rebuild_for()`
and trace-tail resampling. Removal only ever takes an idle, empty-handed robot;
removing a holder would strand its resources.

### Stage 6 - Documentation and packaging: PASSED

Files: `README.md`, `docs/ALGORITHM.md`, `docs/RESULTS.md`,
`docs/ARCHITECTURE.md`, `requirements.txt` (pinned), `Makefile`.

```
$ make lint && make test && make experiment && make figures
LINT OK        (ruff clean, black clean, mypy --strict clean, 32 files)
TEST OK        115 passed
EXPERIMENT OK  48 run CSVs + all_runs.csv + summary.csv
FIGURES OK     10 files (5 figures x PNG + PDF)
```

`make experiment` defaults to the same 12 seeds `docs/RESULTS.md` reports, so
the artefacts on disk always match the published numbers.

**Statistical note.** Because every policy runs the *identical* trace per seed,
comparisons are **paired**, and `RESULTS.md` uses paired t-tests on per-seed
differences. This is both the correct test and far more powerful than comparing
independent confidence intervals - the `lease` conclusions are invisible under
the weaker test, whose intervals overlap.

### Findings (12 seeds, 3600 s, identical replayed traces)

| Policy | Throughput /h | Held-but-unused | Paired vs null |
|---|---|---|---|
| null | 107.33 +/- 13.17 | 0.754 | baseline |
| radius | 100.25 +/- 17.58 | 0.770 | -7.08, p=0.175 ns |
| eta | 105.25 +/- 15.28 | 0.759 | -2.08, p=0.662 ns |
| lease | 75.25 +/- 14.34 | 0.618 | -32.08, p=0.00002 SIGNIFICANT |

The spec requires at least one adverse finding. There are four, and they are
the main result: the baseline wins on throughput; radius (the originally
proposed idea) has no measurable effect; fixing radius's known obstacle-blindness
(ETA) also has no effect; and leasing reduces early-binding waste significantly
(-0.136, p=0.0004) while costing 30% of throughput. The sweep shows the benefit
and the cost are the same mechanism - at lease=600 s the policy has become the
baseline on both metrics. Leasing also makes dock waste significantly *worse*
(+0.027, p=0.006), which an aggregate-only report would have hidden.

---

## FINAL CHECKLIST

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Same seed reproduces identical results | PASS | `test_identical_seeds_identical_state`; also verified across 5 `PYTHONHASHSEED` values in separate processes |
| 2 | Headless and rendered identical | PASS | `test_headless_matches_rendered`: identical `state_hash()` and event-log signature |
| 3 | `mypy --strict` clean | PASS | "Success: no issues found in 32 source files" |
| 4 | All pytest tests pass | PASS | 115 passed |
| 5 | Deadlock constructible and provably prevented | PASS | `test_circular_wait_refused`, `test_force_admit_still_safety_checked`; 0 of 48 runs deadlocked |
| 6 | All four policies on an identical replayed trace | PASS | 367.3 tasks created per run, identical across all four policies, seed for seed |
| 7 | >= 5 seeds with confidence intervals | PASS | 12 seeds, 95% t-intervals, plus paired t-tests |
| 8 | Throughput is the headline metric | PASS | First figure, first results table; wait is secondary and percentile-only |
| 9 | Held-but-unused measured and reported per policy | PASS | `resource_hold_breakdown`, fig5, per-class table in RESULTS.md |
| 10 | Starvation counted, not hidden | PASS | Force-admissions reported per policy (null 149.4 ... lease 1903.7) |
| 11 | Every runtime parameter in both UI and CLI | PASS | policy, radius, eta, lease, max_deferrals, robots, rate + speed/pause/step and display toggles |
| 12 | >= 1 finding contradicting the hypothesis | PASS | Four; see RESULTS.md sections 4.1-4.4 |

No item fails.

### Known limitations, stated rather than buried

Documented in `docs/RESULTS.md` section 9. The most important: the lifecycle
acquires an entire bundle *before* travelling, which follows the spec and
maximises early-binding waste by construction. That single assumption is the
most likely driver of the headline numbers, and a lifecycle interleaving
acquisition with travel would probably show a different lease tradeoff. The
`radius` and `eta` results are correctly stated as *no detected effect*, not as
proof of no effect; an effect smaller than about +/-11 tasks/h would not have
been detected at 12 seeds.
