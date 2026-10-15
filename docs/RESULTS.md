# Results

All numbers below come from actual runs of the code in this repository. Nothing
is projected, rounded up, or reported from a single seed.

**Headline: the originally proposed idea does not work, and the one policy that
does reduce early-binding waste pays for it with a 30% throughput loss.**

---

## 1. Method

| Setting | Value |
|---|---|
| Seeds | 12 (`1`–`12`) |
| Duration | 3600 simulated seconds per run |
| Runs | 48 (12 seeds x 4 policies) |
| Warehouse | 100 x 70 m, 12% obstacle density (rectangular racking) |
| Fleet | 24 robots at 1.5 m/s |
| Resources | 6 cranes, 6 elevators, 8 docks |
| Arrival rate | 0.10 tasks/s (mean 367.3 tasks arrive per run) |
| Defaults | `radius` 40 m, `eta_threshold` 30 s, `lease_duration` 300 s, `max_deferrals` 5 |

Reproduce with:

```bash
python -m warehouse_allocator.experiments.runner \
    --seeds 1,2,3,4,5,6,7,8,9,10,11,12 --duration 3600 --figures
python -m warehouse_allocator.experiments.sweep --seeds 1,2,3,4,5 --duration 3600
```

**One trace per seed, replayed against all four policies.** Verified: every
policy saw exactly 367.3 tasks on average, seed for seed identical. Because the
workload is shared, policy comparisons are **paired**, so the tests below are
paired t-tests on per-seed differences. This is both the correct test and a
substantially more powerful one than comparing independent confidence intervals
— several conclusions here would be invisible under the weaker test.

---

## 2. Early binding is real, and it is enormous

Under the classical baseline, resources are held **97.8%** of the run — the pool
looks saturated. But **75.4%** of all resource-time is *held-but-unused*: granted
to a robot that had not yet arrived.

> Roughly three quarters of the apparent resource utilisation is fiction.

The allocator's books say the cranes are busy. The warehouse floor says they are
sitting there empty waiting for someone to walk over. This gap is the phenomenon
the project set out to measure, and it is large enough to be worth attacking.

See `results/fig5_held_vs_unused.png`.

---

## 3. Throughput — the headline metric

Tasks completed per simulated hour, 12 seeds, mean with 95% CI:

| Policy | Throughput /h | 95% CI | Paired vs `null` | p | Verdict |
|---|---|---|---|---|---|
| `null` | **107.33** | [94.17, 120.50] | — | — | baseline |
| `radius` | 100.25 | [82.67, 117.83] | −7.08 | 0.175 | not significant |
| `eta` | 105.25 | [89.97, 120.53] | −2.08 | 0.662 | not significant |
| `lease` | 75.25 | [60.91, 89.59] | **−32.08** | **0.00002** | **significantly worse** |

**The baseline wins.** No admission policy improved throughput. Two did nothing
measurable, and one was significantly worse.

---

## 4. Findings that contradict the original hypothesis

The spec asks for at least one result where the proposed approach loses. There
are four.

### 4.1 Radius filtering — the originally proposed idea — does nothing

`RadiusPolicy` was the idea that motivated the project. Measured against the
baseline on identical workloads it is indistinguishable from noise on every
metric:

| Metric | Delta vs `null` | 95% CI | p |
|---|---|---|---|
| Throughput | −7.08 /h | [−17.84, +3.67] | 0.175 |
| Held-but-unused | **+0.015** | [−0.011, +0.042] | 0.234 |
| p95 wait | +82.3 s | [−149.9, +314.5] | 0.452 |

Note the sign on waste: radius filtering made early-binding waste *slightly
worse*, not better, though not significantly so. It certainly did not improve
it. The central claim — that refusing distant requesters recovers wasted
resource time — is **not supported**.

### 4.2 Fixing radius's known flaw does not rescue it

`ETAPolicy` was built to correct radius's blindness to obstacles, and it
genuinely does: `test_eta_uses_path_cost_not_straight_line` shows radius
admitting a robot on the far side of a wall that ETA correctly refuses.

The correction changes nothing at the level of outcomes. ETA is also
statistically indistinguishable from the baseline (−2.08 /h, p = 0.662; waste
+0.005, p = 0.699).

So the topology-blindness of the radius rule was *not* what limited it. Both
policies fail for a deeper reason, discussed in §6.

### 4.3 Leasing works exactly as theory predicts, and it is not worth it

`LeasePolicy` is the only policy that does what the project set out to do:

| Metric | Delta vs `null` | 95% CI | p |
|---|---|---|---|
| Held-but-unused | **−0.136** | [−0.197, −0.076] | **0.0004** |
| Throughput | **−32.08 /h** | [−43.07, −21.09] | **0.00002** |

It cuts early-binding waste from 75.4% to 61.8% — a real, significant, sizeable
reduction, confirming the theoretical argument that attacking the grant-to-
arrival window directly beats attacking it by proxy.

It also cuts throughput by 30%. **The mechanism that recovers the wasted time is
the same mechanism that destroys the throughput**: revoking a lease sends its
robot back to re-request, and the churn costs more than the reclaimed capacity
is worth. Per run it produced 85.7 revocations and 1,775.6 unsafe refusals
(against ~0 for every other policy), because constant grant/revoke cycling keeps
the pool at the edge of exhaustion where Banker's must refuse.

Fleet time spent `WAITING` rose from 74.5% under `null` to 81.0% under `lease`.
The robots are not doing more work; they are queueing more.

### 4.4 Leasing makes one resource class actively worse

Held-but-unused, per class, as a share of that class's held time:

| Class | `null` | `radius` | `eta` | `lease` | lease − null | p |
|---|---|---|---|---|---|---|
| crane | 0.717 | 0.745 | 0.727 | **0.460** | −0.257 | 0.0001 |
| elevator | 0.690 | 0.708 | 0.697 | **0.529** | −0.161 | 0.0015 |
| dock | 0.875 | 0.880 | 0.876 | **0.902** | **+0.027** | **0.006** |

Leasing dramatically improves cranes and elevators, and **significantly degrades
docks**. Docks are the most numerous class (8 instances) and the one every
`RECHARGE` task and both multi-class templates touch, so they turn over fastest;
lease churn there adds re-request overhead without buying back much idle time.

An aggregate-only report would have shown "leasing reduces waste by 13.6 points"
and hidden the fact that it moves one of three classes in the wrong direction.

---

## 5. Wait times

Reported as percentiles, never a bare mean, and measured from *first request* to
*final grant* including every deferral — so refusing a robot cannot disguise
itself as an improvement.

| Policy | p95 wait (s) | 95% CI | Paired vs `null` | p |
|---|---|---|---|---|
| `null` | 1499.2 | [1294.3, 1704.0] | — | — |
| `radius` | 1581.5 | [1300.9, 1862.0] | +82.3 | 0.452 |
| `eta` | 1528.2 | [1265.0, 1791.5] | +29.0 | 0.789 |
| `lease` | 1307.4 | [1152.3, 1462.5] | −191.8 | 0.155 |

**No policy significantly changed wait time.** Lease's apparent 192-second
improvement is not significant, and note what it would have meant if reported
carelessly: lease has the *best* mean wait and the *worst* throughput. This is
precisely the trap the spec warns about — an aggressive policy can look good on
wait while completing fewer tasks. Throughput is the headline metric for this
reason.

---

## 6. What the sweeps show

`results/fig3_parameter_sweep.png`, 5 seeds per point.

| Parameter | Range tested | Throughput range | Shape |
|---|---|---|---|
| `radius` | 10–90 m | 96.8 – 104.2 | flat, all within noise |
| `eta_threshold` | 5–70 s | 96.6 – 109.0 | flat, all within noise |
| `lease_duration` | 60–600 s | 6.4 – 108.8 | steeply monotonic |

**Radius and ETA have no optimum because they have no effect.** Across a
9x range of radius and a 14x range of ETA threshold, throughput never moves
outside the confidence band. These policies are not mistuned; they are inert.

**The lease curve is the most informative result in the project.** Throughput
rises monotonically with lease duration — and so does waste:

| `lease_duration` | Throughput /h | Held-but-unused |
|---|---|---|
| 60 s | 6.4 | 0.698 |
| 120 s | 22.2 | 0.702 |
| 180 s | 58.6 | 0.644 |
| 300 s | 83.2 | 0.631 |
| 450 s | 90.8 | 0.676 |
| 600 s | 108.8 | **0.750** |

At 600 s the lease policy has essentially become the baseline: 108.8 tasks/h
versus `null`'s 107.3, and 0.750 waste versus `null`'s 0.754.

> There is no lease duration that both reduces early-binding waste and preserves
> throughput. Tuning the lease long enough to stop the churn tunes away the
> entire benefit.

At 60 s the policy collapses to 6.4 tasks/h — a 94% throughput loss — because
robots are revoked before they can physically cross the warehouse.

---

## 7. Why the spatial policies fail

The measurements suggest the premise, not the implementation, is what is wrong.

Under the baseline the fleet spends **74.5% of its time in `WAITING`** while
resources sit 97.8% held. The system is not short of *admission discipline*; it
is short of *resources relative to the bundle sizes being demanded*. Robots must
assemble a complete multi-resource bundle before travelling at all, so every
resource in a bundle is necessarily held from grant until the robot has finished
with the earlier ones — the waste is structural to the acquire-then-travel
lifecycle, not an artefact of admitting the wrong robots.

Refusing a distant robot does not help, because the resource simply sits unheld
instead of held-but-unreached. Either way no work happens; the filter converts
"wasted while reserved" into "wasted while idle" and the throughput is the same.
That is exactly what the flat radius and ETA curves show.

This also explains why leasing is the only policy that moves the waste number:
it is the only one that can reclaim a resource *after* a bad grant rather than
trying to predict bad grants in advance.

---

## 8. Deadlock avoidance

Deadlock is genuinely constructible in this workload — the three templates form
a hold-and-wait ring over crane, elevator and dock (see
[ALGORITHM.md §4](ALGORITHM.md#4-why-deadlock-is-constructible-here-at-all)) —
and the safety checker demonstrably prevents it:

- `test_circular_wait_refused` builds the ring and asserts the closing request
  is refused.
- `test_force_admit_still_safety_checked` hammers that same request past the
  force-admission threshold and asserts it is *still* refused.
- No run of the 48 deadlocked.

Unsafe refusals per run: ~0.1 for `null`, `radius` and `eta`; **1,775.6** for
`lease`. Under the three non-revoking policies the system rarely approaches an
unsafe state; lease churn drives it there constantly.

Force-admissions per run: `null` 149.4, `radius` 176.0, `eta` 190.1,
`lease` 1,903.7. As expected, filtering policies starve more robots and so force
more often; these are reported rather than hidden.

---

## 9. Limitations

Honest scope on the above.

1. **One warehouse geometry and one load point.** Results are from a 100x70 m
   layout at 12% obstacle density and 0.10 tasks/s. §7 argues the fleet is
   resource-bound; in a resource-rich, travel-bound regime admission filtering
   could plausibly matter more. That regime was not tested.
2. **The lifecycle acquires the whole bundle before moving.** This follows the
   specification, and it maximises early-binding waste by construction. A
   lifecycle that interleaved acquisition with travel would likely show smaller
   baseline waste and a different lease tradeoff. This is the single assumption
   most likely to be driving the headline numbers.
3. **12 seeds still leaves wide intervals.** The throughput CI is roughly ±13
   tasks/h. The paired tests are what make the `lease` conclusions solid; the
   `radius` and `eta` results are correctly stated as *no detected effect*, not
   as *proof of no effect*. A radius effect smaller than about ±11 tasks/h would
   not have been detected here.
4. **Instance selection is nearest-first for every policy**, which already
   embeds a mild spatial heuristic in the baseline. Part of the reason `radius`
   adds nothing may be that the baseline is not purely spatially naive.
5. **`RECHARGE` tasks do not model battery.** Battery drains but never gates
   behaviour, so the `CHARGING` state is unused.

---

## 10. Summary

| Claim | Verdict |
|---|---|
| Early binding wastes substantial resource time | **Supported** — 75.4% of held time is held-but-unused |
| Radius admission control recovers that time | **Refuted** — no effect (p = 0.234), slightly negative sign |
| Radius fails because it ignores obstacles | **Refuted** — fixing it (ETA) also had no effect |
| Leasing reduces early-binding waste | **Supported** — −13.6 points (p = 0.0004) |
| Leasing is therefore the better policy | **Refuted** — costs 30% throughput (p = 0.00002) |
| A lease duration exists that gets both | **Refuted** — the sweep shows the benefit and cost are one mechanism |
| Any policy improves wait time | **Not supported** — no significant difference |
| Banker's prevents constructible deadlock | **Supported** — refusal tests pass, no run deadlocked |

The admission-control hypothesis, as originally proposed, is not supported by
these measurements. The most valuable output of the project is the
held-versus-held-but-unused instrumentation, which quantifies a real and large
inefficiency, and the demonstration that the obvious spatial fixes for it do not
work in a resource-bound regime.
