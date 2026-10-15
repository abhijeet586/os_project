# The two-layer allocator

## 1. This is not a modified Banker's algorithm

It needs saying plainly, because the distinction is the whole point and getting
it wrong invites a reader who knows the algorithm to stop reading.

The structure is:

```
request -> [ AdmissionPolicy ] -> [ SafetyChecker: standard Banker's ] -> grant / defer
```

Two separate layers, with separate jobs:

- **`SafetyChecker`** is the Banker's safety algorithm exactly as published. Not
  a variant, not an extension, not "Banker's with a distance term". It is in
  [`alloc/safety.py`](../warehouse_allocator/alloc/safety.py) and it answers one
  question: *would granting this leave a state from which every robot can still
  finish?*
- **`AdmissionPolicy`** is the new part. It sits in front and answers a
  different question: *should this request even be considered right now, given
  where the robot physically is?*

They are not interchangeable and neither substitutes for the other:

| | `AdmissionPolicy` | `SafetyChecker` |
|---|---|---|
| Question | Should we consider this now? | Is granting this safe? |
| Knows about | Space, distance, travel time | Resource counts and claims only |
| Can grant? | **No** — it can only defer | Yes |
| Can be bypassed? | Yes, by force-admission | **Never, under any circumstance** |
| Novel? | Yes | No, textbook |

**Why the distinction matters.** Banker's is a correctness mechanism: it
prevents deadlock and it is provably sufficient for that. It says nothing about
*efficiency*, and it is entirely blind to geometry — to Banker's, a crane
granted to a robot 200 m away is identical to one granted to a robot standing
beside it. Both are simply "allocated". The admission layer adds the spatial
information Banker's does not model, without touching the correctness
guarantee. Describing this as a modified Banker's would claim a change to the
safety proof that has not been made, and would be wrong.

---

## 2. The Banker's safety algorithm, walked through

### The four matrices

For `n` robots and `m` resource classes:

| Structure | Shape | Meaning |
|---|---|---|
| `Available` | `m` | Instances of each class currently free |
| `Max` | `n x m` | The most robot *i* may ever need of each class |
| `Allocation` | `n x m` | What robot *i* holds right now |
| `Need` | `n x m` | `Max - Allocation`, what it may still ask for |

The invariant `Need == Max - Allocation` is asserted after **every** mutation in
[`alloc/state.py`](../warehouse_allocator/alloc/state.py), along with
`Allocation <= Max`, non-negativity, and
`Available + sum(Allocation) == Total`. These checks are cheap and they are the
tripwire for the subtlest bug in the project — lease revocation during a partial
bundle hold silently corrupting the books.

`Max` is declared **dynamically**: a robot's maximum claim is the bundle of the
task it currently holds, and returns to zero on completion. This is the standard
way to apply Banker's when the process set is not known in advance. Idle robots
have `Max = Need = Allocation = 0`, so they never obstruct a safe sequence.

### The safety check

A state is **safe** if a *safe sequence* exists: an ordering of the robots in
which each can finish in turn, using only what is free plus everything released
by those ahead of it. If such an order exists, deadlock is avoidable however
execution actually interleaves. If none exists the state is **unsafe** — note
that unsafe is not the same as deadlocked. An unsafe state is one from which
deadlock has become *possible*, and Banker's refuses to enter it.

To test a request:

1. **Tentatively grant it.**
   ```
   Available' = Available - Request
   Allocation'[i] = Allocation[i] + Request
   Need'[i]       = Need[i] - Request
   ```
2. **Initialise.** `Work = Available'`; `Finish[r] = False` for all `r`.
3. **Find a robot that can finish**: any `r` with `Finish[r] == False` and
   `Need'[r] <= Work` elementwise. Its entire remaining need can be met from
   what is free, so it is guaranteed to complete.
4. **Let it finish and reclaim**: `Work += Allocation'[r]`; `Finish[r] = True`.
   Go to 3.
5. **Verdict.** All `Finish` true → safe. Otherwise the unfinished robots can
   only be waiting on each other → unsafe, refuse.

The implementation is pure: it works on copies and never mutates the live state.

---

## 3. A worked example

The classic five-process, three-class example. Totals are **crane 10, dock 5,
elevator 7**. (Every number below is printed by the code, not hand-computed.)

### Starting state

| Robot | Max (crane, dock, elev) | Allocation | Need |
|---|---|---|---|
| 0 | 7, 5, 3 | 0, 1, 0 | 7, 4, 3 |
| 1 | 3, 2, 2 | 2, 0, 0 | 1, 2, 2 |
| 2 | 9, 0, 2 | 3, 0, 2 | 6, 0, 0 |
| 3 | 2, 2, 2 | 2, 1, 1 | 0, 1, 1 |
| 4 | 4, 3, 3 | 0, 0, 2 | 4, 3, 1 |

`Available = (3, 3, 2)`.

### This state is SAFE

The algorithm finds the sequence **1 → 3 → 4 → 0 → 2**:

| Step | Robot | Need | Work before | Fits? | Work after (`+= Allocation`) |
|---|---|---|---|---|---|
| 1 | 1 | 1, 2, 2 | 3, 3, 2 | yes | 5, 3, 2 |
| 2 | 3 | 0, 1, 1 | 5, 3, 2 | yes | 7, 4, 3 |
| 3 | 4 | 4, 3, 1 | 7, 4, 3 | yes | 7, 4, 5 |
| 4 | 0 | 7, 4, 3 | 7, 4, 5 | yes | 7, 5, 5 |
| 5 | 2 | 6, 0, 0 | 7, 5, 5 | yes | 10, 5, 7 |

Every robot finishes, so the state is safe. Note robot 0 is not runnable at the
start (it needs 7 cranes, only 3 are free) yet the state is still safe — enough
resources come back from robots 1, 3 and 4 first. This is exactly why Banker's
asks "does *an* order exist" rather than "can everyone run now".

Verified: `test_safety_known_safe`, `test_safety_known_safe_sequence`.

### A safe request

Robot 1 requests `(1, 0, 2)`. Result: **safe, granted.**
`Available` becomes `(2, 3, 0)`, robot 1's `Need` becomes `(0, 2, 0)`, and the
sequence 1 → 3 → 4 → 0 → 2 still exists.

### An UNSAFE request

Robot 4 requests `(3, 3, 0)`. Result: **unsafe, refused.**

Granting would leave `Available = (0, 0, 2)`. Now check every robot's remaining
need against that:

| Robot | Need after grant | `Need <= (0,0,2)`? |
|---|---|---|
| 0 | 7, 4, 3 | no |
| 1 | 1, 2, 2 | no |
| 2 | 6, 0, 0 | no |
| 3 | 0, 1, 1 | no (needs 1 dock, 0 free) |
| 4 | 1, 0, 1 | no (needs 1 crane, 0 free) |

No robot can finish, so no robot ever releases anything, so no robot can ever
finish. The state is unsafe and the request is refused — **even though nothing
is deadlocked at the moment of refusal**. That anticipation is the entire value
of the algorithm.

Verified: `test_safety_known_unsafe`.

---

## 4. Why deadlock is constructible here at all

A safety algorithm with nothing to prevent is dead weight. Two properties of the
workload make genuine deadlock reachable.

**Bundles are acquired incrementally, holding what is already won.** A robot
requests one class at a time and, when blocked, *keeps* everything it has been
granted while it waits. Releasing on block would dissolve every deadlock before
it could form and the whole project would have no subject.

**Acquisition orders deliberately contradict each other.** This is subtler and
easy to get wrong. If every robot acquired classes in one global order (always
crane, then elevator, then dock) then circular wait would be *structurally
impossible* — that is the classic ordered-resource deadlock **prevention**
scheme, and Banker's would be redundant over a FIFO queue.

So each template declares its own order, and the orders conflict:

| Template | Acquisition order | Holds | Waits for |
|---|---|---|---|
| `HEAVY_LIFT` | crane → elevator | crane | elevator |
| `FLOOR_TRANSFER` | elevator → dock | elevator | dock |
| `FULL_CYCLE` | dock → crane → elevator | dock | crane |
| `RECHARGE` | dock | — | — |

Those three compose a ring: the HEAVY_LIFT robot waits on the elevator held by
the FLOOR_TRANSFER robot, which waits on the dock held by the FULL_CYCLE robot,
which waits on the crane held by the HEAVY_LIFT robot. With one instance of each
class, granting the third request closes the cycle and deadlocks the fleet.

`test_circular_wait_refused` constructs exactly this and asserts the allocator
refuses. `test_acquisition_orders_are_inconsistent` asserts no global ordering
could explain the templates, so the property cannot be lost by a careless edit.

---

## 5. The admission policies

Each answers "should this request be considered now?" and may only **defer**.

### `NullPolicy` — admit everything

The classical baseline, reproducing textbook behaviour where position is
invisible to the allocator. Every other policy is measured against it.

*Failure mode:* full early-binding waste. A resource is granted the instant it
is free, to whoever asks, however far away.

### `RadiusPolicy` — admit if euclidean distance <= r

The originally proposed idea, and a cheap proxy for "will arrive soon".

*Failure mode, and it is fundamental:* straight-line distance ignores topology.
A robot 20 m away on the far side of a rack aisle has a **worse** real arrival
time than one 200 m down a clear corridor, and this policy prefers the former.
`test_eta_uses_path_cost_not_straight_line` constructs precisely that geometry
and shows radius admitting the robot it should refuse.

### `ETAPolicy` — admit if A\* path cost <= threshold

Fixes the above by using **actual A\* path cost**, not straight-line distance.
That distinction is the entire reason this policy exists; implementing it with
euclidean distance would collapse it into `RadiusPolicy`.

*Failure mode:* cost. An A\* search per candidate request is orders of magnitude
more expensive than a distance comparison, and it scales with fleet size and map
size together.

### `LeasePolicy` — admit anyone, attach an arrival deadline

The theoretically correct fix. Radius and ETA attack early binding *by proxy*:
they guess in advance who will arrive promptly and pre-emptively exclude
everyone else, which necessarily also excludes robots that would have been fine.
Leasing attacks the problem **directly** — grant to anyone, then reclaim the
resource if the holder has not actually turned up in time. No requester is
excluded in advance, and no resource can sit claimed-but-unreached for longer
than one lease period.

*Failure mode:* revocation churn. Under contention a resource can be granted and
revoked repeatedly, doing no useful work between cycles, and each revocation
sends its robot back to re-request. The measured results show this dominating
when the lease is short.

---

## 6. Starvation and the force-admission rule

Any admission filter starves agents in low-traffic regions: a robot that is
always slightly too far away is deferred forever.

So every deferred request carries a `deferral_count`, and after `max_deferrals`
(default 5) it is **force-admitted**, bypassing the admission policy.

> **The safety check is never bypassed. Not for force-admission, not for
> priority, not for low battery, not for anything.**

Bypassing it would reintroduce deadlock, which is the one failure mode the
allocator exists to prevent. A comment states this at the bypass site in
[`alloc/allocator.py`](../warehouse_allocator/alloc/allocator.py) so that no
future edit removes it by accident, and
`test_force_admit_still_safety_checked` asserts that a force-admitted request
which *would* close a deadlock cycle is still refused.

Force-admissions are counted and reported as a first-class metric. Starvation
that is fixed silently is starvation that gets reported as success.

---

## 7. The request flow, in full

```
1. Select the nearest instance of the class with spare capacity
     (ties broken by resource id, so the choice is reproducible)
   -> none available? defer.

2. deferral_count > max_deferrals?
     yes -> force-admit: skip the policy, emit RequestForceAdmitted
     no  -> AdmissionPolicy.should_admit()
              -> DEFER? increment deferral_count, emit RequestDeferred, stop.

3. SafetyChecker.is_safe()          <- ALWAYS reached, forced or not
     -> unsafe? emit RequestRefusedUnsafe, defer, stop.

4. Grant: update matrices, add holder, emit RequestGranted,
   and register a lease if the policy issued a deadline.
```

Every branch emits a structured event carrying the simulated timestamp, robot
id, resource class, policy verdict, policy reason, safety verdict and deferral
count — so any surprising aggregate can be traced back to the individual
decisions that produced it.

---

## 8. Lease revocation during a partial hold

The trickiest correctness point in the project.

A robot holds a crane and an elevator. It reaches the crane but its elevator
lease expires. Revocation must:

- return **only** the elevator to the pool,
- leave the crane held,
- decrement `Allocation[robot][elevator]` and increment `Available[elevator]`,
- **increase** `Need[robot][elevator]` back by one — the robot still intends to
  finish its task and will ask again, so its need is *restored*, not forgotten,
- leave `Max` untouched,
- step the robot's acquisition cursor back by one and return it to `WAITING`.

Getting the `Need` restoration wrong breaks `Need == Max - Allocation`, and the
assertion fires much later in the run looking like a kinematics bug.
`test_lease_revocation_partial_hold` pins every one of these postconditions.
