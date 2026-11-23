# Commutation and cancellation tests

Run from the project root:

```sh
make test-commutation
```

`make test-gpu` and `make test-all` include this suite as well. Each check prints
PASS or FAIL; failures are collected rather than stopping at the first mismatch.
The executable exits with status 1 if any check fails. No visible CUDA devices
produces an explicit SKIP.

## Reference and coverage

The reference is Figure 5 on printed page 12 and routines 2 and 3 on page 13 of
[Automated optimization of large quantum circuits with continuous parameters](https://arxiv.org/pdf/1710.07345).
The six positive commutation patterns are:

| Moving gate | Subcircuit it crosses |
| --- | --- |
| RZ on wire t | H(t), CX(c,t), H(t) |
| RZ on wire t | CX(c,t), RZ(t), CX(c,t), with identical CX operands |
| RZ on wire c | CX(c,t) |
| CX(a,t) | CX(b,t), sharing the target |
| CX(c,a) | CX(c,b), sharing the control |
| CX(a,b) | H(b), CX(b,c), H(b) |

There is no `can_commute` function in the implementation at the time of writing;
the suite exercises `single_commute_advance` instead.

`tests/commutation_test.cu` includes `optimizer.cu` in its translation unit so a
test kernel can call its static device helpers without changing their visibility
or adding a public testing API. The Makefile does not link optimizer.cu a second
time for this executable.

The 71 checks run independently on each visible GPU:

- 3 buffer checks: all six successful blocks record their crossed gates, and
  insufficient capacity rejects a block without changing the buffer or count.

- 24 commutation checks: all six patterns, nonzero starting indices, incomplete
  blocks, mismatched operands, reversed CXs, blocking gates on either CX wire,
  and circuit boundaries. The expected value is the last crossed gate's actual
  array index, or the original index if blocked.
- 14 termination checks: adjacent RZs, identical CXs, unrelated interleaved gates,
  blockers, reversed CXs, and end-of-wire. The current two-argument
  `check_terminate` API checks immediate RZ/CX partners only; HH/XX are tested
  through `match_pair`.
- 21 pair checks: adjacent HH/XX, ownership claims on both endpoints, rejection
  of blocked pairs, RZ inverse/merge partners, and CX partners across the
  permitted subcircuits. `match_pair` receives the actual next gate on the
  starting gate's first wire. RZ/CX successful matches must return the matching
  endpoint in `next_index`; HH/XX matches must select their cancellation pattern.
- 9 optimizer checks: actual HH/XX elimination, RZ inverse elimination and
  merging (including both angle components), CX elimination, and preservation
  of blocked pairs. These check operator equivalence and surviving gate count.

An independent CPU simulator evaluates all eight basis inputs for these
three-qubit circuits and uses one common global phase for the complete operator.
It verifies all positive commutation references, all positive pair endpoints,
all rewrite references, every reported forward commutation, and all optimized
outputs. The oracle does not call the commutation helpers. Negative cases test
the permitted rules; those rules are sufficient rather than an exhaustive test
of every mathematically possible commutation.

The helper probes use one CUDA thread to isolate traversal and matching logic.
They do not test concurrent ownership conflicts, repeated optimization passes,
or large-circuit performance. Ownership claims for RZ/CX are not required by
these direct tests because the current matcher exposes those partners through
`next_index`; the optimizer checks separately require actual elimination/merging.

## Initial results and diagnosis

The first compiled snapshot on the RTX 2080 produced:

| Checks | Passed | Failed |
| --- | ---: | ---: |
| Commutation | 15 | 9 |
| Termination | 13 | 1 |
| Pair matching | 12 | 9 |
| Optimizer output | 2 | 7 |
| Total | 42 | 26 |

CUDA `compute-sanitizer --tool memcheck --error-exitcode 99
./tests/commutation_test` reproduced these results with **0 memory errors**.
The nonzero application exit is from logical test failures.

The failures point to these implementation issues:

- `single_commute_advance` returns literal `1` for the RZ/CX-control rule instead
  of the next gate's index. A fixture starting at index 1 exposes this.
- The RZ parity-block and H-CX-H target-role rejection predicates use `==`
  where the required target predecessor should be accepted. The parity-block
  code also needs to verify matching CX operands.
- The CX/H-CX-H branch rejects a closing H and accepts a closing non-H.
- Shared-control/target CX tests do not exclude a cross-role dependency or a
  blocker on the other wire. Some accepted advances change the circuit operator.
- `check_terminate` accepts consecutive CXs with reversed control and target:
  adjacency on both wires does not establish equal operand roles.
- `match_pair` requires the immediate successor's type to match before trying
  to commute. This prevents valid searches through H/CX blocks. Its traversal
  also changes the identity of the gate being commuted by calling the helper on
  the crossed gate, then checks termination at the original index. For an
  adjacent identical CX pair it advances past the partner and returns `-1`.
- The main `match` kernel does not call `match_pair`, and `apply` has no cases
  for HH_CANCEL, XX_CANCEL, RZ_MERGE, or CNOT_MERGE. Thus matching an HH/XX pair
  in a direct helper test does not yet eliminate it in the optimizer.

The implementation was being edited during verification. A subsequent
`make test-all` passed the CPU QASM suite but stopped at a new compilation error:
`optimizer.cu` references an undeclared `gate_indices` inside the RZ/CX branch
of `match_pair`. The 42/26 result describes the earlier successfully compiled
snapshot; rebuild after completing that edit to get results for the latest code.
The test changes do not modify `optimizer.cu`.

## Gate-index buffer update

`match_pair` now owns a 64-entry local `int` array initialized with the starting
gate and a populated count of 1. It passes the array, count by reference, and
capacity to `single_commute_advance`. Every successful branch appends the whole
crossed block after checking capacity; a rejected block leaves the array and
count unchanged. The matcher appends the partner before claiming the populated
indices. Capacity exhaustion conservatively stops the search; this bounds the
number of gates considered for a match.

The helper API changes compile, and the current suite reports **52 passed,
19 failed** on the RTX 2080. All three buffer checks pass. The earlier
compilation errors are resolved, along with the RZ target-role predicates,
parity-block operand checks, and the closing-H predicate. Remaining failures
cover CNOT commutation/termination validation, pair traversal, and the missing
end-to-end cancellation rewrites. Buffer wiring does not complete those steps.
