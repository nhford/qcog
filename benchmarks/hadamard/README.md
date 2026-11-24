# Hadamard reduction fixtures

Small inputs for the three Hadamard rewrites in `optimizer.cu`, following the
local match/select/apply scheme and small-circuit equivalence checks described
in `project-write-up.pdf` (Section 1, suggested plan items 2 and 6).
Each input has a corresponding reference circuit in `expected/`.

- `hp_positive.qasm`, `hp_negative.qasm`: H-P-H becomes P^-H-P^ for
  P = RZ(+pi/2) or RZ(-pi/2). Both have 3 -> 3 gates and 2 -> 1 Hadamards.
- `hpc_positive.qasm`, `hpc_negative.qasm`: H-P-CX(target)-P^-H becomes
  P^-CX(target)-P. Both have 5 -> 3 gates and 2 -> 0 Hadamards.
- `hcnot_interleaved.qasm`: Hadamards on both wires before and after CX
  disappear and the CX direction reverses. An unrelated X on a third qubit
  checks that matching follows wire links rather than consecutive array entries.
  This has 6 -> 2 gates and 4 -> 0 Hadamards.
- `no_match_angle.qasm`: pi/4 is not a phase angle accepted by these rules.
  All 3 gates remain.
- `no_match_control.qasm`: the H-P pair is on the CX control instead of its
  target, so the Hadamard rule does not apply. Pair reduction commutes the
  inverse RZ gates through the CX control and cancels them, leaving 3 gates.
- `no_match_same_sign.qasm`: the two phase angles are equal rather than
  opposite. All 5 gates remain.

All matched phase gates have zero real (second) angle component. These QASM
fixtures do not directly test rejection of a nonzero second component in the
in-memory representation.

## Run

From the project root, check that every reference is equivalent to its input:

```sh
python3 benchmarks/hadamard/check.py
```

The checker uses only Python's standard library. It simulates every basis
input and compares the complete operators up to one common global phase.

Run the CUDA test suite, including every fixture and its expected output:

```sh
make test-gpu
```

This checks uploads/downloads and compares the optimized circuit's gate types,
operands, angles, and wire links with the reference on each visible GPU.

To additionally check the output's full quantum operator, build and run:

```sh
make
python3 benchmarks/hadamard/check.py --optimizer ./qcog
```

This checks output equivalence, total gate count, and Hadamard count against
the references. It permits equivalent output orderings instead of requiring a
textual match. These fixtures each need at most one rewrite round.

To inspect a single case:

```sh
./qcog -f benchmarks/hadamard/hpc_positive.qasm
```

Fixture-only checks do not exercise CUDA or validate kernel selection behavior.
