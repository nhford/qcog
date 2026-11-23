# QCOG

Build with `make` (CUDA/nvcc required), then run
`./qcog -f input.qasm > output.qasm`.

The CUDA build uses `g++-11` as its host compiler for compatibility with CUDA
11.7. Override it with `make NVCC_HOST_COMPILER=/path/to/g++` when using a
different supported compiler.

The lifecycle follows hw2's host-loading/device-setup separation:

1. `read_circuit` parses QASM into a host-owned `Circuit` with contiguous SoA
   sections and builds next/previous gate links on each wire.
2. `CudaOptimizer::setup` allocates device storage, uploads the arrays, and
   constructs a `CircuitView` using device addresses.
3. `CudaOptimizer::optimize` runs one matching and rewrite pass for the HP,
   HPC, and HCNOT patterns.
4. `copy_to_host` retrieves the arrays for `write_circuit`. The optimizer owns
   the device allocation and releases it on destruction.

The host parser is adapted from POPQC's Rust QASM parser (see
`THIRD_PARTY_NOTICES`). It flattens multiple named quantum registers into
qubit indices and stores H, X, CX, and RZ directly in SoA form, without
`CircuitLayer`. RZ arithmetic accepts numbers in radians, pi/PI/π, unary signs,
parentheses, and +, -, *, /. Angles are stored as a pi coefficient plus a
floating-point remainder in radians.
Input uses one statement per line, declarations before gates, explicit indexed
operands, and optional // comments. Unsupported gates and nonunitary statements
are rejected; this is a parser for the optimizer's gate set, not full OpenQASM.

Run `make test` to exercise the host parser and round-trip the Toffoli and vqe_4 benchmarks without CUDA.
The tests print PASS messages and a final summary; assertion failures terminate
the parser test with a nonzero exit status.

Run `make test-gpu` to enumerate CUDA devices and test uploads/downloads, all
three rewrite patterns, and preservation of mixed and numeric angles on every
visible GPU. Devices are tested sequentially using `cudaSetDevice`; when two or
more are visible, results are also compared across devices. A single GPU prints
an explicit skip for the cross-device comparison. `make test-all` runs both
suites. Use `nvidia-smi -L` to list GPUs on the current host.

The GPU suite also reads every `benchmarks/hadamard/*.qasm` input and compares
its optimized output with the corresponding circuit in `benchmarks/hadamard/expected/`.
To run the built-in cases plus a single fixture, use
`./tests/gpu_test benchmarks/hadamard/hpc_positive.qasm`.

Run `make test-commutation` for direct GPU tests of `single_commute_advance`
(the current commutation helper), `check_terminate`, and `match_pair`, plus
end-to-end cancellation checks. These 71 checks are also included in
`make test-gpu` and `make test-all`. They cover Figure 5 on page 12 of
[Nam et al.](https://arxiv.org/pdf/1710.07345), including blocked commutations
and single-qubit cancellation/rotation merging. An independent small
statevector simulator checks the full operators up to global phase.
The new suite currently exposes unfinished cancellation behavior and commutation
bugs, so a nonzero exit is expected until those are addressed. See
[test details and initial results](tests/COMMUTATION.md).
