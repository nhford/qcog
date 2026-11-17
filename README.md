# QCOG

Build with `make` (CUDA/nvcc required), then run
`./qcog -f input.qasm > output.qasm`.

The lifecycle follows hw2's host-loading/device-setup separation:

1. `read_circuit` parses QASM into a host-owned `Circuit` with contiguous SoA
   sections and builds next/previous gate links on each wire.
2. `CudaOptimizer::setup` allocates device storage, uploads the arrays, and
   constructs a `CircuitView` using device addresses.
3. `CudaOptimizer::optimize` is the insertion point for optimization kernels.
   No rewrite kernels have been implemented yet.
4. `copy_to_host` retrieves the arrays for `write_circuit`. The optimizer owns
   the device allocation and releases it on destruction.

The host parser is adapted from POPQC's Rust QASM parser (see
`THIRD_PARTY_NOTICES`). It flattens multiple named quantum registers into
qubit indices and stores H, X, CX, and RZ directly in SoA form, without
`CircuitLayer`. RZ arithmetic accepts numbers in radians, pi/PI/π, unary signs,
parentheses, and +, -, *, /. Angles are stored as multiples of pi.
Input uses one statement per line, declarations before gates, explicit indexed
operands, and optional // comments. Unsupported gates and nonunitary statements
are rejected; this is a parser for the optimizer's gate set, not full OpenQASM.

Run `make test` to exercise the host parser and round-trip the Toffoli and vqe_4 benchmarks without CUDA.
