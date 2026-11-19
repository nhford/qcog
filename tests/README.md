# Running the Hadamard benchmarks

From the project root:

```sh
make run-hadamard
```

This builds `qcog`, then calls its `main` with `-f` for each of the eight inputs
in `benchmarks/hadamard/`. Outputs go to `tests/results/hadamard/` with the same
filenames. Generated results are ignored by git. Each run replaces previous
results; failed cases leave no output file and the runner exits nonzero after
trying every input.

To use an already-built executable:

```sh
python3 tests/run_hadamard.py
python3 tests/run_hadamard.py --optimizer /path/to/qcog
```

This runner saves outputs and checks process exit status. To check circuit
equivalence and expected gate counts as well, run the existing checker:

```sh
python3 benchmarks/hadamard/check.py --optimizer ./qcog
```

For one circuit, stdout can be displayed, saved, or both:

```sh
# Display in the terminal.
./qcog -f benchmarks/hadamard/hpc_positive.qasm

# Save to a file.
./qcog -f benchmarks/hadamard/hpc_positive.qasm > output.qasm

# Display and save the same circuit.
./qcog -f benchmarks/hadamard/hpc_positive.qasm | tee output.qasm
```
