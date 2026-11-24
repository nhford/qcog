#!/usr/bin/env python3
"""Run main's executable on each Hadamard benchmark and save its QASM output."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimizer', type=Path, default=root / 'qcog')
    args = parser.parse_args()
    executable = args.optimizer.resolve()
    if not executable.is_file():
        parser.error(f'Executable not found: {executable}. Build it with make first.')

    results = root / 'tests' / 'results' / 'hadamard'
    results.mkdir(parents=True, exist_ok=True)
    failures = 0
    for circuit in sorted((root / 'benchmarks' / 'hadamard').glob('*.qasm')):
        output = results / circuit.name
        output.unlink(missing_ok=True)
        try:
            run = subprocess.run([str(executable), '-f', str(circuit)],
                                 capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, OSError) as error:
            failures += 1
            print(f'FAIL {circuit.name}: {error}', file=sys.stderr)
            if isinstance(error, subprocess.CalledProcessError) and error.stderr:
                print(error.stderr, end='', file=sys.stderr)
            continue
        output.write_text(run.stdout)
        if run.stderr:
            print(run.stderr, end='', file=sys.stderr)
        print(f'Saved {output.relative_to(root)}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
