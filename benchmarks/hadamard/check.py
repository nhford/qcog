#!/usr/bin/env python3
"""Check fixtures, or an optimizer's output, using a tiny statevector simulator."""

import argparse
import ast
import cmath
import math
from pathlib import Path
import re
import subprocess


def angle(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id == 'pi':
        return math.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return (-1 if isinstance(node.op, ast.USub) else 1) * angle(node.operand)
    if isinstance(node, ast.BinOp):
        a, b = angle(node.left), angle(node.right)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div): return a / b
    raise ValueError('Unsupported angle expression')


def parse(text):
    n, gates = None, []
    for line in text.splitlines():
        line = line.split('//')[0].strip()
        if not line or line.startswith(('OPENQASM ', 'include ')):
            continue
        register = re.fullmatch(r'qreg q\[(\d+)\];', line)
        if register:
            n = int(register[1])
            continue
        gate = re.fullmatch(r'(h|x|cx|rz)(?:\((.+)\))?\s+(.+);', line)
        if not gate:
            raise ValueError(f'Unsupported statement: {line}')
        kind, expr, operands = gate.groups()
        qs = tuple(int(q) for q in re.findall(r'q\[(\d+)\]', operands))
        assert n is not None and len(qs) == (2 if kind == 'cx' else 1)
        assert all(0 <= q < n for q in qs)
        gates.append((kind, qs, angle(ast.parse(expr, mode='eval').body) if expr else 0))
    assert n is not None
    return n, gates


def simulate(n, gates, basis):
    state = [0j] * (1 << n)
    state[basis] = 1
    for kind, qs, theta in gates:
        bit = 1 << qs[0]
        if kind == 'rz':
            # Conventional RZ; equivalence is checked up to global phase.
            state = [v * cmath.exp((0.5j if i & bit else -0.5j) * theta)
                     for i, v in enumerate(state)]
            continue
        for i in range(1 << n):
            if kind == 'cx':
                target = 1 << qs[1]
                if i & bit and not i & target:
                    j = i | target
                    state[i], state[j] = state[j], state[i]
            elif not i & bit:
                j = i | bit
                a, b = state[i], state[j]
                state[i], state[j] = ((a + b) / math.sqrt(2), (a - b) / math.sqrt(2)) if kind == 'h' else (b, a)
    return state


def equivalent(n, before, after):
    phase = None
    for basis in range(1 << n):
        a = simulate(n, before, basis)
        b = simulate(n, after, basis)
        for x, y in zip(a, b):
            if phase is None and abs(x) > 1e-8 and abs(y) > 1e-8:
                phase = x / y
                assert abs(abs(phase) - 1) < 1e-7
            assert abs(x - (phase if phase is not None else 1) * y) < 1e-7, 'Circuits differ'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimizer', type=Path, help='Executable accepting -f INPUT')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob('*.qasm')):
        n, initial = parse(path.read_text())
        expected_n, expected = parse((root / 'expected' / path.name).read_text())
        assert n == expected_n
        equivalent(n, initial, expected)
        actual = expected
        if args.optimizer:
            output = subprocess.run([str(args.optimizer.resolve()), '-f', str(path)],
                                    check=True, capture_output=True, text=True)
            actual_n, actual = parse(output.stdout)
            assert actual_n == n, path.name
            equivalent(n, initial, actual)
            assert len(actual) == len(expected), f'{path.name}: wrong gate count'
            assert sum(g[0] == 'h' for g in actual) == sum(g[0] == 'h' for g in expected), f'{path.name}: wrong H count'
        print(f'PASS {path.stem}: gates {len(initial)} -> {len(actual)}, '
              f'H {sum(g[0] == "h" for g in initial)} -> {sum(g[0] == "h" for g in actual)}')
    if not args.optimizer:
        print('Fixture checks only; optimizer execution was not tested.')


if __name__ == '__main__':
    main()
