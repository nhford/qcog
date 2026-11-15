import argparse

from qiskit import qasm2


def draw_circuit(input_path, output_path):
    circuit = qasm2.load(input_path)
    circuit.draw(output="mpl", filename=output_path, fold=24)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="OpenQASM 2 circuit")
    parser.add_argument("output", help="Output PNG path")
    args = parser.parse_args()
    draw_circuit(args.input, args.output)
