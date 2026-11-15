/**
 * Parallel Circuit Optimization
 * Noah Ford (nford), Changxiang Gao (changxig)
 */

#include "circuit.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

static float parse_angle_pi(const std::string &angle) {
  static const std::regex multiple_of_pi(
      R"(\s*([+-]?)\s*(?:([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)\s*\*\s*)?pi(?:\s*/\s*([1-9][0-9]*))?\s*)");
  std::smatch match;
  if (!std::regex_match(angle, match, multiple_of_pi))
    throw std::runtime_error("Expected an angle in terms of pi: " + angle);

  const double numerator = match[2].matched ? std::stod(match[2]) : 1.0;
  const double denominator = match[3].matched ? std::stod(match[3]) : 1.0;
  const float coefficient = numerator / denominator;
  if (!std::isfinite(coefficient))
    throw std::runtime_error("Rotation angle is out of range: " + angle);
  return match[1] == "-" ? -coefficient : coefficient;
}

std::size_t Circuit::bytes() const {
  return static_cast<std::size_t>(num_gates) *
         (sizeof(GATE) + sizeof(float) + 6 * sizeof(int));
}

Circuit::Circuit(int count)
    : num_gates(count), storage(new unsigned char[bytes()]), gates{} {
  // Each section starts on a four-byte boundary.
  static_assert(sizeof(GATE) == 4 && sizeof(float) == 4 && sizeof(int) == 4);
  auto *cursor = storage.get();
  gates.gate_type = new (cursor) GATE[count];
  cursor += sizeof(GATE) * static_cast<std::size_t>(count);
  gates.angle_pi = new (cursor) float[count];
  cursor += sizeof(float) * static_cast<std::size_t>(count);

  int **sections[] = {&gates.qubit_0, &gates.qubit_1,
                     &gates.next_idx[0], &gates.next_idx[1],
                     &gates.prev_idx[0], &gates.prev_idx[1]};
  for (auto section : sections) {
    *section = new (cursor) int[count];
    cursor += sizeof(int) * static_cast<std::size_t>(count);
  }
}

static std::string clean_line(std::string line) {
  line.erase(std::min(line.size(), line.find("//")));
  const auto start = line.find_first_not_of(" \t\r");
  return start == std::string::npos
      ? "" : line.substr(start, line.find_last_not_of(" \t\r") - start + 1);
}

static Circuit read_circuit(std::istream &input) {
  static const std::regex qreg(R"(qreg\s+q\[([1-9][0-9]*)\];)");
  static const std::regex gate(
      R"((h|x|cx|rz)(?:\(([^;]+)\))?\s+q\[([0-9]+)\](?:\s*,\s*q\[([0-9]+)\])?;)");

  int count = 0;
  std::string line;
  while (std::getline(input, line)) {
    if (std::regex_match(clean_line(line), gate)) {
      if (count == std::numeric_limits<int>::max())
        throw std::runtime_error("Circuit exceeds the gate index range");
      ++count;
    }
  }
  if (input.bad())
    throw std::runtime_error("Unable to read circuit");
  input.clear();
  input.seekg(0);
  if (!input)
    throw std::runtime_error("Unable to rewind circuit");

  Circuit circuit(count);
  std::vector<int> last_gate, last_operand;
  int i = 0;
  while (std::getline(input, line)) {
    line = clean_line(line);
    if (line.empty() || line == "OPENQASM 2.0;" ||
        line == "include \"qelib1.inc\";")
      continue;

    std::smatch match;
    if (circuit.num_qubits == 0 && std::regex_match(line, match, qreg)) {
      circuit.num_qubits = std::stoi(match[1]);
      last_gate.assign(circuit.num_qubits, -1);
      last_operand.assign(circuit.num_qubits, -1);
      continue;
    }

    if (!std::regex_match(line, match, gate))
      throw std::runtime_error("Unsupported QASM statement: " + line);
    const std::string name = match[1];
    const int q0 = std::stoi(match[3]);
    const int q1 = match[4].matched ? std::stoi(match[4]) : -1;
    if ((name == "cx") != (q1 >= 0) || (name == "rz") != match[2].matched ||
        q0 >= circuit.num_qubits || q1 >= circuit.num_qubits || q0 == q1)
      throw std::runtime_error("Invalid gate: " + line);
    if (i == circuit.num_gates)
      throw std::runtime_error("Gate count changed while reading circuit");

    circuit.gates.gate_type[i] = name == "h" ? HADAMARD : name == "x" ? NOT :
                                 name == "cx" ? CNOT : RZ;
    circuit.gates.angle_pi[i] = name == "rz" ? parse_angle_pi(match[2]) : 0;
    circuit.gates.qubit_0[i] = q0;
    circuit.gates.qubit_1[i] = q1;
    const int qubits[] = {q0, q1};
    for (int operand = 0; operand < 2; ++operand) {
      const int q = qubits[operand];
      circuit.gates.next_idx[operand][i] = -1;
      circuit.gates.prev_idx[operand][i] = -1;
      if (q < 0)
        continue;
      const int previous = last_gate[q];
      circuit.gates.prev_idx[operand][i] = previous;
      if (previous >= 0)
        circuit.gates.next_idx[last_operand[q]][previous] = i;
      last_gate[q] = i;
      last_operand[q] = operand;
    }
    ++i;
  }
  if (circuit.num_qubits == 0)
    throw std::runtime_error("Missing qreg declaration");
  if (input.bad() || i != circuit.num_gates)
    throw std::runtime_error("Unable to read all gates");
  return circuit;
}

static void write_circuit(std::ostream &output, const Circuit &circuit) {
  output << "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q["
         << circuit.num_qubits << "];\n";
  output.precision(9);
  for (int i = 0; i < circuit.num_gates; ++i) {
    switch (circuit.gates.gate_type[i]) {
    case HADAMARD: output << "h"; break;
    case NOT: output << "x"; break;
    case CNOT: output << "cx"; break;
    case RZ: output << "rz(" << circuit.gates.angle_pi[i] << "*pi)"; break;
    case TOMBSTONE: continue;
    }
    output << " q[" << circuit.gates.qubit_0[i] << ']';
    if (circuit.gates.gate_type[i] == CNOT)
      output << ", q[" << circuit.gates.qubit_1[i] << ']';
    output << ";\n";
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3 || std::string(argv[1]) != "-f") {
    std::cerr << "Usage: " << argv[0] << " -f input.qasm > output.qasm\n";
    return 1;
  }

  try {
    std::ifstream input(argv[2]);
    if (!input)
      throw std::runtime_error(std::string("Unable to open ") + argv[2]);
    Circuit circuit = read_circuit(input);
    optimize(circuit);
    write_circuit(std::cout, circuit);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
