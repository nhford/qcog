/**
 * Parallel Circuit Optimization
 * Noah Ford (nford), Changxiang Gao (changxig)
 */

#include "circuit.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

static float parse_angle(const std::string &angle) {
  static const std::regex pi_fraction(R"(([+-]?)pi(?:/([1-9][0-9]*))?)");
  std::smatch match;
  if (std::regex_match(angle, match, pi_fraction)) {
    const float sign = match[1] == "-" ? -1.0f : 1.0f;
    return sign * std::acos(-1.0f) /
           (match[2].matched ? std::stof(match[2]) : 1.0f);
  }
  size_t end;
  float value = std::stof(angle, &end);
  if (end != angle.size() || !std::isfinite(value))
    throw std::runtime_error("Unsupported rotation angle: " + angle);
  return value;
}

static Circuit read_circuit(std::istream &input) {
  static const std::regex qreg(R"(qreg\s+q\[([1-9][0-9]*)\];)");
  static const std::regex gate(
      R"((h|x|cx|rz)(?:\(([^)]+)\))?\s+q\[([0-9]+)\](?:\s*,\s*q\[([0-9]+)\])?;)");

  Circuit circuit{};
  std::vector<int> last_gate, last_operand;
  std::string line;
  while (std::getline(input, line)) {
    line = line.substr(0, line.find("//"));
    auto start = line.find_first_not_of(" \t\r");
    if (start == std::string::npos)
      continue;
    line = line.substr(start, line.find_last_not_of(" \t\r") - start + 1);
    if (line == "OPENQASM 2.0;" || line == "include \"qelib1.inc\";")
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
    if (circuit.num_gates == N)
      throw std::runtime_error("Circuit exceeds the 128-gate capacity");

    const int i = circuit.num_gates++;
    circuit.gate_type[i] = name == "h" ? HADAMARD : name == "x" ? NOT :
                           name == "cx" ? CNOT : RZ;
    circuit.angle[i] = name == "rz" ? parse_angle(match[2]) : 0;
    circuit.qubit_0[i] = q0;
    circuit.qubit_1[i] = q1;
    const int qubits[] = {q0, q1};
    for (int operand = 0; operand < 2; ++operand) {
      const int q = qubits[operand];
      circuit.next_idx[operand][i] = -1;
      circuit.prev_idx[operand][i] = -1;
      if (q < 0)
        continue;
      const int previous = last_gate[q];
      circuit.prev_idx[operand][i] = previous;
      if (previous >= 0)
        circuit.next_idx[last_operand[q]][previous] = i;
      last_gate[q] = i;
      last_operand[q] = operand;
    }
  }
  if (circuit.num_qubits == 0)
    throw std::runtime_error("Missing qreg declaration");
  return circuit;
}

static void write_circuit(std::ostream &output, const Circuit &circuit) {
  output << "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q["
         << circuit.num_qubits << "];\n";
  output.precision(9);
  for (int i = 0; i < circuit.num_gates; ++i) {
    switch (circuit.gate_type[i]) {
    case HADAMARD: output << "h"; break;
    case NOT: output << "x"; break;
    case CNOT: output << "cx"; break;
    case RZ: output << "rz(" << circuit.angle[i] << ")"; break;
    case TOMBSTONE: continue;
    }
    output << " q[" << circuit.qubit_0[i] << ']';
    if (circuit.gate_type[i] == CNOT)
      output << ", q[" << circuit.qubit_1[i] << ']';
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
