// QASM parsing adapted from POPQC resources/roqc/circuit/src/seq/qasm_parser.rs.
// See THIRD_PARTY_NOTICES for the upstream license.
#include "qasm.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <istream>
#include <ostream>
#include <unordered_map>

namespace {
constexpr double pi = 3.14159265358979323846;
// Small arithmetic expression evaluator, replacing Rust's meval dependency.
class Angle {
  std::string text;
  std::size_t pos = 0;
  void space() { while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos; }
  bool take(char c) { space(); if (pos < text.size() && text[pos] == c) { ++pos; return true; } return false; }
  double atom() {
    if (take('+')) return atom();
    if (take('-')) return -atom();
    if (take('(')) { double v = sum(); if (!take(')')) throw std::runtime_error("Missing angle parenthesis"); return v; }
    space();
    if (text.compare(pos, 2, "pi") == 0 || text.compare(pos, 2, "PI") == 0) { pos += 2; return pi; }
    if (text.compare(pos, std::string("π").size(), "π") == 0) { pos += std::string("π").size(); return pi; }
    std::size_t used = 0;
    double v = std::stod(text.substr(pos), &used); pos += used; return v;
  }
  double product() {
    double v = atom();
    for (;;) { if (take('*')) v *= atom(); else if (take('/')) { double d = atom(); if (d == 0) throw std::runtime_error("Zero angle divisor"); v /= d; } else return v; }
  }
  double sum() {
    double v = product();
    for (;;) { if (take('+')) v += product(); else if (take('-')) v -= product(); else return v; }
  }
public:
  explicit Angle(std::string value) : text(std::move(value)) {}
  float parse() {
    double value = sum() / pi; space();
    float result = static_cast<float>(value);
    if (pos != text.size() || !std::isfinite(result)) throw std::runtime_error("Invalid rotation angle: " + text);
    return result;
  }
};
}
static float parse_angle_pi(const std::string &angle) { return Angle(angle).parse(); }

static std::string clean_line(std::string line) {
  line.erase(std::min(line.size(), line.find("//")));
  const auto start = line.find_first_not_of(" \t\r");
  return start == std::string::npos
      ? "" : line.substr(start, line.find_last_not_of(" \t\r") - start + 1);
}

// Split the operation before touching operands, so spaces inside angles survive.
static std::string operation(const std::string &line) {
  return line.substr(0, line.find_first_of("( \t\r;"));
}

struct RegisterOperand {
  std::string name;
  int index;
};

static RegisterOperand parse_register_operand(std::string text) {
  text = clean_line(text);
  const auto open = text.find('[');
  const auto close = text.find(']');
  if (open == std::string::npos || close == std::string::npos || close < open ||
      !clean_line(text.substr(close + 1)).empty())
    throw std::runtime_error("Expected register[index]: " + text);
  const auto name = clean_line(text.substr(0, open));
  if (name.empty() || !(std::isalpha(static_cast<unsigned char>(name[0])) || name[0] == '_'))
    throw std::runtime_error("Invalid register name: " + name);
  for (char c : name)
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_'))
      throw std::runtime_error("Invalid register name: " + name);
  const auto digits = clean_line(text.substr(open + 1, close - open - 1));
  if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("Invalid register index: " + text);
  return {name, std::stoi(digits)};
}

static int calculate_qubit_index(
    const std::string &operand,
    const std::unordered_map<std::string, std::pair<int, int>> &registers) {
  const auto qubit = parse_register_operand(operand);
  const auto reg = registers.find(qubit.name);
  if (reg == registers.end())
    throw std::runtime_error("Unknown quantum register: " + qubit.name);
  if (qubit.index >= reg->second.second)
    throw std::runtime_error("Qubit index out of bounds: " + operand);
  return reg->second.first + qubit.index;
}

Circuit read_circuit(std::istream &input) {
  int count = 0;
  std::string line;
  while (std::getline(input, line)) {
    const auto name = operation(clean_line(line));
    if (name == "h" || name == "x" || name == "cx" || name == "rz") {
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
  std::unordered_map<std::string, std::pair<int, int>> registers;
  int i = 0;
  while (std::getline(input, line)) {
    line = clean_line(line);
    if (line.empty() || line == "OPENQASM 2.0;" ||
        line == "include \"qelib1.inc\";")
      continue;

    if (line.back() != ';')
      throw std::runtime_error("Missing QASM semicolon: " + line);
    line.pop_back();
    const auto name = operation(line);
    if (name == "qreg") {
      if (i) throw std::runtime_error("qreg declarations must precede gates");
      const auto reg = parse_register_operand(line.substr(name.size()));
      const int size = reg.index;
      if (size == 0) throw std::runtime_error("Quantum register must be nonempty");
      if (size > std::numeric_limits<int>::max() - circuit.num_qubits)
        throw std::runtime_error("Too many qubits");
      if (!registers.emplace(reg.name, std::make_pair(circuit.num_qubits, size)).second)
        throw std::runtime_error("Duplicate quantum register");
      circuit.num_qubits += size;
      last_gate.resize(circuit.num_qubits, -1);
      last_operand.resize(circuit.num_qubits, -1);
      continue;
    }
    if (name != "h" && name != "x" && name != "cx" && name != "rz")
      throw std::runtime_error("Unsupported QASM operation: " + name);

    auto operands = clean_line(line.substr(name.size()));
    float angle = 0;
    if (name == "rz") {
      if (operands.empty() || operands.front() != '(')
        throw std::runtime_error("RZ requires an angle");
      const auto end = operands.rfind(')');
      if (end == std::string::npos)
        throw std::runtime_error("Missing RZ closing parenthesis");
      angle = parse_angle_pi(operands.substr(1, end - 1));
      operands = clean_line(operands.substr(end + 1));
    }
    const auto comma = operands.find(',');
    if ((name == "cx") != (comma != std::string::npos))
      throw std::runtime_error("Wrong number of operands: " + line);
    const int q0 = calculate_qubit_index(operands.substr(0, comma), registers);
    const int q1 = comma == std::string::npos ? -1 :
        calculate_qubit_index(operands.substr(comma + 1), registers);
    if (q0 == q1) throw std::runtime_error("CNOT operands must differ");
    if (i == circuit.num_gates)
      throw std::runtime_error("Gate count changed while reading circuit");

    circuit.gates.gate_type[i] = name == "h" ? HADAMARD : name == "x" ? NOT :
                                 name == "cx" ? CNOT : RZ;
    circuit.gates.angle_pi[i] = angle;
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

void write_circuit(std::ostream &output, const Circuit &circuit) {
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

