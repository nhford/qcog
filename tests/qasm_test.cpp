#include "qasm.h"
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

int main(int argc, char **argv) {
  std::istringstream input(
      "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg a[2];\nqreg b[1];\n"
      "h a[1]; // comment\ncx b[0], a[1];\nrz( - (PI + pi) / 8 ) b[0];\n"
      "rz(0.5) a[0];\n");
  Circuit c = read_circuit(input);
  assert(c.num_qubits == 3 && c.num_gates == 4);
  assert(c.gates.qubit_0[1] == 2 && c.gates.qubit_1[1] == 1);
  assert(c.gates.next_idx[0][0] == 1);
  assert(c.gates.prev_idx[1][1] == 0);
  assert(c.gates.next_idx[0][1] == 2);
  assert(c.gates.prev_idx[0][2] == 1);
  assert(c.gates.next_idx[1][0] == -1);
  assert(std::abs(c.gates.angle_pi[2] + 0.25f) < 1e-6f);
  assert(std::abs(c.gates.angle_pi[3] - 0.5f / 3.14159265f) < 1e-6f);
  for (const char *bad : {"qreg a[1];\nh missing[0];\n",
                          "qreg a[1];\nh a[1];\n",
                          "qreg a[1];\nrz(pi/0) a[0];\n",
                          "qreg a[1];\ny a[0];\n",
                          "qreg a[1];\nh a[0] junk;\n",
                          "qreg a[1];\nh a[0], a[0];\n",
                          "qreg a[1];\nh(pi) a[0];\n",
                          "qreg a[2];\ncx a[0], a[1], a[0];\n",
                          "qreg a[1];\nh a[0]\n"}) {
    bool rejected = false;
    try { std::istringstream stream(bad); auto unused = read_circuit(stream); }
    catch (const std::exception &) { rejected = true; }
    assert(rejected);
  }
  for (int arg = 1; arg < argc; ++arg) {
    std::ifstream file(argv[arg]);
    Circuit original = read_circuit(file);
    std::ostringstream output;
    write_circuit(output, original);
    std::istringstream encoded(output.str());
    Circuit copy = read_circuit(encoded);
    assert(copy.num_gates == original.num_gates);
    assert(copy.num_qubits == original.num_qubits);
    for (int i = 0; i < copy.num_gates; ++i) {
      assert(copy.gates.gate_type[i] == original.gates.gate_type[i]);
      assert(copy.gates.angle_pi[i] == original.gates.angle_pi[i]);
      for (int operand = 0; operand < 2; ++operand) {
        assert(copy.gates.next_idx[operand][i] == original.gates.next_idx[operand][i]);
        assert(copy.gates.prev_idx[operand][i] == original.gates.prev_idx[operand][i]);
      }
    }
  }
}
