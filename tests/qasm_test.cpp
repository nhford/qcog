#include "qasm.h"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
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
  assert(std::abs(c.gates.angle_pi[0][2] + 0.25f) < 1e-6f);
  assert(c.gates.angle_pi[1][2] == 0);
  assert(c.gates.angle_pi[0][3] == 0 && c.gates.angle_pi[1][3] == 0.5f);
  for (int i = 0; i < 2; ++i)
    assert(c.gates.angle_pi[0][i] == 0 && c.gates.angle_pi[1][i] == 0);

  std::istringstream mixed(
      "qreg q[1];\nrz(pi/2 + 0.125) q[0];\n"
      "rz((pi - 0.5)/2) q[0];\nrz(2*(pi + 0.25)) q[0];\n"
      "rz(π/4) q[0];\nrz(pi*pi) q[0];\nrz(1/pi) q[0];\n"
      "rz(0) q[0];\n");
  Circuit angles = read_circuit(mixed);
  const float coefficients[] = {0.5f, 0.5f, 2, 0.25f, 0, 0, 0};
  const float remainders[] = {0.125f, -0.25f, 0.5f, 0,
                              static_cast<float>(3.14159265358979323846 * 3.14159265358979323846),
                              static_cast<float>(1 / 3.14159265358979323846), 0};
  for (int i = 0; i < angles.num_gates; ++i) {
    assert(angles.gates.angle_pi[0][i] == coefficients[i]);
    assert(angles.gates.angle_pi[1][i] == remainders[i]);
  }
  std::ostringstream mixed_output;
  write_circuit(mixed_output, angles);
  std::istringstream mixed_encoded(mixed_output.str());
  Circuit mixed_copy = read_circuit(mixed_encoded);
  assert(mixed_copy.num_gates == angles.num_gates);
  for (int i = 0; i < angles.num_gates; ++i)
    for (int component = 0; component < 2; ++component)
      assert(mixed_copy.gates.angle_pi[component][i] == angles.gates.angle_pi[component][i]);
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
  std::cout << "PASS: parser, angle components, mixed-angle round trip, invalid input\n";
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
      for (int component = 0; component < 2; ++component)
        assert(copy.gates.angle_pi[component][i] == original.gates.angle_pi[component][i]);
      for (int operand = 0; operand < 2; ++operand) {
        assert(copy.gates.next_idx[operand][i] == original.gates.next_idx[operand][i]);
        assert(copy.gates.prev_idx[operand][i] == original.gates.prev_idx[operand][i]);
      }
    }
    std::cout << "PASS: round trip " << argv[arg] << '\n';
  }
  std::cout << "All QASM tests passed (" << argc - 1 << " benchmark files).\n";
}
