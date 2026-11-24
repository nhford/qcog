#include "circuit.h"
#include <algorithm>
#include <new>
#include <stdexcept>

std::size_t Circuit::bytes() const {
  return static_cast<std::size_t>(num_gates) *
         (sizeof(GATE) + 2 * sizeof(float) + 6 * sizeof(int)) +
         static_cast<std::size_t>(num_qubits) * sizeof(int);
}

Circuit::Circuit(int gate_count, int qubit_count)
    : num_qubits(qubit_count), num_gates(gate_count),
      storage(gate_count >= 0 && qubit_count >= 0
                  ? new unsigned char[bytes()]
                  : throw std::invalid_argument("Negative circuit dimensions")), gates{} {
  gates = circuit_view(storage.get(), gate_count, qubit_count);
  if (num_qubits)
    std::fill_n(gates.qstart_indices, num_qubits, -1);
}

CircuitView circuit_view(unsigned char *storage, int count, int qubit_count) {
  if (count == 0 && qubit_count == 0) return {};
  // Each section starts on a four-byte boundary.
  static_assert(sizeof(GATE) == 4 && sizeof(float) == 4 && sizeof(int) == 4);
  auto *cursor = storage;
  CircuitView gates{};
  gates.gate_type = reinterpret_cast<GATE *>(cursor);
  cursor += sizeof(GATE) * static_cast<std::size_t>(count);
  gates.angle_pi[0] = reinterpret_cast<float *>(cursor);
  cursor += sizeof(float) * static_cast<std::size_t>(count);
  gates.angle_pi[1] = reinterpret_cast<float *>(cursor);
  cursor += sizeof(float) * static_cast<std::size_t>(count);

  int **sections[] = {&gates.qubit_0, &gates.qubit_1,
                     &gates.next_idx[0], &gates.next_idx[1],
                     &gates.prev_idx[0], &gates.prev_idx[1]};
  for (auto section : sections) {
    *section = reinterpret_cast<int *>(cursor);
    cursor += sizeof(int) * static_cast<std::size_t>(count);
  }
  gates.qstart_indices = reinterpret_cast<int *>(cursor);
  return gates;
}
