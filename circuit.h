#pragma once

#include <cstddef>
#include <memory>

enum GATE { HADAMARD, CNOT, NOT, RZ, TOMBSTONE };

struct CircuitView {
  GATE *gate_type;
  float *angle_pi[2];
  int *qubit_0;
  int *qubit_1;
  int *next_idx[2];
  int *prev_idx[2];
};

struct Circuit {
  int num_qubits = 0;
  int num_gates;
  std::unique_ptr<unsigned char[]> storage;
  CircuitView gates;

  explicit Circuit(int count);
  std::size_t bytes() const;
};

// Construct a view into host or device storage without copying host pointers.
CircuitView circuit_view(unsigned char *storage, int count);
