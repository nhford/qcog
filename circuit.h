#pragma once

#include <cstddef>
#include <memory>

enum GATE { HADAMARD, CNOT, NOT, RZ, TOMBSTONE };

struct CircuitView {
  GATE *gate_type;
  float *angle_pi;
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

void optimize(Circuit &circuit);
