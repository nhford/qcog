#pragma once

constexpr int N = 128;

enum GATE { HADAMARD, CNOT, NOT, RZ, TOMBSTONE };

struct Circuit {
  int num_qubits;
  int num_gates;
  GATE gate_type[N];
  float angle[N];
  int qubit_0[N];
  int qubit_1[N];
  // First index selects the operand. -1 means no neighbor.
  int next_idx[2][N];
  int prev_idx[2][N];
};

void optimize(Circuit &circuit);
