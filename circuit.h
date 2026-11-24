#pragma once

#include <cstddef>
#include <memory>

#define NO_CARRY 0
#define X_CARRY 1
#define Z_CARRY 2
#define BOTH_CARRY 3

enum GATE { HADAMARD, CNOT, NOT, RZ, TOMBSTONE };

struct CircuitView {
  GATE *gate_type;
  // angle = angle_pi[0][gate] * pi + angle_pi[1][gate] (radians)
  float *angle_pi[2];
  int *qubit_0;
  int *qubit_1;
  int *next_idx[2];
  int *prev_idx[2]; /** TODO: need these be initialized to -1? */
  // First gate on each wire, or -1 for an empty wire.
  int *qstart_indices;

  // 2 bits per qubit to mark carry of x, z
  char *xz_carry;
};

struct Circuit {
  int num_qubits = 0;
  int num_gates;
  std::unique_ptr<unsigned char[]> storage;
  CircuitView gates;

  Circuit(int gate_count, int qubit_count);
  std::size_t bytes() const;
};

// Construct a view into host or device storage without copying host pointers.
CircuitView circuit_view(unsigned char *storage, int count, int qubit_count);
