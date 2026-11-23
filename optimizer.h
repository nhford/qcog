#pragma once
#include "circuit.h"

enum Pattern {
  NONE = 0,
  HP_TOGGLE,
  HPC_TOGGLE,
  HCNOT_CANCEL,
  HH_CANCEL,
  XX_CANCEL,
  CNOT_CANCEL,
  RZ_MERGE
};

// Like hw2's renderer: load host data, setup device data, run, retrieve
// results.
class CudaOptimizer {
public:
  CudaOptimizer() = default;
  CudaOptimizer(const int num_threads);
  ~CudaOptimizer();
  CudaOptimizer(const CudaOptimizer &) = delete;
  CudaOptimizer &operator=(const CudaOptimizer &) = delete;
  void setup(const Circuit &host);
  void optimize();
  void copy_to_host(Circuit &host) const;

private:
  unsigned char *device_storage_ = nullptr;
  // The view itself lives on the GPU so kernels can receive it by reference.
  CircuitView *device_gates_ = nullptr;
  int num_gates_ = 0;
  int num_qubits_ = 0;
  bool ready_ = false;

  int _num_threads = 256;

  /* which gate thread's pattern writes to this gate */
  int *device_owning_pattern_ = nullptr;

  /* which gate threads are disjoint and selected */
  int *device_is_selected_ = nullptr;

  /* for commutating updates (Rz, CNOT): must store where apply */
  int *next_idx = nullptr;
};
