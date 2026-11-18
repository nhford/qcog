#pragma once
#include "circuit.h"

enum Pattern {
  NONE = 0,
  HP_TOGGLE,
  HCNOT_CANCELLATION,
  HPC_TOGGLE,
};

// Like hw2's renderer: load host data, setup device data, run, retrieve
// results.
class CudaOptimizer {
public:
  CudaOptimizer() = default;
  ~CudaOptimizer();
  CudaOptimizer(const CudaOptimizer &) = delete;
  CudaOptimizer &operator=(const CudaOptimizer &) = delete;
  void setup(const Circuit &host);
  void optimize();
  void copy_to_host(Circuit &host) const;

private:
  unsigned char *device_storage_ = nullptr;
  CircuitView device_gates_{};
  int num_gates_ = 0;
  int num_qubits_ = 0;
  bool ready_ = false;

  /* which gate thread's pattern writes to this gate */
  int *device_owning_pattern_ = nullptr;

  /* which gate threads are disjoint and selected */
  int *device_is_selected_ = nullptr;
};
