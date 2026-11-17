#pragma once
#include "circuit.h"

// Like hw2's renderer: load host data, setup device data, run, retrieve results.
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
};
