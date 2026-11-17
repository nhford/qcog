#include "circuit.h"
#include "optimizer.h"
#include <cuda_runtime.h>
#include <stdexcept>

static void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

CudaOptimizer::~CudaOptimizer() {
  if (device_storage_)
    cudaFree(device_storage_);
}

void CudaOptimizer::setup(const Circuit &host) {
  if (ready_)
    throw std::logic_error("Optimizer already set up");
  if (host.num_gates) {
    check(cudaMalloc(&device_storage_, host.bytes()));
    check(cudaMemcpy(device_storage_, host.storage.get(), host.bytes(),
                     cudaMemcpyHostToDevice));
    check(cudaMalloc(&device_owning_pattern_, host.num_gates * sizeof(int)));
  }
  num_gates_ = host.num_gates;
  num_qubits_ = host.num_qubits;
  device_gates_ = circuit_view(device_storage_, num_gates_);
  ready_ = true;
}

/** @brief Find the start of Hadamard patterns to optimize */
__global__ void CudaOptimizer::kernelMatchPattern() {
  int tid = threadIdx.x;
  /* all patterns search at least three gates */
  if (tid >= num_gates_ - 2) {
    return;
  }

  GATE initial_gate = device_gates_.gate_type[tid];
  if (initial_gate != GATE::HADAMARD) {
    return;
  }

  GATE next_gate = device_gates_.gate_type[tid + 1];
  GATE next_next_gate = device_gates_.gate_type[tid + 2];

  if (next_gate == GATE::RZ) {
    float *next_angle = device_gates_.angle_pi[tid + 1];
    if (next_angle[0] != 0.5)
    /* pattern 1: H - P - H => P^ - H - P^ */

    /* pattern 2: H - P - CNOT target - P^ - H => P^ - CNOT - P */
  } else if (next_gate == GATE::CNOT) {
    /* pattern 3: H - CNOT control - H => CNOT target
                    H - CNOT target  - H => CNOT control
      */
  }
}

void CudaOptimizer::optimize() {
  if (!ready_)
    throw std::logic_error("Call setup before optimize");
  if (!num_gates_)
    return;
  // Launch kernels with device_gates_ and num_gates_. All pointers in this
  // view refer to device storage; the host circuit remains on the CPU.
  check(cudaDeviceSynchronize());
}

void CudaOptimizer::copy_to_host(Circuit &host) const {
  if (!ready_)
    throw std::logic_error("Call setup before copying results");
  if (host.num_gates != num_gates_ || host.num_qubits != num_qubits_)
    throw std::invalid_argument("Host circuit dimensions differ from device");
  if (num_gates_)
    check(cudaMemcpy(host.storage.get(), device_storage_, host.bytes(),
                     cudaMemcpyDeviceToHost));
}
