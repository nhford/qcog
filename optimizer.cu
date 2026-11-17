#include "circuit.h"
#include "optimizer.h"
#include <cuda_runtime.h>
#include <stdexcept>

static void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

CudaOptimizer::~CudaOptimizer() {
  if (device_storage_) cudaFree(device_storage_);
}

void CudaOptimizer::setup(const Circuit &host) {
  if (ready_) throw std::logic_error("Optimizer already set up");
  if (host.num_gates) {
    check(cudaMalloc(&device_storage_, host.bytes()));
    check(cudaMemcpy(device_storage_, host.storage.get(), host.bytes(),
                     cudaMemcpyHostToDevice));
  }
  num_gates_ = host.num_gates;
  num_qubits_ = host.num_qubits;
  device_gates_ = circuit_view(device_storage_, num_gates_);
  ready_ = true;
}

void CudaOptimizer::optimize() {
  if (!ready_) throw std::logic_error("Call setup before optimize");
  if (!num_gates_) return;
  // Launch kernels with device_gates_ and num_gates_. All pointers in this
  // view refer to device storage; the host circuit remains on the CPU.
  check(cudaDeviceSynchronize());
}

void CudaOptimizer::copy_to_host(Circuit &host) const {
  if (!ready_) throw std::logic_error("Call setup before copying results");
  if (host.num_gates != num_gates_ || host.num_qubits != num_qubits_)
    throw std::invalid_argument("Host circuit dimensions differ from device");
  if (num_gates_)
    check(cudaMemcpy(host.storage.get(), device_storage_, host.bytes(),
                     cudaMemcpyDeviceToHost));
}

__global__ void 
apply(int N, CircuitView *circuit, int *device_is_selected) {
  // index of the first (h) block
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) {
    // check if the device is selected 
    switch (device_is_selected[i]) {
      case Pattern::NONE: {
        break;
      }
      case Pattern::HP_TOGGLE: {
        // switch the H to P and vice versa
        int h1 = i;
        int p = circuit->next_idx[0][h1];
        int h2 = circuit->next_idx[0][p];

        std::pair<float, float> p_angle = {circuit->angle_pi[0][p], circuit->angle_pi[1][p]};

        // change h1 and h2 to p_conj types
        circuit->gate_type[h1] = GATE::RZ;
        circuit->angle_pi[0][h1] = -1 * p_angle.first;
        circuit->angle_pi[1][h1] = -1 * p_angle.second;

        circuit->gate_type[h2] = GATE::RZ;
        circuit->angle_pi[0][h2] = -1 * p_angle.first;
        circuit->angle_pi[1][h2] = -1 * p_angle.second;

        // change p to h
        circuit->gate_type[p] = GATE::HADAMARD;

        break;
      }
      case Pattern::HCNOT_CANCELLATION: {
        int h1a = i;
        int cnot = circuit->next_idx[0][h1a];
        int h1b = circuit->prev_idx[0][cnot] != h1a ? circuit->prev_idx[0][cnot] : circuit->prev_idx[1][cnot];
        int h2a = circuit->next_idx[0][cnot];
        int h2b = circuit->next_idx[1][cnot];

        // insert tombstone in all the H gates (not sure if other variables need to be changed)
        circuit->gate_type[h1a] = GATE::TOMBSTONE;
        circuit->gate_type[h1b] = GATE::TOMBSTONE;
        circuit->gate_type[h2a] = GATE::TOMBSTONE;
        circuit->gate_type[h2b] = GATE::TOMBSTONE;

        // flip the control and target of cnot gate
        int control = circuit->qubit_0[cnot];
        circuit->qubit_0[cnot] = circuit->qubit_1[cnot];
        circuit->qubit_1[cnot] = control;
        break;
      }
      case Pattern::HPC_TOGGLE: {
        int h1 = i;
        int p = circuit->next_idx[0][h1];
        int cnot = circuit->next_idx[0][p];
        int pconj = circuit->next_idx[1][cnot];
        int h2 = circuit->next_idx[0][pconj];

        // tombstone the H gates
        circuit->gate_type[h1] = GATE::TOMBSTONE;
        circuit->gate_type[h2] = GATE::TOMBSTONE;

        // flip both p and pconj
        std::pair<float, float> p_angle = {circuit->angle_pi[0][p], circuit->angle_pi[1][p]};
        circuit->angle_pi[0][p] = -1 * p_angle.first;
        circuit->angle_pi[1][p] = -1 * p_angle.second;
        circuit->angle_pi[0][pconj] = p_angle.first;
        circuit->angle_pi[1][pconj] = p_angle.second;

      }
      default: {
      }
    }
  }
}