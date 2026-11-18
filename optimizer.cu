#include "circuit.h"
#include "optimizer.h"
#include <climits>
#include <cuda_runtime.h>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

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
    /* need to initialize owning pattern to MAX value*/
    check(cudaMalloc(&device_owning_pattern_, host.num_gates * sizeof(int)));

    /** TODO: re-initialize these two on successive passes */

    auto begin = thrust::device_pointer_cast(device_owning_pattern_);
    thrust::fill(begin, begin + host.num_gates, INT_MAX);

    check(cudaMalloc(&device_is_selected_, host.num_gates * sizeof(int)));
    begin = thrust::device_pointer_cast(device_is_selected_);
    thrust::fill(begin, begin + host.num_gates, Pattern::NONE);
  }
  num_gates_ = host.num_gates;
  num_qubits_ = host.num_qubits;
  device_gates_ = circuit_view(device_storage_, num_gates_);
  ready_ = true;
}

/** @brief Asserts rz gate angle is pi/2 || -pi/2 */
static __device__ inline bool is_phase_gate(const float *angle) {
  return (angle[0] == 0.5f || angle[0] == -0.5f) && angle[1] == 0.0f;
}

//
/** @brief Update owning_pattern & is_selected for discovered pattern match
    Overlapping patterns give priority to the smaller starting index.
    note: Cascading overlaps can happen if p0 conflicts with p1, and p1
    conflicts with p2, then even if p0 and p2 are disjoint, only p0 is selected
 */
static __device__ inline void claim_pattern(int idx, Pattern pattern,
                                            const int *gate_indices,
                                            int gate_count, int *owning_pattern,
                                            int *is_selected) {
  atomicExch(&is_selected[idx], pattern);
  __threadfence(); // recall MFENCE from lec; ensures selection precedes loop

  for (int gate = 0; gate < gate_count; ++gate) {
    int previous_owner = atomicMin(&owning_pattern[gate_indices[gate]], idx);
    if (previous_owner < idx) {
      atomicExch(&is_selected[idx], Pattern::NONE);
    } else if (previous_owner > idx && previous_owner != INT_MAX) {
      atomicExch(&is_selected[previous_owner], Pattern::NONE);
    }
  }
}

/** @brief Find the start of Hadamard patterns to optimize
  1. H - P - H => P^ - H - P^
  2. H - P - CNOT target - P^ - H => P^ - CNOT target - P
  3. H - CNOT control - H => CNOT target
     H - CNOT target  - H => CNOT control
 */
__global__ void match(int N, CircuitView *circuit, int *owning_pattern,
                      int *is_selected) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  /* only consider patterns starting with Hadamard at least 3 long */
  if (idx < N - 2 && circuit->gate_type[idx] == GATE::HADAMARD) {
    int initial_qubit = circuit->qubit_0[idx];
    int next_1_idx = circuit->next_idx[0][idx];
    if (next_1_idx == -1)
      return;

    GATE next_1_gate = circuit->gate_type[next_1_idx];

    int next_2_idx = circuit->next_idx[0][next_1_idx];

    if (next_2_idx == -1) {
      return;
    }

    GATE next_2_gate = circuit->gate_type[next_2_idx];

    if (next_1_gate == GATE::RZ) {
      float next_1_angle[] = {circuit->angle_pi[0][next_1_idx],
                              circuit->angle_pi[1][next_1_idx]};

      /* need be -pi/2 or pi/2 */
      if (!is_phase_gate(next_1_angle)) {
        return;
      }
      if (next_2_gate == GATE::HADAMARD) {
        /* pattern 1: H - P - H => P^ - H - P^ */

        int gate_indices[] = {idx, next_1_idx, next_2_idx};
        claim_pattern(idx, Pattern::HP_TOGGLE, gate_indices, 3, owning_pattern,
                      is_selected);

      } else if (next_2_gate == GATE::CNOT) {
        /* ensure initial qubit is the target */
        int target_qubit = circuit->qubit_1[next_2_idx];

        if (initial_qubit != target_qubit) {
          return;
        }

        int next_3_idx = circuit->next_idx[1][next_2_idx];

        if (next_3_idx == -1) {
          return;
        }
        GATE next_3_gate = circuit->gate_type[next_3_idx];

        if (next_3_gate != GATE::RZ) {
          return;
        }

        float next_3_angle[] = {circuit->angle_pi[0][next_3_idx],
                                circuit->angle_pi[1][next_3_idx]};

        /* need be opposite of next_1_angle */
        if ((next_3_angle[0] != -1 * next_1_angle[0]) || next_3_angle[1] != 0) {
          return;
        }

        int next_4_idx = circuit->next_idx[0][next_3_idx];

        if (next_4_idx == -1) {
          return;
        }
        GATE next_4_gate = circuit->gate_type[next_4_idx];

        if (next_4_gate != GATE::HADAMARD) {
          return;
        }

        /* pattern 2: H - P - CNOT target - P^ - H => P^ - CNOT target - P */

        int gate_indices[] = {idx, next_1_idx, next_2_idx, next_3_idx,
                              next_4_idx};
        claim_pattern(idx, Pattern::HPC_TOGGLE, gate_indices, 5, owning_pattern,
                      is_selected);
      }

    } else if (next_1_gate == GATE::CNOT) {
      if (next_2_gate != GATE::HADAMARD) {
        return;
      }

      /* ensure initial qubit is the control */
      int control_qubit = circuit->qubit_0[next_1_idx];

      if (initial_qubit != control_qubit) {
        return;
      }

      int target_prev_1_idx = circuit->prev_idx[1][next_1_idx];

      if (target_prev_1_idx == -1) {
        return;
      }

      GATE target_prev_1_gate = circuit->gate_type[target_prev_1_idx];

      if (target_prev_1_gate != GATE::HADAMARD) {
        return;
      }

      int target_next_2_idx = circuit->next_idx[1][next_1_idx];

      if (target_next_2_idx == -1) {
        return;
      }

      GATE target_next_2_gate = circuit->gate_type[target_next_2_idx];

      if (target_next_2_gate != GATE::HADAMARD) {
        return;
      }

      /* pattern 3: H - CNOT control - H => CNOT target
                      H - CNOT target  - H => CNOT control
        */
      int gate_indices[] = {idx, next_1_idx, next_2_idx, target_prev_1_idx,
                            target_next_2_idx};
      claim_pattern(idx, Pattern::HCNOT_CANCELLATION, gate_indices, 5,
                    owning_pattern, is_selected);
    }
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

__global__ void apply(int N, CircuitView *circuit, int *device_is_selected) {
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

      std::pair<float, float> p_angle = {circuit->angle_pi[0][p],
                                         circuit->angle_pi[1][p]};

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
      int h1b = circuit->prev_idx[0][cnot] != h1a ? circuit->prev_idx[0][cnot]
                                                  : circuit->prev_idx[1][cnot];
      int h2a = circuit->next_idx[0][cnot];
      int h2b = circuit->next_idx[1][cnot];

      // insert tombstone in all the H gates (not sure if other variables need
      // to be changed)
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
      std::pair<float, float> p_angle = {circuit->angle_pi[0][p],
                                         circuit->angle_pi[1][p]};
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