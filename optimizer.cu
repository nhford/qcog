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

    auto begin = thrust::device_pointer_cast(device_owning_pattern_);
    thrust::fill(begin, begin + host.num_gates, INT_MAX);

    check(cudaMalloc(&device_is_selected, host.num_gates * sizeof(int)));
  }
  num_gates_ = host.num_gates;
  num_qubits_ = host.num_qubits;
  device_gates_ = circuit_view(device_storage_, num_gates_);
  ready_ = true;
}

// __device__ __inline__ CudaOptimizer::Swap

/** @brief Find the start of Hadamard patterns to optimize */
__global__ void CudaOptimizer::kernelMatchPattern() {
  int tid = threadIdx.x;
  /* all patterns search at least three gates */
  if (tid >= num_gates_ - 2) {
    return;
  }

  int initial_qubit = device_gates_.qubit_0[tid];
  GATE initial_gate = device_gates_.gate_type[tid];

  if (initial_gate != GATE::HADAMARD) {
    return;
  }

  int next_1_idx = device_gates_.next_idx[tid][0];

  if (next_1_idx == -1) {
    return;
  }

  GATE next_1_gate = device_gates_.gate_type[next_1_idx];

  int next_2_idx = device_gates_.next_idx[next_1_idx][0];

  if (next_2_idx == -1) {
    return;
  }

  GATE next_2_gate = device_gates_.gate_type[next_2_idx];

  if (next_1_gate == GATE::RZ) {
    float *next_1_angle = device_gates_.angle_pi[next_1_idx];

    /* need be -pi/2 or pi/2 */
    if ((next_1_angle[0] != 0.5 && next_1_angle[0] != -0.5) ==
        next_1_angle[1] != 0) {
      return;
    }
    if (next_2_gate == GATE::HADAMARD) {
      /* pattern 1: H - P - H => P^ - H - P^ */

      int owns_initial = atomicMin(&device_owning_pattern_[tid], tid);
      int owns_next_1 = atomicMin(&device_owning_pattern_[next_1_idx], tid);
      int owns_next_2 = atomicMin(&device_owning_pattern_[next_2_idx], tid);

      /** TODO: split into helper function  */
      /* if tid owns everything it spans, then mark tid as selected */
      if (owns_initial >= tid && owns_next_1 >= tid && owns_next_2 >= tid) {
        device_is_selected[tid] = Pattern::HP_TOGGLE;

        /* need to de-select everything else */
        if (owns_initial < INT_MAX) {
          device_is_selected[owns_initial] = Pattern::None;
        }
        if (owns_next_1 < INT_MAX) {
          device_is_selected[owns_next_1] = Pattern::None;
        }
        if (owns_next_2 < INT_MAX) {
          device_is_selected[owns_next_2] = Pattern::None;
        }
      }

    } else if (next_2_gate == GATE::CNOT) {

      /* ensure initial qubit is the target */
      int target_qubit = device_gates_.qubit_1[next_2_idx];

      if (initial_qubit != target_qubit) {
        return;
      }

      int next_3_idx = device_gates_.next_idx[next_2_idx][0];

      if (next_3_idx == -1) {
        return;
      }
      GATE next_3_gate = device_gates_.gate_type[next_3_idx];

      if (next_3_gate != GATE::RZ) {
        return;
      }

      float *next_3_angle = device_gates_.angle_pi[next_3_idx];

      /* need be -pi/2 or pi/2 */
      if ((next_3_angle[0] != 0.5 && next_3_angle[0] != -0.5) ==
          next_3_angle[1] != 0) {
        return;
      }

      int next_4_idx = device_gates_.next_idx[next_3_idx][0];

      if (next_4_idx == -1) {
        return;
      }
      GATE next_4_gate = device_gates_.gate_type[next_4_idx];

      if (next_4_gate != GATE::HADAMARD) {
        return;
      }

      /* pattern 2: H - P - CNOT target - P^ - H => P^ - CNOT target - P */

      int owns_initial = atomicMin(&device_owning_pattern_[tid], tid);
      int owns_next_1 = atomicMin(&device_owning_pattern_[next_1_idx], tid);
      int owns_next_2 = atomicMin(&device_owning_pattern_[next_2_idx], tid);
      int owns_next_3 = atomicMin(&device_owning_pattern_[next_3_idx], tid);
      int owns_next_4 = atomicMin(&device_owning_pattern_[next_4_idx], tid);

      /** TODO: split into helper function  */
      /* if tid owns everything it spans, then mark tid as selected */
      if (owns_initial >= tid && owns_next_1 >= tid && owns_next_2 >= tid &&
          owns_next_3 >= tid && owns_next_4 >= tid) {
        device_is_selected[tid] = Pattern::HPC_TOGGLE;

        /* need to de-select everything else */
        if (owns_initial < INT_MAX) {
          device_is_selected[owns_initial] = Pattern::None;
        }
        if (owns_next_1 < INT_MAX) {
          device_is_selected[owns_next_1] = Pattern::None;
        }
        if (owns_next_2 < INT_MAX) {
          device_is_selected[owns_next_2] = Pattern::None;
        }
        if (owns_next_3 < INT_MAX) {
          device_is_selected[owns_next_3] = Pattern::None;
        }
        if (owns_next_4 < INT_MAX) {
          device_is_selected[owns_next_4] = Pattern::None;
        }
      }
    }

  } else if (next_gate == GATE::CNOT) {
    if (next_2_gate != GATE::HADAMARD) {
      return;
    }

    /* ensure initial qubit is the control */
    int control_qubit = device_gates_.qubit_1[next_1_idx];

    if (initial_qubit != control_qubit) {
      return;
    }

    int target_prev_1_idx = device_gates_.prev_idx[next_1_idx][1];

    if (target_prev_1_idx == -1) {
      return;
    }

    GATE target_prev_1_gate = device_gates_.gate_type[target_prev_1_idx];

    if (target_prev_1_gate != GATE::HADAMARD) {
      return;
    }

    int target_next_2_idx = device_gates_.next_idx[next_1_idx][1];

    if (target_next_2_idx == -1) {
      return;
    }

    GATE target_next_2_gate = device_gates_.gate_type[target_next_2_idx];

    if (target_next_2_gate != GATE::HADAMARD) {
      return;
    }

    /* pattern 3: H - CNOT control - H => CNOT target
                    H - CNOT target  - H => CNOT control
      */
    /** TODO: atomic min writes to device_owning_pattern_ */
    int owns_initial = atomicMin(&device_owning_pattern_[tid], tid);
    int owns_next_1 = atomicMin(&device_owning_pattern_[next_1_idx], tid);
    int owns_next_2 = atomicMin(&device_owning_pattern_[next_2_idx], tid);
    int owns_target_prev_1 =
        atomicMin(&device_owning_pattern_[target_prev_1_idx], tid);
    int owns_target_next_2 =
        atomicMin(&device_owning_pattern_[target_next_2_idx], tid);

    /** TODO: split into helper function  */
    /* if tid owns everything it spans, then mark tid as selected */
    if (owns_initial >= tid && owns_next_1 >= tid && owns_next_2 >= tid &&
        owns_target_prev_1 >= tid && owns_target_next_2 >= tid) {
      device_is_selected[tid] = Pattern::HCNOT_CANCELLATION;

      /* need to de-select everything else */
      if (owns_initial < INT_MAX) {
        device_is_selected[owns_initial] = Pattern::None;
      }
      if (owns_next_1 < INT_MAX) {
        device_is_selected[owns_next_1] = Pattern::None;
      }
      if (owns_next_2 < INT_MAX) {
        device_is_selected[owns_next_2] = Pattern::None;
      }
      if (owns_target_prev_1 < INT_MAX) {
        device_is_selected[owns_target_prev_1] = Pattern::None;
      }
      if (owns_target_next_2 < INT_MAX) {
        device_is_selected[owns_target_next_2] = Pattern::None;
      }
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