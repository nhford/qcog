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

      /* update each gate */
      device_gates_.gate_type[tid] = GATE::RZ;
      device_gates_.angle_pi[tid][0] = -next_1_angle[0];
      device_gates_.angle_pi[tid][1] = 0;

      device_gates_.gate_type[next_1_idx] = GATE::HADAMARD;
      device_gates_.angle_pi[next_1_idx][0] = 0;
      device_gates_.angle_pi[next_1_idx][1] = 0;

      device_gates_.gate_type[next_2_idx] = GATE::RZ;
      device_gates_.angle_pi[next_2_idx][0] = -next_1_angle[0];
      device_gates_.angle_pi[next_2_idx][1] = 0;

      /** TODO: atomic min writes to device_owning_pattern_ */

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

      /* update each gate */
      device_gates_.gate_type[tid] = GATE::TOMBSTONE;

      device_gates_.angle_pi[next_1_idx][0] *= -1;

      device_gates_.angle_pi[next_3_idx][0] *= -1;

      device_gates_.gate_type[next_4_idx] = GATE::TOMBSTONE;

      /** TODO: atomic min writes to device_owning_pattern_ */
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

    /* update each gate */
    device_gates_.gate_type[tid] = GATE::TOMBSTONE;
    device_gates_.gate_type[target_prev_1_idx] = GATE::TOMBSTONE;

    int temp = device_gates_.qubit_0[next_1_idx];
    device_gates_.qubit_0[next_1_idx] = device_gates_.qubit_1[next_1_idx];
    device_gates_.qubit_1[next_1_idx] = temp;

    device_gates_.gate_type[next_1_idx] = GATE::TOMBSTONE;
    device_gates_.gate_type[target_next_2_gate] = GATE::TOMBSTONE;
    /** TODO: atomic min writes to device_owning_pattern_ */
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
