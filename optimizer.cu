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

CudaOptimizer::CudaOptimizer(int num_threads) : _num_threads(num_threads) {
  if (num_threads <= 0)
    throw std::invalid_argument("Thread count must be positive");
}

CudaOptimizer::~CudaOptimizer() {
  if (device_gates_)
    cudaFree(device_gates_);
  if (device_owning_pattern_)
    cudaFree(device_owning_pattern_);
  if (device_is_selected_)
    cudaFree(device_is_selected_);
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
  const CircuitView view = circuit_view(device_storage_, num_gates_);
  check(cudaMalloc(&device_gates_, sizeof(CircuitView)));
  check(cudaMemcpy(device_gates_, &view, sizeof(CircuitView),
                   cudaMemcpyHostToDevice));
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

// Each helper checks one pattern, given the starting H and its next two gates.
static __device__ inline bool match_hp(int idx, int next_1_idx, int next_2_idx,
                                       const CircuitView &circuit,
                                       int *owning_pattern, int *is_selected) {
  if (circuit.gate_type[next_1_idx] != GATE::RZ ||
      circuit.gate_type[next_2_idx] != GATE::HADAMARD)
    return false;

  float angle[] = {circuit.angle_pi[0][next_1_idx],
                   circuit.angle_pi[1][next_1_idx]};
  if (!is_phase_gate(angle))
    return false;

  int gate_indices[] = {idx, next_1_idx, next_2_idx};
  claim_pattern(idx, Pattern::HP_TOGGLE, gate_indices, 3, owning_pattern,
                is_selected);
  return true;
}

static __device__ inline bool match_hpc(int idx, int next_1_idx, int next_2_idx,
                                        const CircuitView &circuit,
                                        int *owning_pattern, int *is_selected) {
  if (circuit.gate_type[next_1_idx] != GATE::RZ ||
      circuit.gate_type[next_2_idx] != GATE::CNOT)
    return false;

  float next_1_angle[] = {circuit.angle_pi[0][next_1_idx],
                          circuit.angle_pi[1][next_1_idx]};
  if (!is_phase_gate(next_1_angle))
    return false;

  // The starting H must be on the CNOT target.
  if (circuit.qubit_0[idx] != circuit.qubit_1[next_2_idx])
    return false;

  int next_3_idx = circuit.next_idx[1][next_2_idx];
  if (next_3_idx == -1 || circuit.gate_type[next_3_idx] != GATE::RZ)
    return false;

  // The second phase gate must have the opposite angle.
  if (circuit.angle_pi[0][next_3_idx] != -1 * next_1_angle[0] ||
      circuit.angle_pi[1][next_3_idx] != 0)
    return false;

  int next_4_idx = circuit.next_idx[0][next_3_idx];
  if (next_4_idx == -1 || circuit.gate_type[next_4_idx] != GATE::HADAMARD)
    return false;

  int gate_indices[] = {idx, next_1_idx, next_2_idx, next_3_idx, next_4_idx};
  claim_pattern(idx, Pattern::HPC_TOGGLE, gate_indices, 5, owning_pattern,
                is_selected);
  return true;
}

static __device__ inline bool
match_hcnot(int idx, int next_1_idx, int next_2_idx, const CircuitView &circuit,
            int *owning_pattern, int *is_selected) {
  if (circuit.gate_type[next_1_idx] != GATE::CNOT ||
      circuit.gate_type[next_2_idx] != GATE::HADAMARD)
    return false;

  // The starting H must be on the CNOT control.
  if (circuit.qubit_0[idx] != circuit.qubit_0[next_1_idx])
    return false;

  int target_prev_1_idx = circuit.prev_idx[1][next_1_idx];
  if (target_prev_1_idx == -1 ||
      circuit.gate_type[target_prev_1_idx] != GATE::HADAMARD)
    return false;

  int target_next_2_idx = circuit.next_idx[1][next_1_idx];
  if (target_next_2_idx == -1 ||
      circuit.gate_type[target_next_2_idx] != GATE::HADAMARD)
    return false;

  int gate_indices[] = {idx, next_1_idx, next_2_idx, target_prev_1_idx,
                        target_next_2_idx};
  claim_pattern(idx, Pattern::HCNOT_CANCEL, gate_indices, 5, owning_pattern,
                is_selected);
  return true;
}

/** @brief Find the start of Hadamard patterns to optimize
  1. H - P - H => P^ - H - P^
  2. H - P - CNOT target - P^ - H => P^ - CNOT target - P
  3. H - CNOT control - H => CNOT target
     H - CNOT target  - H => CNOT control
 */
__global__ void match(int N, const CircuitView &circuit, int *owning_pattern,
                      int *is_selected) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  /* only consider patterns starting with Hadamard at least 3 long */
  if (idx < N - 2 && circuit.gate_type[idx] == GATE::HADAMARD) {
    int next_1_idx = circuit.next_idx[0][idx];
    if (next_1_idx == -1)
      return;

    int next_2_idx = circuit.next_idx[0][next_1_idx];
    if (next_2_idx == -1)
      return;

    if (match_hp(idx, next_1_idx, next_2_idx, circuit, owning_pattern,
                 is_selected))
      return;
    if (match_hpc(idx, next_1_idx, next_2_idx, circuit, owning_pattern,
                  is_selected))
      return;
    match_hcnot(idx, next_1_idx, next_2_idx, circuit, owning_pattern,
                is_selected);
  }
}

/**
 * @brief For the chosen disjoint optimizations, update the circuit
 * TODO: should next_index be populated for more than just the RZ/CNOT case?
 */
__global__ void apply(int N, CircuitView &circuit, int *device_is_selected,
                      int *next_index) {
  // index of the first (h) block
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < N) {
    // check if the device is selected
    switch (device_is_selected[i]) {
    case Pattern::NONE: {
      break;
    }
    case Pattern::HP_TOGGLE: {
      // switch the H to P and vice versa
      int h1 = idx;
      int p = circuit.next_idx[0][h1];
      int h2 = circuit.next_idx[0][p];

      float p_angle = circuit.angle_pi[0][p];

      // change h1 and h2 to p_conj types
      circuit.gate_type[h1] = GATE::RZ;
      circuit.angle_pi[0][h1] = -1 * p_angle;
      circuit.angle_pi[1][h1] = 0;

      circuit.gate_type[h2] = GATE::RZ;
      circuit.angle_pi[0][h2] = -1 * p_angle;
      circuit.angle_pi[1][h2] = 0;

      // change p to h
      circuit.gate_type[p] = GATE::HADAMARD;

      break;
    }
    case Pattern::HPC_TOGGLE: {
      int h1 = idx;
      int p = circuit.next_idx[0][h1];
      int cnot = circuit.next_idx[0][p];
      int pconj = circuit.next_idx[1][cnot];
      int h2 = circuit.next_idx[0][pconj];

      // tombstone the H gates
      circuit.gate_type[h1] = GATE::TOMBSTONE;
      circuit.gate_type[h2] = GATE::TOMBSTONE;

      // flip both p and pconj
      float p_angle = circuit.angle_pi[0][p];
      circuit.angle_pi[0][p] = -1 * p_angle;
      circuit.angle_pi[0][pconj] = p_angle;
      break;
    }
    case Pattern::HCNOT_CANCEL: {
      int h1a = idx;
      int cnot = circuit.next_idx[0][h1a];
      int h1b = circuit.prev_idx[1][cnot];
      int h2a = circuit.next_idx[0][cnot];
      int h2b = circuit.next_idx[1][cnot];

      // insert tombstone in all the H gates (not sure if other variables need
      // to be changed)
      circuit.gate_type[h1a] = GATE::TOMBSTONE;
      circuit.gate_type[h1b] = GATE::TOMBSTONE;
      circuit.gate_type[h2a] = GATE::TOMBSTONE;
      circuit.gate_type[h2b] = GATE::TOMBSTONE;

      // flip the control and target of cnot gate
      int control = circuit.qubit_0[cnot];
      circuit.qubit_0[cnot] = circuit.qubit_1[cnot];
      circuit.qubit_1[cnot] = control;
      // Link slots must continue to refer to the corresponding operand wire.
      int next = circuit.next_idx[0][cnot];
      circuit.next_idx[0][cnot] = circuit.next_idx[1][cnot];
      circuit.next_idx[1][cnot] = next;
      int prev = circuit.prev_idx[0][cnot];
      circuit.prev_idx[0][cnot] = circuit.prev_idx[1][cnot];
      circuit.prev_idx[1][cnot] = prev;
      break;
    }
    case Pattern:
    HH_CANCEL: {
      int h1 = idx;
      int h2 = circuit.next_idx[0][h1];

      // write tombstones to the H gates
      circuit.gate_type[h1] = GATE::TOMBSTONE;
      circuit.gate_type[h2] = GATE::TOMBSTONE;
      break;
    }
    case Pattern:
    XX_CANCEL: {
      int x1 = idx;
      int x2 = circuit.next_idx[0][x1];

      // write tombstones to the X gates
      circuit.gate_type[x1] = GATE::TOMBSTONE;
      circuit.gate_type[x2] = GATE::TOMBSTONE;
      break;
    }
    case Pattern:
    CNOT_CANCEL: {
      int cx1 = idx;
      int cx2 = next_index[idx];

      circuit.gate_type[cx1] = GATE::TOMBSTONE;
      circuit.gate_type[cx2] = GATE::TOMBSTONE;
      break;
    }
    case Pattern:
    RZ_MERGE: {
      int rz1 = idx;
      int rz2 = next_index[idx];

      // write the combined angle into the first index
      circuit.angle_pi[0][rz1] += circuit.angle_pi[0][rz2];
      circuit.angle_pi[1][rz1] += circuit.angle_pi[1][rz2];

      circuit.gate_type[rz2] = GATE::TOMBSTONE;
      break;
    }
    default: {
    }
    }
  }
}

/**
 * @brief Unlink tombstones and rebuild wire starts and live-gate links.
 * Gate data stays in place; physical storage packing is a separate operation.
 * @pre One thread per qubit, after apply has completed. Tombstones retain
 * their qubit fields and links so deleted two-qubit gates remain traversable.
 */
__global__ void relink(int N, int num_qubits, CircuitView &circuit) {
  int qubit = blockDim.x * blockIdx.x + threadIdx.x;

  if (qubit < num_qubits) {
    int next = circuit.qstart_indices[qubit];
    int prev = -1;
    int prev_operand = 0;
    circuit.qstart_indices[qubit] = -1;

    while (next != -1) {
      // Use operands rather than gate type: a tombstone may have been a CNOT.
      int operand = circuit.qubit_0[next] == qubit ? 0 : 1;
      int successor = circuit.next_idx[operand][next];
      if (circuit.gate_type[next] != GATE::TOMBSTONE) {
        circuit.prev_idx[operand][next] = prev;
        if (prev != -1) {
          circuit.next_idx[prev_operand][prev] = next;
        } else {
          circuit.qstart_indices[qubit] = next;
        }
        prev = next;
        prev_operand = operand;
      }
      next = successor;
    }
    if (prev != -1)
      circuit.next_idx[prev_operand][prev] = -1;
  }
}

/**
 * @brief pushes X gates to end of circuit, cancels where applicable
 */
__global__ void not_propagation(int N, int num_qubits, CircuitView &circuit) {
  /* ensure only one thread is launched */

  /* need to maintain two bits per qubit: carry of X, Z */
  for (int idx = 0; idx < N; idx++) {
    GATE gate = circuit.gate_type[idx];
    if (gate == GATE::TOMBSTONE) {
      continue;
    }
    int q0 = circuit.qubit_0[idx];
    int q1 = circuit.qubit_1[idx];

    if (gate == GATE::NOT) {
      /* has X carry -> cancel X gate */
      if (circuit.xz_carry[q0] & X_CARRY) {
        circuit.gate_type[idx] = TOMBSTONE;
        /** TODO: extract helper to clean up links */
      }
      circuit.xz_carry[q0] &= X_CARRY;
    } else if (gate == GATE::HADAMARD) {
      int carry = circuit.xz_carry[q0];
      /* swap the X, Z carry: two lowest bits */
      circuit.xz_carry[q0] =
          ((carry & X_CARRY) << 1) | ((carry & Z_CARRY) >> 1);
    } else if (gate == GATE::RZ) {
      /* X carry negates angle */
      if (circuit.xz_carry[q0] & X_CARRY) {
        circuit.angle_pi[0][idx] *= -1;
        circuit.angle_pi[1][idx] *= -1;
      }
      /* spend Z carry on RZ gate */
      if (circuit.xz_carry[q0] & Z_CARRY) {
        circuit.angle_pi[0][idx] = (circuit.angle_pi[0][idx] + 1) % 2;
        circuit.xz_carry[q0] = 0;
      }
    } else { // GATE::CNOT
      /* xor control onto target */
      circuit.xz_carry[q1] ^= circuit.xz_carry[q0];
    }
  }

  /* write final X, or Z to each qubit if has active carry value */
  for (int qubit = 0; qubit < num_qubits; qubit++) {
    int carry = circuit.xz_carry[qubit];
    if (carry & X_CARRY) {
      /** TODO: add final X gate */
    }
    if (carry & Z_CARRY) {
      /** TODO: add final Z gate */
    }
  }
}

void CudaOptimizer::optimize() {
  if (!ready_)
    throw std::logic_error("Call setup before optimize");
  if (!num_gates_)
    return;
  // Bind kernel references to the device-resident view. Its arrays also
  // reside on the GPU; dereferencing here binds a reference without a host
  // read.

  // apply the match kernel
  int numBlocks = (num_gates_ + _num_threads - 1) / _num_threads;
  match<<<numBlocks, _num_threads>>>(
      num_gates_, *device_gates_, device_owning_pattern_, device_is_selected_);
  check(cudaGetLastError());

  // apply the apply kernels
  apply<<<numBlocks, _num_threads>>>(num_gates_, *device_gates_,
                                     device_is_selected_);
  check(cudaGetLastError());

  int qubitBlocks = (num_qubits_ + _num_threads - 1) / _num_threads;
  relink<<<qubitBlocks, _num_threads>>>(num_gates_, num_qubits_,
                                        *device_gates_);

  // launch only one thread
  not_propagation<<<1, 1>>>(*device_gates_);

  check(cudaGetLastError());

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
