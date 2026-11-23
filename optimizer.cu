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

CudaOptimizer::CudaOptimizer(int num_threads): _num_threads(num_threads) {
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

static __device__ inline bool match_hcnot(int idx, int next_1_idx, int next_2_idx,
                                          const CircuitView &circuit,
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
  claim_pattern(idx, Pattern::HCNOT_CANCEL, gate_indices, 5,
                owning_pattern, is_selected);
  return true;
}

// A bounded per-thread buffer avoids host-only std::vector operations in device
// code. Reaching the limit conservatively stops the search.
constexpr int pair_gate_capacity = 64;

static __device__ inline bool append_gate_indices(
    int *gate_indices, int &gate_count, int capacity,
    const int *block, int block_count) {
  // Check the entire block first: failure must not append a partial block.
  if (block_count > capacity - gate_count) return false;
  for (int i = 0; i < block_count; ++i)
    gate_indices[gate_count++] = block[i];
  return true;
}

/** @brief returns the index of the last element of commutation group.
 * returns (original) idx if it cannot commute any further
 * idx: index of the RZ/CNOT gate
 * gate_indices: caller-owned array of the start gate and crossed gates.
 * gate_count: populated length, updated only for a complete successful block.
 * capacity: maximum number of indices the caller's array can hold.
 */
static __device__ inline
int single_commute_advance(const CircuitView& circuit, const int idx,
                           int *gate_indices, int &gate_count, int capacity) {
  auto next_idx = circuit.next_idx;
  if (circuit.gate_type[idx] == GATE::RZ) {
    int next_1_idx = next_idx[0][idx];
    int next_2_idx;
    int next_3_idx;
    if (next_1_idx != -1) {
      // c (control) pattern check
      if (circuit.gate_type[next_1_idx] == GATE::CNOT) {
        // check if cnot is control or target along the same wire
        if (circuit.prev_idx[0][next_1_idx] == idx) {
          const int block[] = {next_1_idx};
          if (!append_gate_indices(gate_indices, gate_count, capacity, block, 1))
            return idx;
          return next_1_idx;
        }
        else {
          // make sure middle gate is RZ
          next_2_idx = next_idx[1][next_1_idx];
          if (next_2_idx == -1 || circuit.gate_type[next_2_idx] != GATE::RZ)
            return idx;

          // make sure last gate is also cnot target
          next_3_idx = next_idx[0][next_2_idx];
          if (next_3_idx == -1 || circuit.gate_type[next_3_idx] != GATE::CNOT || circuit.prev_idx[1][next_3_idx] != next_2_idx)
            return idx;

          if (circuit.qubit_0[next_1_idx] != circuit.qubit_0[next_3_idx] ||
              circuit.qubit_1[next_1_idx] != circuit.qubit_1[next_3_idx] ||
              circuit.next_idx[0][next_1_idx] != next_3_idx)
            return idx;

          // entire gate sequence established -> true case
          const int block[] = {next_1_idx, next_2_idx, next_3_idx};
          if (!append_gate_indices(gate_indices, gate_count, capacity, block, 3))
            return idx;
          return next_3_idx;
        }
      }
      // h c h (target) pattern check
      else if (circuit.gate_type[next_1_idx] == GATE::HADAMARD) {
        // check middle gate is cnot target
        next_2_idx = next_idx[0][next_1_idx];
        if (next_2_idx == -1 || circuit.gate_type[next_2_idx] != GATE::CNOT || circuit.prev_idx[1][next_2_idx] != next_1_idx)
          return idx;

        // check 3rd gate is H
        next_3_idx = next_idx[1][next_2_idx];
        if (next_3_idx == -1 || circuit.gate_type[next_3_idx] != GATE::HADAMARD)
          return idx;

        // entire gate sequence complete -> return true
        const int block[] = {next_1_idx, next_2_idx, next_3_idx};
        if (!append_gate_indices(gate_indices, gate_count, capacity, block, 3))
          return idx;
        return next_3_idx;
      }
    }
    return idx;
  }
  else if (circuit.gate_type[idx] == GATE::CNOT) {
    int next_ctrl_idx = circuit.next_idx[0][idx];
    int next_tgt_idx = circuit.next_idx[1][idx];
    // check if control qubit matches
    if (next_ctrl_idx != -1 &&
        circuit.gate_type[next_ctrl_idx] == GATE::CNOT &&
        circuit.prev_idx[0][next_ctrl_idx] == idx) {
      const int block[] = {next_ctrl_idx};
      if (!append_gate_indices(gate_indices, gate_count, capacity, block, 1))
        return idx;
      return next_ctrl_idx;
    }
    // check if target qubit matches
    else if (next_tgt_idx != -1) {
      if (circuit.gate_type[next_tgt_idx] == GATE::CNOT &&
        circuit.prev_idx[1][next_tgt_idx] == idx) {
        const int block[] = {next_tgt_idx};
        if (!append_gate_indices(gate_indices, gate_count, capacity, block, 1))
          return idx;
        return next_tgt_idx;
      }
      else if (circuit.gate_type[next_tgt_idx] == GATE::HADAMARD) {
        int next_2_idx = circuit.next_idx[0][next_tgt_idx];
        // check middle is control cnot
        if (next_2_idx == -1 ||
            circuit.gate_type[next_2_idx] != GATE::CNOT ||
            circuit.prev_idx[0][next_2_idx] != next_tgt_idx) {
          return idx;
        }
        // check last qubit is H
        int next_3_idx = circuit.next_idx[0][next_2_idx];
        if (next_3_idx == -1 || circuit.gate_type[next_3_idx] != GATE::HADAMARD) {
          return idx;
        }

        // commute sequence complete -> return true
        const int block[] = {next_tgt_idx, next_2_idx, next_3_idx};
        if (!append_gate_indices(gate_indices, gate_count, capacity, block, 3))
          return idx;
        return next_3_idx;
      }
    }
  }
  return idx;
}

static __device__ inline bool check_terminate(const CircuitView &circuit, int idx) {
  int next_1_idx = circuit.next_idx[0][idx];
  if (next_1_idx == -1) return false;
  switch (circuit.gate_type[idx]) {
    case GATE::RZ:
      return circuit.gate_type[next_1_idx] == GATE::RZ;
    case GATE::CNOT:
      return circuit.gate_type[next_1_idx] == GATE::CNOT &&
        circuit.next_idx[1][idx] == next_1_idx;
    default:
      return false;
  }
}

static __device__ inline bool match_pair(int idx, int next_1_idx, const CircuitView &circuit,
                                         int *owning_pattern, int *is_selected, int *next_index,
                                         int threadidx) {
  int gate_indices[pair_gate_capacity] = {idx};
  int gate_count = 1;
  next_index[threadidx] = idx;
  if (next_1_idx < 0) return false;
  // Handle adjacent self-inverse single-qubit gates first.
  if (circuit.gate_type[idx] == circuit.gate_type[next_1_idx]) {
    switch (circuit.gate_type[idx]) {
      case GATE::HADAMARD: {
        gate_indices[gate_count++] = next_1_idx;
        next_index[threadidx] = next_1_idx;
        claim_pattern(idx, Pattern::HH_CANCEL, gate_indices, gate_count, owning_pattern, is_selected);
        return true;
      }
      case GATE::NOT: {
        gate_indices[gate_count++] = next_1_idx;
        next_index[threadidx] = next_1_idx;
        claim_pattern(idx, Pattern::XX_CANCEL, gate_indices, gate_count, owning_pattern, is_selected);
        return true;
      }
      default: break;
    }
  }
  if (circuit.gate_type[idx] == GATE::RZ || circuit.gate_type[idx] == GATE::CNOT) {
    // TODO: retain idx as the moving gate and use a separate wire cursor;
    // check for a partner at each position before attempting another advance.
    // need to consider all possible commutations towards the right
    int last_idx = idx;
    int temp = last_idx;
    temp = single_commute_advance(circuit, last_idx, gate_indices, gate_count,
                                  pair_gate_capacity);
    // keep commuting if possible
    while (last_idx != temp) {
      last_idx = temp;
      temp = single_commute_advance(circuit, last_idx, gate_indices, gate_count,
                                    pair_gate_capacity);
    }

    // when we can commute no more, check if next gate is same as current gate (can tombstone together)
    if (check_terminate(circuit, idx)) {
      // set the last index to cancel out with
      last_idx = circuit.next_idx[0][last_idx];
      if (last_idx < 0 ||
          !append_gate_indices(gate_indices, gate_count, pair_gate_capacity,
                               &last_idx, 1))
        return false;
      next_index[threadidx] = last_idx;

      const Pattern pattern = circuit.gate_type[idx] == GATE::RZ
          ? Pattern::RZ_MERGE : Pattern::CNOT_MERGE;
      claim_pattern(idx, pattern, gate_indices, gate_count, owning_pattern, is_selected);

      return true;
    }
    // failure: return original index
    next_index[threadidx] = idx;
    return false;
  }
  return false;
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

    if (match_hp(idx, next_1_idx, next_2_idx, circuit, owning_pattern, is_selected))
      return;
    if (match_hpc(idx, next_1_idx, next_2_idx, circuit, owning_pattern, is_selected))
      return;
    match_hcnot(idx, next_1_idx, next_2_idx, circuit, owning_pattern, is_selected);
  }
}

__global__ void apply(int N, CircuitView &circuit, int *device_is_selected) {
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
      int h1 = i;
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
      int h1a = i;
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
    }
    default: {
    }
    }
  }
}

__global__ void
compact(int N, CircuitView &circuit) {

}

void CudaOptimizer::optimize() {
  if (!ready_)
    throw std::logic_error("Call setup before optimize");
  if (!num_gates_)
    return;
  // Bind kernel references to the device-resident view. Its arrays also
  // reside on the GPU; dereferencing here binds a reference without a host read.

  // apply the match kernel
  int numBlocks = (num_gates_ + _num_threads - 1) / _num_threads;
  match<<<numBlocks, _num_threads>>>(num_gates_, *device_gates_, device_owning_pattern_, device_is_selected_);
  check(cudaGetLastError());
  // apply the apply kernels
  apply<<<numBlocks, _num_threads>>>(num_gates_, *device_gates_, device_is_selected_);
  check(cudaGetLastError());

  // TODO 

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

