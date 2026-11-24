// Include the implementation to exercise its static device helpers directly.
// This executable must not also link a separately compiled optimizer.cu.
#include "../optimizer.cu"
#include "qasm.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// Figure 5, printed page 12, https://arxiv.org/pdf/1710.07345.
// Indices are positions in the input gate array, excluding the qreg declaration.
struct AdvanceCase {
  const char *name;
  const char *body;
  int start;
  int expected;
};

static const AdvanceCase advance_cases[] = {
    {"RZ through H-CX(target)-H", "rz(0.3) q[1];\nh q[1];\ncx q[0],q[1];\nh q[1];\n", 0, 3},
    {"RZ through CX-RZ-CX parity block", "rz(0.3) q[1];\ncx q[0],q[1];\nrz(0.7) q[1];\ncx q[0],q[1];\n", 0, 3},
    {"RZ through CX control", "rz(0.3) q[0];\ncx q[0],q[1];\n", 0, 1},
    {"RZ control returns actual index", "x q[2];\nrz(0.3) q[0];\nx q[2];\ncx q[0],q[1];\n", 1, 3},
    {"CX through shared-target CX", "cx q[0],q[2];\ncx q[1],q[2];\n", 0, 1},
    {"CX through shared-control CX", "cx q[0],q[1];\ncx q[0],q[2];\n", 0, 1},
    {"CX through H-CX(control)-H on target", "cx q[0],q[1];\nh q[1];\ncx q[1],q[2];\nh q[1];\n", 0, 3},
    {"RZ on CX target alone is blocked", "rz(0.3) q[1];\ncx q[0],q[1];\n", 0, 0},
    {"RZ cannot cross lone H", "rz(0.3) q[0];\nh q[0];\n", 0, 0},
    {"RZ cannot cross X", "rz(0.3) q[0];\nx q[0];\n", 0, 0},
    {"RZ H-CX-H needs closing H", "rz(0.3) q[1];\nh q[1];\ncx q[0],q[1];\n", 0, 0},
    {"RZ H-CX(control)-H is blocked", "rz(0.3) q[0];\nh q[0];\ncx q[0],q[1];\nh q[0];\n", 0, 0},
    {"RZ parity block needs closing CX", "rz(0.3) q[1];\ncx q[0],q[1];\nrz(0.7) q[1];\n", 0, 0},
    {"RZ parity block needs same CX control", "rz(0.3) q[1];\ncx q[0],q[1];\nrz(0.7) q[1];\ncx q[2],q[1];\n", 0, 0},
    {"RZ parity block rejects closing CX control role", "rz(0.3) q[1];\ncx q[0],q[1];\nrz(0.7) q[1];\ncx q[1],q[0];\n", 0, 0},
    {"CX chain target-to-control is blocked", "cx q[0],q[1];\ncx q[1],q[2];\n", 0, 0},
    {"CX chain control-to-target is blocked", "cx q[0],q[1];\ncx q[2],q[0];\n", 0, 0},
    {"CX reversed operands are blocked", "cx q[0],q[1];\ncx q[1],q[0];\n", 0, 0},
    {"shared control cannot hide target blocker", "cx q[0],q[1];\nh q[1];\ncx q[0],q[2];\n", 0, 0},
    {"shared target cannot hide control blocker", "cx q[0],q[1];\nh q[0];\ncx q[2],q[1];\n", 0, 0},
    {"CX H-CX-H needs closing H", "cx q[0],q[1];\nh q[1];\ncx q[1],q[2];\n", 0, 0},
    {"CX H-CX-X is blocked", "cx q[0],q[1];\nh q[1];\ncx q[1],q[2];\nx q[1];\n", 0, 0},
    {"RZ at end", "rz(0.3) q[0];\n", 0, 0},
    {"CX at end", "cx q[0],q[1];\n", 0, 0},
};

struct TerminateCase {
  const char *name;
  const char *body;
  bool expected;
};

// Termination is now tested through match_pair, which owns partner detection.
static const TerminateCase terminate_cases[] = {
    {"adjacent RZs can merge", "rz(0.3) q[0];\nrz(0.7) q[0];\n", true},
    {"numeric and pi angles can merge", "rz(pi/4) q[0];\nrz(0.125) q[0];\n", true},
    {"RZ adjacency follows wire links", "rz(0.3) q[0];\nx q[2];\nrz(0.7) q[0];\n", true},
    {"RZ followed by H", "rz(0.3) q[0];\nh q[0];\n", false},
    {"RZ followed by CX", "rz(0.3) q[0];\ncx q[0],q[1];\n", false},
    {"RZ at end", "rz(0.3) q[0];\n", false},
    {"identical CXs cancel", "cx q[0],q[1];\ncx q[0],q[1];\n", true},
    {"CX adjacency follows both wire links", "cx q[0],q[1];\nx q[2];\ncx q[0],q[1];\n", true},
    {"reversed CXs do not cancel", "cx q[0],q[1];\ncx q[1],q[0];\n", false},
    {"same control alone is insufficient", "cx q[0],q[1];\ncx q[0],q[2];\n", false},
    {"same target alone is insufficient", "cx q[0],q[2];\ncx q[1],q[2];\n", false},
    {"target blocker prevents CX cancellation", "cx q[0],q[1];\nh q[1];\ncx q[0],q[1];\n", false},
    {"control blocker prevents CX cancellation", "cx q[0],q[1];\nh q[0];\ncx q[0],q[1];\n", false},
    {"CX at end", "cx q[0],q[1];\n", false},
};

struct PairCase {
  const char *name;
  const char *body;
  bool expected;
  int partner; // Expected matching gate; -1 means no match.
  Pattern pattern; // HH/XX must claim both endpoints. RZ/CX return a partner.
};

static const PairCase pair_cases[] = {
    {"HH adjacent", "h q[0];\nh q[0];\n", true, 1, HH_CANCEL},
    {"XX adjacent", "x q[0];\nx q[0];\n", true, 1, XX_CANCEL},
    {"HH with unrelated wire", "h q[0];\nx q[2];\nh q[0];\n", true, 2, HH_CANCEL},
    {"XX with unrelated wire", "x q[0];\nh q[2];\nx q[0];\n", true, 2, XX_CANCEL},
    {"H then X is not a pair", "h q[0];\nx q[0];\n", false, -1, NONE},
    {"HH blocked by RZ", "h q[0];\nrz(0.3) q[0];\nh q[0];\n", false, -1, NONE},
    {"XX blocked by H", "x q[0];\nh q[0];\nx q[0];\n", false, -1, NONE},
    {"adjacent RZ inverse", "rz(pi/4+0.125) q[0];\nrz(-pi/4-0.125) q[0];\n", true, 1, NONE},
    {"adjacent RZ merge", "rz(pi/4) q[0];\nrz(0.125) q[0];\n", true, 1, NONE},
    {"RZ merge through control CX", "rz(0.3) q[0];\ncx q[0],q[1];\nrz(0.7) q[0];\n", true, 2, NONE},
    {"RZ merge through two control CXs", "rz(0.3) q[0];\ncx q[0],q[1];\ncx q[0],q[2];\nrz(0.7) q[0];\n", true, 3, NONE},
    {"RZ merge through H-CX-H", "rz(0.3) q[1];\nh q[1];\ncx q[0],q[1];\nh q[1];\nrz(0.7) q[1];\n", true, 4, NONE},
    {"RZ merge through parity block", "rz(0.3) q[1];\ncx q[0],q[1];\nrz(0.7) q[1];\ncx q[0],q[1];\nrz(0.2) q[1];\n", true, 4, NONE},
    {"RZ cannot merge through bare CX target", "rz(0.3) q[1];\ncx q[0],q[1];\nrz(0.7) q[1];\n", false, -1, NONE},
    {"RZ cannot merge through H", "rz(0.3) q[0];\nh q[0];\nrz(0.7) q[0];\n", false, -1, NONE},
    {"CX adjacent pair", "cx q[0],q[1];\ncx q[0],q[1];\n", true, 1, NONE},
    {"CX pair through shared control", "cx q[0],q[1];\ncx q[0],q[2];\ncx q[0],q[1];\n", true, 2, NONE},
    {"CX pair through shared target", "cx q[0],q[2];\ncx q[1],q[2];\ncx q[0],q[2];\n", true, 2, NONE},
    {"CX pair through H-CX-H", "cx q[0],q[1];\nh q[1];\ncx q[1],q[2];\nh q[1];\ncx q[0],q[1];\n", true, 4, NONE},
    {"CX reversed pair is invalid", "cx q[0],q[1];\ncx q[1],q[0];\n", false, -1, NONE},
    {"CX control blocker", "cx q[0],q[1];\nh q[0];\ncx q[0],q[1];\n", false, -1, NONE},
};

struct RewriteCase {
  const char *name;
  const char *body;
  const char *expected;
};

static const RewriteCase rewrite_cases[] = {
    {"HH elimination", "h q[0];\nh q[0];\n", ""},
    {"XX elimination", "x q[0];\nx q[0];\n", ""},
    {"HH elimination with unrelated wire", "h q[0];\nx q[2];\nh q[0];\n", "x q[2];\n"},
    {"mixed RZ inverse elimination", "rz(pi/4+0.125) q[0];\nrz(-pi/4-0.125) q[0];\n", ""},
    {"mixed RZ merge", "rz(pi/4) q[0];\nrz(0.125) q[0];\n", "rz(pi/4+0.125) q[0];\n"},
    {"RZ inverse through control CX", "rz(pi/4) q[0];\ncx q[0],q[1];\nrz(-pi/4) q[0];\n", "cx q[0],q[1];\n"},
    {"CX pair elimination", "cx q[0],q[1];\ncx q[0],q[1];\n", ""},
    {"blocked XX preserved", "x q[0];\nh q[0];\nx q[0];\n", "x q[0];\nh q[0];\nx q[0];\n"},
    {"blocked RZ pair preserved", "rz(0.3) q[1];\ncx q[0],q[1];\nrz(-0.3) q[1];\n", "rz(0.3) q[1];\ncx q[0],q[1];\nrz(-0.3) q[1];\n"},
};

static Circuit parse(const char *body) {
  std::istringstream input(std::string("qreg q[3];\n") + body);
  return read_circuit(input);
}

static void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

// Independent, small full-operator oracle. Check every basis state with one
// common global phase; do not derive expected commutations from the helpers.
using Complex = std::complex<double>;
static std::vector<Complex> simulate(const Circuit &c, const std::vector<int> &order, int basis) {
  std::vector<Complex> state(1 << c.num_qubits);
  state[basis] = 1;
  for (int g : order) {
    const auto type = c.gates.gate_type[g];
    if (type == TOMBSTONE) continue;
    const int bit = 1 << c.gates.qubit_0[g];
    if (type == RZ) {
      double theta = c.gates.angle_pi[0][g] * std::acos(-1.0) + c.gates.angle_pi[1][g];
      for (int i = 0; i < static_cast<int>(state.size()); ++i)
        state[i] *= std::exp(Complex(0, (i & bit ? 0.5 : -0.5) * theta));
    } else {
      for (int i = 0; i < static_cast<int>(state.size()); ++i) {
        if (type == CNOT) {
          const int target = 1 << c.gates.qubit_1[g];
          if ((i & bit) && !(i & target)) std::swap(state[i], state[i | target]);
        } else if (!(i & bit)) {
          const Complex a = state[i], b = state[i | bit];
          state[i] = type == HADAMARD ? (a + b) / std::sqrt(2.0) : b;
          state[i | bit] = type == HADAMARD ? (a - b) / std::sqrt(2.0) : a;
        }
      }
    }
  }
  return state;
}

static std::vector<int> order(const Circuit &c) {
  std::vector<int> result(c.num_gates);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

static bool equivalent(const Circuit &a, const std::vector<int> &ao,
                       const Circuit &b, const std::vector<int> &bo) {
  Complex phase = 1;
  bool have_phase = false;
  for (int basis = 0; basis < (1 << a.num_qubits); ++basis) {
    auto av = simulate(a, ao, basis), bv = simulate(b, bo, basis);
    for (std::size_t i = 0; i < av.size(); ++i) {
      if (!have_phase && std::abs(av[i]) > 1e-7 && std::abs(bv[i]) > 1e-7) {
        phase = av[i] / bv[i];
        have_phase = true;
      }
      if (std::abs(av[i] - phase * bv[i]) > 1e-6) return false;
    }
  }
  return have_phase && std::abs(std::abs(phase) - 1) < 1e-6;
}

struct Result {
  int value, next, pattern, owner_a, owner_b;
  int gate_count;
  int gate_indices[pair_gate_capacity];
};
enum Operation { ADVANCE, TERMINATE, PAIR };

__global__ void probe(CircuitView circuit, int start, int next, Operation op,
                      int *owners, int *selected, Result *result, int capacity) {
  if (op == ADVANCE) {
    int gate_indices[pair_gate_capacity];
    for (int &gate : gate_indices) gate = -777;
    gate_indices[0] = start;
    int gate_count = 1;
    if (circuit.gate_type[start] == RZ) {
      advance_rz(circuit, start, circuit.next_idx[0][start], gate_indices,
                 gate_count, capacity);
    } else {
      cuda::std::pair<int, int> cursor = {circuit.next_idx[0][start],
                                        circuit.next_idx[1][start]};
      advance_cnot(circuit, start, cursor, gate_indices, gate_count, capacity);
    }
    // The split helpers expose the next cursor; these fixtures identify the
    // last crossed gate, available from the accumulated ownership buffer.
    result->value = gate_indices[gate_count - 1];
    result->gate_count = gate_count;
    for (int i = 0; i < pair_gate_capacity; ++i)
      result->gate_indices[i] = gate_indices[i];
  }
  else if (op == TERMINATE)
    result->value = match_pair(start, next, circuit, owners, selected, &result->next, 0);
  else {
    result->value = match_pair(start, next, circuit, owners, selected, &result->next, 0);
    result->pattern = selected[start];
    result->owner_a = owners[start];
    result->owner_b = result->value ? owners[result->next] : INT_MAX;
  }
}

// The single-thread probe isolates matching logic from pattern competition.
static Result evaluate(const Circuit &c, int start, Operation op,
                       int capacity = pair_gate_capacity) {
  struct Allocation {
    unsigned char *storage = nullptr;
    int *owners = nullptr, *selected = nullptr;
    Result *result = nullptr;
    ~Allocation() { cudaFree(result); cudaFree(selected); cudaFree(owners); cudaFree(storage); }
  } d;
  check(cudaMalloc(&d.storage, c.bytes()));
  check(cudaMemcpy(d.storage, c.storage.get(), c.bytes(), cudaMemcpyHostToDevice));
  check(cudaMalloc(&d.owners, c.num_gates * sizeof(int)));
  check(cudaMalloc(&d.selected, c.num_gates * sizeof(int)));
  std::vector<int> owners(c.num_gates, INT_MAX), selected(c.num_gates, NONE);
  check(cudaMemcpy(d.owners, owners.data(), owners.size() * sizeof(int), cudaMemcpyHostToDevice));
  check(cudaMemcpy(d.selected, selected.data(), selected.size() * sizeof(int), cudaMemcpyHostToDevice));
  check(cudaMalloc(&d.result, sizeof(Result)));
  Result result{-777, -777, NONE, INT_MAX, INT_MAX};
  check(cudaMemcpy(d.result, &result, sizeof(result), cudaMemcpyHostToDevice));
  int next = c.gates.next_idx[0][start];
  require(op != PAIR || next >= 0, "pair fixture must have a next gate");
  probe<<<1, 1>>>(circuit_view(d.storage, c.num_gates, c.num_qubits), start, next, op,
                  d.owners, d.selected, d.result, capacity);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  check(cudaMemcpy(&result, d.result, sizeof(result), cudaMemcpyDeviceToHost));
  return result;
}

static void expect(int actual, int expected, const char *field) {
  require(actual == expected, std::string(field) + ": expected " +
          std::to_string(expected) + ", got " + std::to_string(actual));
}

int main() {
  try {
    int count = 0;
    auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && count == 0)) {
      std::cout << "SKIP: no CUDA devices available\n";
      return 0;
    }
    check(status);
    int passed = 0, failed = 0;
    for (int device = 0; device < count; ++device) {
      check(cudaSetDevice(device));
      std::cout << "Commutation tests on GPU " << device << '\n';
      auto run = [&](const char *group, const char *name, auto test) {
        try {
          test();
          ++passed;
          std::cout << "PASS [" << group << "] " << name << std::endl;
        } catch (const std::exception &e) {
          ++failed;
          std::cout << "FAIL [" << group << "] " << name << ": " << e.what() << std::endl;
        }
      };
      run("buffer", "records each crossed gate for all six commutation blocks", [&] {
        const int cases[] = {0, 1, 2, 4, 5, 6};
        for (int index : cases) {
          const auto &t = advance_cases[index];
          auto c = parse(t.body);
          auto result = evaluate(c, t.start, ADVANCE);
          expect(result.value, t.expected, t.name);
          expect(result.gate_count, t.expected + 1, "recorded count");
          for (int i = 0; i < result.gate_count; ++i)
            expect(result.gate_indices[i], i, "recorded gate index");
        }
      });
      run("buffer", "insufficient capacity leaves a three-gate block unmodified", [&] {
        auto c = parse(advance_cases[0].body);
        auto result = evaluate(c, 0, ADVANCE, 2);
        expect(result.value, 0, "blocked advance");
        expect(result.gate_count, 1, "unchanged count");
        expect(result.gate_indices[0], 0, "start gate");
        for (int i = 1; i < pair_gate_capacity; ++i)
          expect(result.gate_indices[i], -777, "untouched slot");
      });
      run("buffer", "full buffer rejects a one-gate block", [&] {
        auto c = parse(advance_cases[2].body);
        auto result = evaluate(c, 0, ADVANCE, 1);
        expect(result.value, 0, "blocked advance");
        expect(result.gate_count, 1, "unchanged count");
        expect(result.gate_indices[1], -777, "untouched slot");
      });
      for (const auto &t : advance_cases) run("commute", t.name, [&] {
        auto c = parse(t.body);
        if (t.expected != t.start) {
          auto moved = order(c);
          moved.erase(moved.begin() + t.start);
          moved.insert(moved.begin() + t.expected, t.start);
          require(equivalent(c, order(c), c, moved), "invalid commutation reference");
        }
        auto result = evaluate(c, t.start, ADVANCE);
        // A false-positive advance must also preserve the complete operator.
        if (result.value > t.start && result.value < c.num_gates) {
          auto moved = order(c);
          moved.erase(moved.begin() + t.start);
          moved.insert(moved.begin() + result.value, t.start);
          require(equivalent(c, order(c), c, moved), "reported commute changes circuit operator");
        }
        expect(result.value, t.expected, "advance index");
      });
      for (const auto &t : terminate_cases) run("terminate", t.name, [&] {
        auto c = parse(t.body);
        expect(evaluate(c, 0, TERMINATE).value, t.expected, "termination");
      });
      for (const auto &t : pair_cases) run("pair", t.name, [&] {
        auto c = parse(t.body);
        // Independently validate that each positive endpoint is cancellable
        // or mergeable while retaining all intervening gates in order.
        if (t.expected) {
          auto merged = parse(t.body);
          if (merged.gates.gate_type[0] == RZ) {
            for (int component = 0; component < 2; ++component)
              merged.gates.angle_pi[component][t.partner] += merged.gates.angle_pi[component][0];
          } else merged.gates.gate_type[t.partner] = TOMBSTONE;
          merged.gates.gate_type[0] = TOMBSTONE;
          require(equivalent(c, order(c), merged, order(merged)), "invalid pair reference");
        }
        auto result = evaluate(c, 0, PAIR);
        expect(result.value, t.expected, "matched");
        if (t.expected && t.pattern != NONE) {
          expect(result.pattern, t.pattern, "selected pattern");
          expect(result.owner_a, 0, "first owner");
          expect(result.owner_b, 0, "second owner");
        } else if (t.expected) {
          expect(result.next, t.partner, "matching endpoint");
          expect(result.pattern, c.gates.gate_type[0] == RZ ? RZ_MERGE : CNOT_CANCEL,
                 "selected pattern");
          expect(result.owner_a, 0, "first owner");
          expect(result.owner_b, 0, "partner owner");
        }
        else expect(result.pattern, NONE, "selected pattern after rejection");
      });
      for (const auto &t : rewrite_cases) run("optimizer", t.name, [&] {
        auto c = parse(t.body), original = parse(t.body), expected = parse(t.expected);
        require(equivalent(original, order(original), expected, order(expected)), "invalid rewrite reference");
        {
          CudaOptimizer optimizer;
          optimizer.setup(c);
          optimizer.optimize();
          optimizer.copy_to_host(c);
        }
        require(equivalent(original, order(original), c, order(c)), "optimizer changes circuit operator");
        int live = 0;
        for (int i = 0; i < c.num_gates; ++i) live += c.gates.gate_type[i] != TOMBSTONE;
        expect(live, expected.num_gates, "surviving gate count");
      });
    }
    std::cout << "Commutation/cancellation: " << passed << " passed, " << failed << " failed.\n";
    return failed ? 1 : 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
