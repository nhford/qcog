#include "optimizer.h"
#include "qasm.h"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

static void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

struct TestCase {
  const char *name;
  const char *input;
  const char *expected;
};

// Expected circuits describe the surviving gates after one optimizer pass.
static const TestCase cases[] = {
    {"empty circuit", "", ""},
    {"single gate", "h q[0];\n", "h q[0];\n"},
    {"positive HP", "h q[0];\nrz(pi/2) q[0];\nh q[0];\n",
     "rz(-pi/2) q[0];\nh q[0];\nrz(-pi/2) q[0];\n"},
    {"negative HP", "h q[0];\nrz(-pi/2) q[0];\nh q[0];\n",
     "rz(pi/2) q[0];\nh q[0];\nrz(pi/2) q[0];\n"},
    {"mixed angle is preserved", "h q[0];\nrz(pi/2+0.125) q[0];\nh q[0];\n",
     "h q[0];\nrz(pi/2+0.125) q[0];\nh q[0];\n"},
    {"numeric angle is preserved", "h q[0];\nrz(0.5) q[0];\nh q[0];\n",
     "h q[0];\nrz(0.5) q[0];\nh q[0];\n"},
    {"HPC", "h q[1];\nrz(pi/2) q[1];\ncx q[0], q[1];\nrz(-pi/2) q[1];\nh q[1];\n",
     "rz(-pi/2) q[1];\ncx q[0], q[1];\nrz(pi/2) q[1];\n"},
    {"HCNOT", "h q[0];\nh q[1];\ncx q[0], q[1];\nh q[0];\nh q[1];\n",
     "cx q[1], q[0];\n"},
};

static Circuit parse(const char *body) {
  std::istringstream input(std::string("qreg q[2];\n") + body);
  return read_circuit(input);
}

static Circuit read_file(const std::filesystem::path &path) {
  std::ifstream input(path);
  require(input.is_open(), "Cannot open " + path.string());
  return read_circuit(input);
}

static void equal(const Circuit &actual, const Circuit &expected) {
  require(actual.num_gates == expected.num_gates, "gate count differs");
  require(actual.num_qubits == expected.num_qubits, "qubit count differs");
  for (int i = 0; i < actual.num_gates; ++i) {
    require(actual.gates.gate_type[i] == expected.gates.gate_type[i], "gate type differs");
    require(actual.gates.qubit_0[i] == expected.gates.qubit_0[i] &&
            actual.gates.qubit_1[i] == expected.gates.qubit_1[i], "operands differ");
    for (int component = 0; component < 2; ++component) {
      require(actual.gates.angle_pi[component][i] == expected.gates.angle_pi[component][i],
              "angle component differs");
      require(actual.gates.next_idx[component][i] == expected.gates.next_idx[component][i] &&
              actual.gates.prev_idx[component][i] == expected.gates.prev_idx[component][i],
              "wire links differ");
    }
  }
}

int main(int argc, char **argv) {
  try {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && count == 0)) {
      std::cout << "SKIP: no CUDA devices available\n";
      return 0;
    }
    check(status);
    std::cout << "CUDA devices available: " << count << std::endl;
    std::vector<std::string> baseline;
    for (int device = 0; device < count; ++device) {
      check(cudaSetDevice(device));
      cudaDeviceProp properties{};
      check(cudaGetDeviceProperties(&properties, device));
      std::cout << "GPU " << device << ": " << properties.name << std::endl;
      int index = 0;
      auto run_test = [&](const std::string &name, Circuit circuit,
                          const Circuit &original, const Circuit &expected) {
        std::cout << "  TEST: " << name << std::endl;
        // Destroy the optimizer before switching devices so all frees use
        // the device that owns its allocations.
        {
          CudaOptimizer optimizer;
          optimizer.setup(circuit);
          optimizer.copy_to_host(circuit);
          equal(circuit, original);
          optimizer.optimize();
          optimizer.copy_to_host(circuit);
        }
        std::ostringstream output;
        write_circuit(output, circuit);
        std::istringstream encoded(output.str());
        Circuit result = read_circuit(encoded);
        equal(result, expected);
        if (device == 0) baseline.push_back(output.str());
        else require(output.str() == baseline[index], "results differ between GPUs");
        ++index;
        std::cout << "  PASS: " << name << std::endl;
      };
      for (const auto &test : cases) {
        run_test(test.name, parse(test.input), parse(test.input), parse(test.expected));
      }
      for (int arg = 1; arg < argc; ++arg) {
        const std::filesystem::path path(argv[arg]);
        const auto expected_path = path.parent_path() / "expected" / path.filename();
        run_test(path.string(), read_file(path), read_file(path), read_file(expected_path));
      }
    }
    if (count < 2)
      std::cout << "SKIP: cross-device comparison requires at least two GPUs\n";
    std::cout << "All GPU tests passed on " << count << " device(s).\n";
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
