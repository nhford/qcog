#include "circuit.h"
#include "qasm.h"
#include "optimizer.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char *argv[]) {
  if (argc != 3 || std::string(argv[1]) != "-f") {
    std::cerr << "Usage: " << argv[0] << " -f input.qasm > output.qasm\n";
    return 1;
  }

  try {
    std::ifstream input(argv[2]);
    if (!input)
      throw std::runtime_error(std::string("Unable to open ") + argv[2]);
    Circuit circuit = read_circuit(input);
    CudaOptimizer optimizer;
    optimizer.setup(circuit);
    optimizer.optimize();
    optimizer.copy_to_host(circuit);
    write_circuit(std::cout, circuit);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
