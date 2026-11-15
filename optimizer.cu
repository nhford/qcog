#include "circuit.h"
#include <cuda_runtime.h>
#include <stdexcept>

static void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

void optimize(Circuit &circuit) {
  if (circuit.num_gates == 0)
    return;

  unsigned char *device;
  const auto bytes = circuit.bytes();
  check(cudaMalloc(&device, bytes));
  const auto upload =
      cudaMemcpy(device, circuit.storage.get(), bytes, cudaMemcpyHostToDevice);
  // Launch optimization kernels here using the device buffer and gate count.
  const auto download = upload == cudaSuccess
      ? cudaMemcpy(circuit.storage.get(), device, bytes, cudaMemcpyDeviceToHost)
      : cudaSuccess;
  const auto release = cudaFree(device);
  check(upload);
  check(download);
  check(release);
}
