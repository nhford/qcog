#include "circuit.h"
#include <cuda_runtime.h>
#include <stdexcept>

static void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

void optimize(Circuit &circuit) {
  Circuit *device;
  check(cudaMalloc(&device, sizeof(Circuit)));
  const auto upload =
      cudaMemcpy(device, &circuit, sizeof(Circuit), cudaMemcpyHostToDevice);
  // Launch optimization kernels here once they are implemented.
  const auto download = upload == cudaSuccess
      ? cudaMemcpy(&circuit, device, sizeof(Circuit), cudaMemcpyDeviceToHost)
      : cudaSuccess;
  const auto release = cudaFree(device);
  check(upload);
  check(download);
  check(release);
}
