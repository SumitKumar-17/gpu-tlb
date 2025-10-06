#include <cuda.h>
#include <stdio.h>


#define WAIT_TIME   2000000000L // about 1 second

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    printf("CUDA ERROR: %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__global__ void
test_time(uint64_t *page)
{
  for (int i = 0; i < 5; i++) {
    printf("Timer loop\n");
    uint64_t clk0 = clock64();
    uint64_t clk1 = 0;
    while (clk1 < WAIT_TIME)
      clk1 = clock64() - clk0;
  }
}

int main() {
  printf("Ready\n");
  test_time<<<1, 1>>>(0);
      checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Synchronize to flush printf() output
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");
  printf("Done\n");
  return 0;
}