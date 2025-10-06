#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <stdio.h>

// Simple timing function using clock64
__device__ __forceinline__ unsigned long long get_time() {
    unsigned long long time;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(time));
    return time;
}

// CUDA error checking macro
#define CUDA_CHECK(err) { \
    cudaError_t result = (err); \
    if (result != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel configuration (adjust if needed)
#define THREADS_PER_BLOCK 256
#define BLOCKS 1 // Keep simple for this simulation

#endif // COMMON_H