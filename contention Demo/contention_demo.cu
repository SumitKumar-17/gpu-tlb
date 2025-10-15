#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

// A simple kernel representing a "victim" process doing some work.
// It just performs a few calculations.
__global__ void simple_work_kernel(float* result) {
    int idx = threadIdx.x;
    float val = 0.0f;
    for (int i = 0; i < 500; ++i) {
        val += sinf(idx + i);
    }
    result[idx] = val;
}

// An aggressive kernel designed to create contention.
// It thrashes the memory controller and L2 cache by accessing a large array.
__global__ void contention_kernel(int* large_buffer, int buffer_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread repeatedly accesses memory, causing high traffic
    // on the memory bus and evicting other data from the L2 cache.
    for (int i = 0; i < 2000; ++i) {
        idx = (idx + stride) % buffer_size;
        large_buffer[idx] *= (i % 2 == 0) ? 1 : -1; // Read and write
    }
}

int main() {
    // --- SETUP ---
    
    // Contention buffer setup (much larger than a typical L2 cache)
    const int L2_CACHE_THRASHER_SIZE = 32 * 1024 * 1024; // 32 MB
    int* d_large_buffer;
    cudaMalloc(&d_large_buffer, L2_CACHE_THRASHER_SIZE);

    // Victim kernel setup
    const int VICTIM_DATA_SIZE = 256;
    float* d_result;
    cudaMalloc(&d_result, VICTIM_DATA_SIZE * sizeof(float));

    // CUDA events for accurate timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time_baseline_ms = 0.0f;
    float time_contention_ms = 0.0f;

    // --- 1. BASELINE TIMING (NO CONTENTION) ---
    printf("Running baseline test (no contention)...\n");

    cudaEventRecord(start);
    simple_work_kernel<<<1, 256>>>(d_result);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_baseline_ms, start, stop);

    printf("Baseline execution time: %.4f ms\n\n", time_baseline_ms);

    // --- 2. CONTENTION TIMING ---
    printf("Running contention test...\n");

    cudaEventRecord(start);
    // Launch the aggressive kernel to create memory traffic
    contention_kernel<<<128, 256>>>(d_large_buffer, L2_CACHE_THRASHER_SIZE / sizeof(int));
    // Immediately launch the simple kernel. It will have to compete for resources.
    simple_work_kernel<<<1, 256>>>(d_result);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_contention_ms, start, stop);

    printf("Execution time with contention: %.4f ms\n\n", time_contention_ms);

    // --- RESULTS ---
    printf("--- Results ---\n");
    float slowdown = time_contention_ms / time_baseline_ms;
    printf("The simple kernel was slowed down by a factor of %.2fx due to resource contention.\n", slowdown);

    // --- CLEANUP ---
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_large_buffer);
    cudaFree(d_result);

    return 0;
}