#include <cuda.h>
#include <iostream>
#define MATRIX_SIZE 4096

#define CHECK(call)                                                                                \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "                   \
                      << cudaGetErrorString(err) << "\n";                                          \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

// Simulate workload kernel
__global__ void matrix_workload_kernel(float *A, float *B, float *C, float *D, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        // Add matrix D to the result
        C[row * N + col] = sum + D[row * N + col];
    }
}

// Memory latency probing kernel (pointer chasing)
__global__ void memory_latency_kernel(int *ptr_chain, unsigned long long *latencies,
                                      int iterations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) { // only one thread measures latency
        unsigned long long start, end;
        volatile int *ptr = ptr_chain;
        int index = 0;

        for (int i = 0; i < iterations; ++i) {
            start = clock64();
            index = ptr[index]; // pointer chasing load
            end = clock64();
            latencies[i] = end - start;
        }
    }
}

int main() {
    const int iterations = 1000;
    int *d_ptr_chain;
    unsigned long long *d_latencies, *h_latencies;

    const int N = MATRIX_SIZE;
    size_t bytes = N * N * sizeof(float);

    float *d_A, *d_B, *d_C, *d_D;
    CHECK(cudaMalloc(&d_A, bytes));
    CHECK(cudaMalloc(&d_B, bytes));
    CHECK(cudaMalloc(&d_C, bytes));
    CHECK(cudaMalloc(&d_D, bytes));

    CHECK(cudaMemset(d_A, 1, bytes));
    CHECK(cudaMemset(d_B, 2, bytes));
    CHECK(cudaMemset(d_D, 3, bytes));

    // Launch matrix workload kernel on stream1
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    // Create a pointer chasing structure in GPU memory
    int *h_ptr_chain = new int[iterations];
    for (int i = 0; i < iterations - 1; ++i)
        h_ptr_chain[i] = i + 1;
    h_ptr_chain[iterations - 1] = 0; // loop back

    CHECK(cudaMalloc(&d_ptr_chain, iterations * sizeof(int)));
    CHECK(cudaMemcpy(d_ptr_chain, h_ptr_chain, iterations * sizeof(int), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&d_latencies, iterations * sizeof(unsigned long long)));
    h_latencies = new unsigned long long[iterations];

    // Create streams
    cudaStream_t stream1, stream2;
    CHECK(cudaStreamCreate(&stream1));
    CHECK(cudaStreamCreate(&stream2));

    // Launch workload kernel on stream1
    matrix_workload_kernel<<<blocks, threads, 0, stream1>>>(d_A, d_B, d_C, d_D, N);

    // Launch memory latency probing kernel on stream2
    memory_latency_kernel<<<1, 1, 0, stream2>>>(d_ptr_chain, d_latencies, iterations);

    CHECK(cudaStreamSynchronize(stream1));
    CHECK(cudaStreamSynchronize(stream2));

    CHECK(cudaMemcpy(h_latencies, d_latencies, iterations * sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost));

    for (int i = 0; i < 20; ++i) {
        std::cout << "Latency [" << i << "] = " << h_latencies[i] << " cycles\n";
    }

    // Cleanup
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_D));
    CHECK(cudaFree(d_ptr_chain));
    CHECK(cudaFree(d_latencies));
    delete[] h_ptr_chain;
    delete[] h_latencies;
    CHECK(cudaStreamDestroy(stream1));
    CHECK(cudaStreamDestroy(stream2));

    return 0;
}
