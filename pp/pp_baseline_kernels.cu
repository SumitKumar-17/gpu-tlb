#include <cuda_runtime.h>
#include <stdint.h>
#define MAX_BLOCKS 1024 // Maximum number of blocks supported
// Timing function
__device__ __forceinline__ unsigned long long get_time() {
    unsigned long long time;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(time));
    return time;
}
#define MAX_THREADS_PER_BLOCK 256 // Maximum threads per block supported

// --- Helper Function ---
__device__ unsigned long long get_global_thread_id() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

extern "C" {

/*
 * probe_time_kernel (Includes L1/L2 Flush): Accesses L1/L2 eviction set before probing.
 */
__global__ void probe_time_kernel(uint64_t* page_vas, // Pointer to VAs for P+P set
                                  int num_pages,
                                  uint64_t* l1_l2_evict_vas, // Pointer to VAs for L1/L2 Eviction
                                  int num_l1_l2_evict,     // Count for L1/L2 Eviction
                                  uint64_t* result_time)   // Single output value
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    volatile char dummy_val;
    unsigned long long time_start, time_end;

    // --- L1/L2 TLB/Cache Flush ---
    for (int i = 0; i < num_l1_l2_evict; ++i) {
        uint64_t evict_addr = l1_l2_evict_vas[i];
        dummy_val = *((volatile char*)evict_addr);
    }

    // --- Timed Probe ---
    time_start = get_time();
    for (int i = 0; i < num_pages; ++i) {
        uint64_t page_addr = page_vas[i];
        dummy_val = *((volatile char*)page_addr);
    }
    time_end = get_time();

    // Store the total time
    *result_time = time_end - time_start;
}

__global__ void probe_time_kernel2(unsigned long long* target_vas, int num_pages,
    unsigned long long* l1l2_evict_vas, int num_l1l2_pages,
    unsigned long long* result_time, unsigned long long* thread_timing_buffer) {
unsigned long long start, end;
int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

// L1/L2 eviction
for (int i = 0; i < num_l1l2_pages; ++i) {
    volatile char* p = (char*)l1l2_evict_vas[i];
    for (int j = 0; j < 16; ++j) {
        p[j * 64];
    }
}

start = clock64();
for (int i = 0; i < num_pages; ++i) {
    volatile char* p = (char*)target_vas[i];
    for (int j = 0; j < 16; ++j) {  // Increased stride for more accurate measurement
        p[j * 64];
    }
}
end = clock64();

// Store timing in both result_time (for backward compatibility) and thread_timing_buffer
*result_time = end - start;
if (thread_idx < 256 * 2048) {  // Ensure we don't exceed buffer size
    thread_timing_buffer[thread_idx] = end - start;
}
}

__global__ void matrix_mul_kernel(float *A, float *B, float *C, int N, unsigned long long *thread_timing_buffer, int timing_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long start_time = clock64();
    if (idx < N && idy < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[idy * N + k] * B[k * N + idx];
        }
        C[idy * N + idx] = sum;
    }
    unsigned long long end_time = clock64();
    int thread_id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x + threadIdx.x);
    if (thread_id < MAX_THREADS_PER_BLOCK * MAX_BLOCKS) {
        thread_timing_buffer[timing_offset + thread_id] = end_time - start_time;
    }
}

__global__ void matrix_add_kernel(float *C, float *D, float *E, int elements, unsigned long long *thread_timing_buffer, int timing_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long start_time = clock64();
    if (idx < elements) {
        E[idx] = C[idx] + D[idx];
    }
    unsigned long long end_time = clock64();
    if (idx < MAX_THREADS_PER_BLOCK * MAX_BLOCKS) {
        thread_timing_buffer[timing_offset + idx] = end_time - start_time;
    }
}




/*
 * flush_kernel_pp: Accesses a large number of pages sequentially to pollute L3 TLB.
 */
__global__ void flush_kernel_pp(uint64_t* l3_fill_pages_va, int num_l3_fill)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    volatile char dummy_val;
    for (int i = 0; i < num_l3_fill; ++i) {
        dummy_val = *((volatile char*)l3_fill_pages_va[i]);
    }
}

/*
 * sender_contention_kernel_pp: Renamed sender kernel, placed here for consolidation.
 *                              Uses list of target VAs. Runs indefinitely.
 * page_vas:      GPU VA of device buffer containing actual target page VAs.
 * num_pages:     Number of VAs in the buffer.
 */
__global__ void sender_contention_kernel_pp(uint64_t* page_vas, // Pointer to VAs in device mem
                                            int num_pages)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    volatile char dummy_val;
    int current_idx = 0;
    while (true) {
        uint64_t page_addr = page_vas[current_idx];
        dummy_val = *((volatile char*)page_addr);
        current_idx = (current_idx + 1) % num_pages;
        // asm volatile("yield;"); // Optional
    }
}


// // Simple Matrix Multiplication Kernel: C = A * B
// __global__ void matrix_mul_kernel(const float* A, const float* B, float* C, int N) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < N && col < N) {
//         float sum = 0.0f;
//         for (int k = 0; k < N; ++k) {
//             sum += A[row * N + k] * B[k * N + col];
//         }
//         C[row * N + col] = sum;
//     }
// }

// // Simple Matrix Addition Kernel: E = C_in + D
// // (Assumes matrices are flattened in row-major order)
// __global__ void matrix_add_kernel(const float* C_in, const float* D, float* E, int num_elements) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < num_elements) {
//         E[idx] = C_in[idx] + D[idx];
//     }
// }


} // extern "C"