#include <cuda_runtime.h>
#include <stdint.h> // For uint64_t

// Timing function using clock64
__device__ __forceinline__ unsigned long long get_time() {
    unsigned long long time;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(time));
    return time;
}

// <<<--- Add extern "C" wrapper --- >>>
extern "C" {

/*
 * timing_kernel: Measures access time to a target VA after perturbing L1/L2 TLBs.
 */
__global__ void timing_kernel(uint64_t target_page_va, uint64_t *l1_l2_evict_pages_va,
                              int num_l1_l2_evict, uint64_t *results_buffer, int buffer_offset,
                              int num_samples) {
    // Use only one thread for consistent timing measurement
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    volatile char *target_ptr = (volatile char *)target_page_va;
    // Use unsigned int for the dummy variable to match asm constraint type if needed
    // Though simple read might not need explicit asm here if volatile works.
    // Let's keep it simple first.
    volatile char dummy_val;
    unsigned long long time_start, time_end;

    for (int i = 0; i < num_samples; ++i) {
        // 1. Perturb L1/L2 TLB
        for (int j = 0; j < num_l1_l2_evict; ++j) {
            dummy_val = *((volatile char *)l1_l2_evict_pages_va[j]);
        }

        // 2. Time the access to the target page (Modified: Inner-page chase)
        volatile uint64_t *chase_ptr = (volatile uint64_t *)target_page_va;
        int chase_length = 8; // Chase a few pointers within the page

        time_start = get_time();
        for (int k = 0; k < chase_length; ++k) {
            // Ensure pointer stays within the first cache line or so to minimize cache effects
            // This assumes the first 8*8=64 bytes are set up for pointer chasing.
            // Requires initialization step in run_timing.py.
            chase_ptr = (volatile uint64_t *)(*chase_ptr);
        }
        time_end = get_time();

        // Read the final value to ensure the loop isn't optimized out
        dummy_val = *((volatile char *)chase_ptr); // Read dependent value

        // 3. Store the result
        results_buffer[buffer_offset + i] = time_end - time_start;
    }
}

/*
 * flush_kernel: Accesses a large number of pages sequentially to pollute L3 TLB.
 */
__global__ void flush_kernel(uint64_t *l3_fill_pages_va, int num_l3_fill) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    volatile char dummy_val;
    for (int i = 0; i < num_l3_fill; ++i) {
        dummy_val = *((volatile char *)l3_fill_pages_va[i]);
    }
}

// <<<--- End extern "C" wrapper --- >>>
} // extern "C"