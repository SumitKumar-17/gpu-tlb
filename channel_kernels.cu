#include <cuda_runtime.h>
#include <stdint.h> // For uint64_t

// Basic timing function
__device__ __forceinline__ unsigned long long get_time() {
    unsigned long long time;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(time));
    return time;
}

extern "C" { // Prevent C++ name mangling

/*
 * sender_contention_kernel (Revised): Uses list of target VAs
 * page_vas:      GPU VA of device buffer containing actual target page VAs.
 * num_pages:     Number of VAs in the buffer.
 * stop_flag:     Pointer to stop flag in device memory.
 */
__global__ void sender_contention_kernel(uint64_t *page_vas, // Pointer to VAs in device mem
                                         int num_pages, int *stop_flag) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    volatile char dummy_val;
    int current_idx = 0; // Index within the page_vas array

    // Loop until stop flag is set
    while (*stop_flag == 0) {
        // Access the page specified by the current index
        uint64_t page_addr = page_vas[current_idx]; // Get VA directly
        // Volatile read from the page to ensure TLB access
        dummy_val = *((volatile char *)page_addr);

        // Move to next index
        current_idx = (current_idx + 1) % num_pages;

        // Yield might prevent busy-waiting too aggressively
        // asm volatile("yield;");
    }
}

/*
 * receiver_probe_kernel (Revised): Uses lists of target VAs
 * page_vas0:         GPU VA of device buffer containing page VAs for Set 0.
 * page_vas1:         GPU VA of device buffer containing page VAs for Set 1.
 * num_pages_per_set: Number of VAs per set (N_PAGES).
 * results_buffer_va: GPU VA of managed buffer for timing pairs (t0, t1).
 * max_samples:       Max number of timing PAIRS to collect.
 */
__global__ void receiver_probe_kernel(uint64_t *page_vas0,   // Pointer to VAs
                                      uint64_t *page_vas1,   // Pointer to VAs
                                      int num_pages_per_set, // N
                                      uint64_t *results_buffer_va, int max_samples) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    int current_page_idx = 0; // Index within the set (0 to N-1)
    int sample_count = 0;     // Number of pairs collected
    unsigned long long time_start, time_end, t0, t1;
    int chase_length = 8;         // Match chase length used in baseline init
    volatile char final_read_val; // To ensure chase isn't optimized out

    while (sample_count < max_samples) {
        // Get VAs for the current pages in Set 0 and Set 1
        uint64_t page0_va = page_vas0[current_page_idx]; // Get VA directly
        uint64_t page1_va = page_vas1[current_page_idx]; // Get VA directly

        // --- Time Set 1 Access FIRST ---
        volatile uint64_t *chase_ptr1 = (volatile uint64_t *)page1_va;
        time_start = get_time();
#pragma unroll
        for (int k = 0; k < chase_length; ++k) {
            chase_ptr1 = (volatile uint64_t *)(*chase_ptr1);
        }
        time_end = get_time();
        final_read_val = *((volatile char *)chase_ptr1); // Dependent read
        t1 = time_end - time_start;                      // Store T1

        // --- Time Set 0 Access SECOND ---
        volatile uint64_t *chase_ptr0 = (volatile uint64_t *)page0_va;
        time_start = get_time();
#pragma unroll
        for (int k = 0; k < chase_length; ++k) {
            chase_ptr0 = (volatile uint64_t *)(*chase_ptr0);
        }
        time_end = get_time();
        final_read_val = *((volatile char *)chase_ptr0); // Dependent read
        t0 = time_end - time_start;                      // Store T0

        // Store results (t0, t1 pair) - check bounds carefully
        int results_base_idx = sample_count * 2;
        // Check if pointer is valid before writing (basic check)
        if (results_buffer_va != NULL && (results_base_idx + 1) < (max_samples * 2)) {
            results_buffer_va[results_base_idx + 0] = t0;
            results_buffer_va[results_base_idx + 1] = t1;
            sample_count++;
        } else {
            break; // Buffer likely full or invalid
        }

        current_page_idx = (current_page_idx + 1) % num_pages_per_set;
    }
}

} // extern "C"