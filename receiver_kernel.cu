#include "common.h"

__global__ void receiver_kernel(volatile char **seq0, volatile char **seq1, int n_pages,
                                unsigned long long *results, int max_samples,
                                volatile int *stop_flag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0)
        return;

    unsigned long long start_time, end_time, t0, t1;
    int sample_count = 0;
    int page_idx = 0;

    while (!(*stop_flag) && sample_count < max_samples * 2) {
        page_idx = (page_idx + 1) % n_pages;

        // Measure time for seq0 access
        start_time = get_time();
        // Read char into larger temp variable
        unsigned int temp_val0 = (unsigned int)seq0[page_idx][0];
        // Use temp variable in asm
        asm volatile("" : "+r"(temp_val0)::"memory");
        end_time = get_time();
        t0 = end_time - start_time;

        // Measure time for seq1 access
        start_time = get_time();
        // Read char into larger temp variable
        unsigned int temp_val1 = (unsigned int)seq1[page_idx][0];
        // Use temp variable in asm
        asm volatile("" : "+r"(temp_val1)::"memory");
        end_time = get_time();
        t1 = end_time - start_time;

        if (sample_count < max_samples * 2) {
            results[sample_count++] = t0;
            results[sample_count++] = t1;
        }
    }
}