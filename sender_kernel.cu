#include "common.h"

__global__ void sender_kernel(volatile char** seq0, volatile char** seq1, int n_pages,
    int bit_to_send, unsigned long long delay_cycles,
    volatile int* stop_flag)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid != 0) return;

volatile char** target_seq = (bit_to_send == 0) ? seq0 : seq1;
unsigned long long start_time, current_time;
// int page_idx = 0; // Removed as requested by warning

while (!(*stop_flag)) {
for (int i = 0; i < n_pages; ++i) {
// Read the char into a larger register-sized variable
unsigned int temp_val = (unsigned int)target_seq[i][0]; // Read the byte

// Use the larger variable in the asm volatile to satisfy 'r' constraint
// The "+r" indicates read/write, ensuring the compiler knows temp_val is used.
// The memory clobber "memory" ensures the initial read isn't moved arbitrarily.
asm volatile("" : "+r"(temp_val) :: "memory");

// We don't actually need to write back temp_val for this purpose.
// The goal was just to make the read of target_seq[i][0] happen
// and not be optimized away.
}

if (delay_cycles > 0) {
start_time = get_time();
do {
current_time = get_time();
} while (current_time - start_time < delay_cycles);
}
}
}