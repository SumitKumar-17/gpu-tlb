# Filename: measure_tlb_timing_range.py
import subprocess
import re
import os
import tempfile
import sys
import time

# --- Configuration ---
NVCC_PATH = "nvcc"
GPU_ARCH = "sm_86"  # Compute Capability for RTX 3060
PAGE_SIZE_BYTES = 4096
REPETITIONS = 10000 # Pointer-chasing loops inside the kernel
STRIDE = 1          # Stride in pages (1 means consecutive pages)

# --- Range of Page Counts to Test ---
PAGE_COUNTS_TO_TEST = list(range(2, 33)) + \
                      [36, 40, 48, 56, 60] + \
                      list(range(64, 129, 4)) + \
                      list(range(128, 257, 8)) + \
                      list(range(256, 513, 16))
# Example: list(range(2, 1025, 2)) # Test more points if needed

MEASUREMENT_DELAY = 0.1 # Optional delay

# --- CUDA Kernel Template (Timing moved to host) ---
CUDA_KERNEL_TEMPLATE = """
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdint.h> // For uintptr_t

// Kernel parameters (defines)
#define Numpages {NUM_PAGES}
#define Rep {REPETITIONS}
// #define Page_size {PAGE_SIZE_BYTES} // Defined in host code now
// #define Stride {STRIDE} // Defined in host code now

// Kernel: Performs the pointer chasing only
__global__ void PtrChase(uintptr_t *start_ptr)
{{ // Escaped C++ brace
    uintptr_t *ptr = start_ptr;
    uintptr_t target = 0;

    // The actual pointer chasing loop
    for (int j = 0; j < Rep; ++j) {{ // Escaped C++ brace
        for (int i = 0; i < Numpages; ++i) {{ // Escaped C++ brace
            // Follow the pointer chain
            ptr = (uintptr_t *)(*ptr);
        }} // Escaped C++ brace
        target = (uintptr_t)ptr; // Use the final ptr value
    }} // Escaped C++ brace

    // Use volatile write to prevent loop optimization (read is implicit above)
     // Ensure target is used somehow, maybe write to global memory if needed,
     // but often the pointer chain dependency is enough for modern compilers.
     // A simple volatile operation helps ensure it.
    volatile uintptr_t temp = target;
    if (temp == 0xBADBADBAD) {{}} // Dummy check to ensure temp is "used"
}} // Escaped C++ brace

// Host code: Sets up memory, launches kernel, handles timing
int main()
{{ // Escaped C++ brace
    // Get parameters from defines
    int num_pages_val = Numpages;
    int repetitions_val = Rep;
    int page_size_val = {PAGE_SIZE_BYTES}; // Use literal value passed by Python
    int stride_val = {STRIDE}; // Use literal value passed by Python

    size_t element_size = sizeof(uintptr_t);
    // Calculate total memory needed
    size_t total_pages_in_chain = (size_t)num_pages_val;
    size_t allocation_pages = total_pages_in_chain * stride_val;
    size_t range_bytes = allocation_pages * page_size_val;

    uintptr_t *h_mem = NULL; // Host pointer (for setup)
    uintptr_t *d_mem = NULL; // Device pointer (passed to kernel)
    float h_time_elapsed = 0.0f; // Host variable for time result

    printf("Config: Pages=%d, Reps=%d, Stride=%d, PageSize=%d B, ElementSize=%zu B, Range=%zu B\\n",
           num_pages_val, repetitions_val, stride_val, page_size_val, element_size, range_bytes);

    // Allocate managed memory
    cudaError_t mem_stat = cudaMallocManaged(&d_mem, range_bytes, cudaMemAttachGlobal);
    if (mem_stat != cudaSuccess) {{ // Escaped C++ brace
        fprintf(stderr, "Failed to allocate managed memory (size %zu B): %s\\n", range_bytes, cudaGetErrorString(mem_stat));
        return 1;
    }} // Escaped C++ brace

    h_mem = d_mem; // Host can access managed memory directly

    // Setup pointer chain
    for (size_t i = 0; i < total_pages_in_chain; ++i) {{ // Escaped C++ brace
        size_t current_page_idx = i * stride_val;
        size_t next_page_idx = ((i + 1) % total_pages_in_chain) * stride_val;

        size_t current_byte_offset = current_page_idx * page_size_val;
        size_t next_byte_offset = next_page_idx * page_size_val;

        size_t current_element_index = current_byte_offset / element_size;
        uintptr_t next_page_start_addr = (uintptr_t)&d_mem[next_byte_offset / element_size];

        if (current_byte_offset + element_size > range_bytes || next_byte_offset >= range_bytes) {{ // Escaped C++ brace
             fprintf(stderr, "Error: Memory offset calculation out of bounds during pointer setup.\\n");
             cudaFree(d_mem); return 1;
        }} // Escaped C++ brace
        h_mem[current_element_index] = next_page_start_addr;
    }} // Escaped C++ brace


    // --- Timing Setup ---
    cudaEvent_t start, stop;
    cudaError_t event_stat1 = cudaEventCreate(&start);
    cudaError_t event_stat2 = cudaEventCreate(&stop);
     if (event_stat1 != cudaSuccess || event_stat2 != cudaSuccess) {{ // Escaped C++ brace
        fprintf(stderr, "Failed to create CUDA events: %s / %s\\n", cudaGetErrorString(event_stat1), cudaGetErrorString(event_stat2));
        cudaFree(d_mem); return 1;
    }} // Escaped C++ brace


    // --- Kernel Execution and Timing ---
    cudaDeviceSynchronize(); // Ensure setup is complete before starting timer
    cudaError_t start_stat = cudaEventRecord(start, 0); // Record start event

    // Launch kernel (no timing parameter needed)
    PtrChase<<<1, 1>>>(d_mem);

    cudaError_t stop_stat = cudaEventRecord(stop, 0); // Record stop event
    cudaError_t kernel_stat = cudaGetLastError(); // Check for launch errors AFTER recording stop event

    // Check for errors during recording or launch
    if (start_stat != cudaSuccess || stop_stat != cudaSuccess || kernel_stat != cudaSuccess) {{ // Escaped C++ brace
        fprintf(stderr, "Error during kernel execution/event recording: Launch(%s), Start(%s), Stop(%s)\\n",
                cudaGetErrorString(kernel_stat), cudaGetErrorString(start_stat), cudaGetErrorString(stop_stat));
        cudaEventDestroy(start); cudaEventDestroy(stop); cudaFree(d_mem); return 1;
    }} // Escaped C++ brace


    // Synchronize host thread with the stop event
    cudaError_t sync_stat = cudaEventSynchronize(stop);
     if (sync_stat != cudaSuccess) {{ // Escaped C++ brace
        fprintf(stderr, "Failed to synchronize on stop event: %s\\n", cudaGetErrorString(sync_stat));
        // Continue to cleanup, but timing might be invalid
    }} // Escaped C++ brace


    // Calculate elapsed time
    cudaError_t elapsed_stat = cudaEventElapsedTime(&h_time_elapsed, start, stop);
     if (elapsed_stat != cudaSuccess) {{ // Escaped C++ brace
        fprintf(stderr, "Failed to get elapsed time: %s\\n", cudaGetErrorString(elapsed_stat));
        // Continue to cleanup, timing is invalid
        h_time_elapsed = -1.0f; // Indicate error
    }} // Escaped C++ brace

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // --- Report Results ---
    if (h_time_elapsed >= 0.0f) {{ // Escaped C++ brace
        double total_accesses = (double)repetitions_val * num_pages_val;
        double avg_time_ns = (total_accesses > 0) ? (h_time_elapsed * 1e6) / total_accesses : 0.0;

        printf("Total time: %f ms\\n", h_time_elapsed);
        printf("Total accesses: %.0f\\n", total_accesses);
        printf("Average time per access (ns): %f ns\\n", avg_time_ns);
        printf("RESULT_PAGES=%d\\n", num_pages_val);
        printf("RESULT_TIME_NS=%f\\n", avg_time_ns);
    }} else {{ // Escaped C++ brace
         printf("Timing measurement failed.\\n");
         // Still print pages, but indicate failed time
         printf("RESULT_PAGES=%d\\n", num_pages_val);
         printf("RESULT_TIME_NS=-1.0\\n"); // Use -1 to signify failure
    }} // Escaped C++ brace


    // Cleanup memory
    cudaFree(d_mem);

    return 0;
}} // Escaped C++ brace
"""

def run_timing_experiment(num_pages, stride, repetitions, page_size, gpu_arch):
    """Compiles and runs the CUDA kernel for given parameters."""
    print(f"--- Running Experiment: {num_pages} pages, Stride {stride} ---")
    try:
        # Pass fixed values into template for PageSize and Stride now
        source_code = CUDA_KERNEL_TEMPLATE.format(
            NUM_PAGES=num_pages,
            REPETITIONS=repetitions,
            PAGE_SIZE_BYTES=page_size, # Pass as literal
            STRIDE=stride              # Pass as literal
        )
    except Exception as e:
        print(f"Error formatting template: {e}")
        return None

    tmp_dir = tempfile.mkdtemp()
    # Ensure unique filenames if running in parallel someday
    cu_filename = os.path.join(tmp_dir, f"kernel_{num_pages}.cu")
    exe_filename = os.path.join(tmp_dir, f"kernel_{num_pages}_exe")

    try:
        with open(cu_filename, "w") as f:
            f.write(source_code)

        compile_cmd = [NVCC_PATH, cu_filename, "-o", exe_filename, f"-arch={gpu_arch}", "--default-stream", "per-thread", "-O3"]
        print(f"Compiling: {' '.join(compile_cmd)}")
        compile_proc = subprocess.run(compile_cmd, capture_output=True, text=True, check=True, timeout=90)
        # if compile_proc.stdout: print("NVCC Output:", compile_proc.stdout) # Uncomment for debug
        if compile_proc.stderr: print("NVCC Warning/Error:", compile_proc.stderr) # Show warnings/errors

        print(f"Running: {exe_filename}")
        # Increased timeout for potentially longer runs with more pages
        run_proc = subprocess.run([exe_filename], capture_output=True, text=True, check=True, timeout=180)
        if run_proc.stdout: print("Executable Output:\n", run_proc.stdout) # Show output
        if run_proc.stderr: print("Executable Error:", run_proc.stderr) # Show errors


        # Parse the result time
        match_time = re.search(r"RESULT_TIME_NS=([0-9.-]+)", run_proc.stdout) # Allow negative for errors
        match_pages = re.search(r"RESULT_PAGES=([0-9]+)", run_proc.stdout)

        if match_time and match_pages:
            pages = int(match_pages.group(1))
            avg_time_ns = float(match_time.group(1))
            if pages == num_pages: # Sanity check
                 # Check if C++ code reported an error (-1.0)
                 if avg_time_ns < 0:
                      print("Timing reported as invalid by C++ code.")
                      return None
                 else:
                      print(f"Parsed: Pages={pages}, Average time per access: {avg_time_ns:.4f} ns")
                      return avg_time_ns
            else:
                 print(f"Error: Parsed page count ({pages}) != expected ({num_pages}).")
                 return None
        else:
            print("Error: Could not parse RESULT_PAGES or RESULT_TIME_NS from output.")
            print("Full Output:\n---\n", run_proc.stdout, "\n---")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error during {'compilation' if NVCC_PATH in e.cmd else 'execution'}:")
        print("Command:", ' '.join(e.cmd))
        print("Return Code:", e.returncode)
        # Ensure output/error streams are decoded if captured as bytes (text=True should handle this)
        output = e.stdout if isinstance(e.stdout, str) else e.stdout.decode(errors='ignore') if e.stdout else ""
        error = e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors='ignore') if e.stderr else ""
        print("Output:", output)
        print("Error:", error)
        return None
    except subprocess.TimeoutExpired as e:
        print(f"Error: {'Compilation' if NVCC_PATH in e.cmd else 'Execution'} timed out after {e.timeout} seconds.")
        print("Command:", ' '.join(e.cmd))
        return None
    except FileNotFoundError:
        print(f"Error: '{NVCC_PATH}' command not found. Is CUDA Toolkit installed and in PATH?")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred in run_timing_experiment: {type(e).__name__} - {e}")
        # Optionally print traceback for debugging
        # import traceback
        # traceback.print_exc()
        return None
    finally:
        # Clean up
        try:
            if os.path.exists(cu_filename): os.remove(cu_filename)
            if os.path.exists(exe_filename): os.remove(exe_filename)
            if os.path.exists(tmp_dir): os.rmdir(tmp_dir)
        except OSError as e:
            print(f"Warning: Could not clean up temporary file/directory: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting TLB Timing Measurement across range of page counts...")
    print(f"Testing GPU Architecture: {GPU_ARCH}")
    print(f"Page counts to test: {PAGE_COUNTS_TO_TEST}")

    results = {} # Dictionary to store {page_count: time_ns}

    for page_count in PAGE_COUNTS_TO_TEST:
        # Run multiple times and average? For now, run once per point.
        avg_time = run_timing_experiment(
            num_pages=page_count,
            stride=STRIDE,
            repetitions=REPETITIONS,
            page_size=PAGE_SIZE_BYTES,
            gpu_arch=GPU_ARCH
        )

        if avg_time is not None:
            results[page_count] = avg_time
        else:
            print(f"*** Failed to get timing for {page_count} pages. ***")
            results[page_count] = -1.0 # Mark failure clearly

        # Optional delay
        if MEASUREMENT_DELAY > 0:
            time.sleep(MEASUREMENT_DELAY)


    # --- Report Results ---
    print("\n--- Timing Results (Pages vs. Avg Access Time ns) ---")
    print("Pages\tTime_ns")
    for pages in sorted(results.keys()):
        # Use a different format specifier for potentially negative error values
        if results[pages] < 0:
            print(f"{pages}\tFAILED")
        else:
            print(f"{pages}\t{results[pages]:.4f}")

    print("\nExperiment Complete. Plot 'Pages' vs 'Time_ns' data.")
    print("Look for sharp increases in Time_ns around powers of 2 (e.g., 32, 64, 128, ...) which may indicate TLB capacity.")