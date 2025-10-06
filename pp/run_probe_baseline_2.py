import ctypes
import numpy as np
import time
import os
import sys
from gpu_utils_pp import (load_cuda_libs_pp, CU_CHECK, CUDA_RT_CHECK,
                          compile_kernels_to_ptx_pp, find_pages_for_set,
                          PAGE_SIZE, PRIME_PROBE_N, L3_FILL_COUNT)

# --- Configuration ---
GPU_ID = 0
TARGET_SET_ID = 1 # Which L3 set index to prime/probe
NUM_SAMPLES = 2000 # Number of hit/miss measurements to take
ALLOC_POOL_MB = 4 * 1024 # 8 GB pool for finding pages
KERNELS_FILE = "pp_baseline_kernels.cu"
KERNELS_PTX = "pp_baseline_kernels.ptx"

def main():
    # Load libs and get local handles
    driver_ok, local_libcuda, local_libcudart = load_cuda_libs_pp()
    if not driver_ok: print("Failed loading libcuda."); exit(1)
    if not local_libcudart: print("Error: libcudart handle is None, needed for sync."); exit(1)

    cu_context = None
    cu_module = None
    pool_gpu_va_start = 0
    d_prime_probe_vas_ptr = ctypes.c_ulonglong(0)
    d_flush_vas_ptr = ctypes.c_ulonglong(0)
    d_l2_flush_addr_ptr = ctypes.c_ulonglong(0) # New: Device pointer for L2 flush address
    result_time_gpu_ptr = ctypes.c_ulonglong(0)
    result_time_gpu_va = 0

    hit_timings = []
    miss_timings = []

    try:
        # --- Init CUDA ---
        CU_CHECK(local_libcuda.cuInit(0), "Main cuInit")
        cu_device = ctypes.c_int()
        CU_CHECK(local_libcuda.cuDeviceGet(ctypes.byref(cu_device), GPU_ID), "Main cuDeviceGet")
        cu_context_ptr = ctypes.c_void_p()
        CU_CHECK(local_libcuda.cuCtxCreate_v2(ctypes.byref(cu_context_ptr), 0, cu_device), "Main cuCtxCreate")
        cu_context = cu_context_ptr
        print("CUDA Context created.")

        # --- Compile PTX Module ---
        # compile_kernels_to_ptx_pp(KERNELS_FILE, KERNELS_PTX)

        # --- Load PTX Module & Kernels ---
        cu_module_ptr = ctypes.c_void_p()
        with open(KERNELS_PTX, 'rb') as f: ptx_content = f.read()
        ptx_content_null_terminated = ctypes.create_string_buffer(ptx_content + b'\x00')
        CU_CHECK(local_libcuda.cuModuleLoadData(ctypes.byref(cu_module_ptr), ptx_content_null_terminated), "Main cuModuleLoadData")
        cu_module = cu_module_ptr
        probe_kernel_func = ctypes.c_void_p()
        flush_kernel_func = ctypes.c_void_p()
        l2_flush_kernel_func = ctypes.c_void_p() # New: Handle for L2 flush kernel
        CU_CHECK(local_libcuda.cuModuleGetFunction(ctypes.byref(probe_kernel_func), cu_module, b"probe_time_kernel"), "Main GetFunc probe")
        CU_CHECK(local_libcuda.cuModuleGetFunction(ctypes.byref(flush_kernel_func), cu_module, b"flush_kernel_pp"), "Main GetFunc flush")
        CU_CHECK(local_libcuda.cuModuleGetFunction(ctypes.byref(l2_flush_kernel_func), cu_module, b"l2_flush_kernel"), "Main GetFunc l2_flush") # New
        print("Kernels loaded.")

        # --- Allocate Memory ---
        print("Allocating memory...")
        pool_size_bytes = ALLOC_POOL_MB * 1024 * 1024
        pool_gpu_ptr = ctypes.c_ulonglong()
        flags = 1 # CU_MEM_ATTACH_GLOBAL
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(pool_gpu_ptr), pool_size_bytes, flags), "Main cuMemAllocManaged Pool")
        pool_gpu_va_start = pool_gpu_ptr.value
        if not pool_gpu_va_start: raise RuntimeError("Pool allocation failed")
        print(f"Pool GPU VA Start: {hex(pool_gpu_va_start)}")

        # Find pages for the target set to Prime/Probe
        prime_probe_vas = find_pages_for_set(local_libcuda, pool_gpu_va_start, pool_size_bytes, TARGET_SET_ID, PRIME_PROBE_N)
        if len(prime_probe_vas) < PRIME_PROBE_N: raise RuntimeError(f"Could not find enough pages for target set {TARGET_SET_ID}")

        # Find pages for the L3 flush
        flush_set_id = (TARGET_SET_ID + 5) % 256
        flush_vas_list = find_pages_for_set(local_libcuda, pool_gpu_va_start, pool_size_bytes, flush_set_id, L3_FILL_COUNT)
        if len(flush_vas_list) < L3_FILL_COUNT:
            print(f"Warning: Only found {len(flush_vas_list)} pages for flush set {flush_set_id}. Using any available pages.")
            available_pages = pool_size_bytes // PAGE_SIZE
            needed = L3_FILL_COUNT - len(flush_vas_list)
            start_idx = PRIME_PROBE_N + 5 # Avoid prime/probe pages
            for i in range(needed):
                idx = (start_idx + i) % available_pages
                if pool_gpu_va_start + idx * PAGE_SIZE not in prime_probe_vas: # Basic check to avoid overlap
                    flush_vas_list.append(pool_gpu_va_start + idx * PAGE_SIZE)
            flush_vas_list = flush_vas_list[:L3_FILL_COUNT] # Ensure not too many

        print(f"Using {len(flush_vas_list)} pages for L3 flush.")
        if len(flush_vas_list) == 0: raise RuntimeError("Cannot proceed with 0 flush pages.")

        # Allocate device memory for VA lists
        VasListTypePP = ctypes.c_ulonglong * PRIME_PROBE_N
        h_pp_vas = VasListTypePP(*prime_probe_vas)
        CU_CHECK(local_libcuda.cuMemAlloc_v2(ctypes.byref(d_prime_probe_vas_ptr), ctypes.sizeof(h_pp_vas)), "Main Alloc PP VAs")
        CU_CHECK(local_libcuda.cuMemcpyHtoD_v2(d_prime_probe_vas_ptr.value, ctypes.byref(h_pp_vas), ctypes.sizeof(h_pp_vas)), "Main Memcpy PP VAs")

        VasListTypeFlush = ctypes.c_ulonglong * len(flush_vas_list)
        h_flush_vas = VasListTypeFlush(*flush_vas_list)
        CU_CHECK(local_libcuda.cuMemAlloc_v2(ctypes.byref(d_flush_vas_ptr), ctypes.sizeof(h_flush_vas)), "Main Alloc Flush VAs")
        CU_CHECK(local_libcuda.cuMemcpyHtoD_v2(d_flush_vas_ptr.value, ctypes.byref(h_flush_vas), ctypes.sizeof(h_flush_vas)), "Main Memcpy Flush VAs")

        # Allocate a small buffer on the device for the L2 flush address
        l2_flush_address = prime_probe_vas[0] # Use the first prime/probe address
        h_l2_flush_addr = ctypes.c_ulonglong(l2_flush_address)
        CU_CHECK(local_libcuda.cuMemAlloc_v2(ctypes.byref(d_l2_flush_addr_ptr), ctypes.sizeof(h_l2_flush_addr)), "Main Alloc L2 Flush Addr")
        CU_CHECK(local_libcuda.cuMemcpyHtoD_v2(d_l2_flush_addr_ptr.value, ctypes.byref(h_l2_flush_addr), ctypes.sizeof(h_l2_flush_addr)), "Main Memcpy L2 Flush Addr")

        # Allocate result time buffer using Managed Alloc
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(result_time_gpu_ptr),
                                                 ctypes.sizeof(ctypes.c_ulonglong), flags),
                 "Main cuMemAllocManaged Result Time")
        result_time_gpu_va = result_time_gpu_ptr.value
        if not result_time_gpu_va: raise RuntimeError("Result time buffer alloc failed")
        print("Device memory setup complete.")

        # --- Prepare Kernel Args ---
        num_pp_pages_val = ctypes.c_int(PRIME_PROBE_N)
        num_flush_pages_val = ctypes.c_int(len(flush_vas_list))
        d_pp_vas_val = ctypes.c_ulonglong(d_prime_probe_vas_ptr.value)
        d_flush_vas_val = ctypes.c_ulonglong(d_flush_vas_ptr.value)
        d_l2_flush_addr_val = ctypes.c_ulonglong(d_l2_flush_addr_ptr.value) # New: Device address for L2 flush
        d_result_time_kernel_arg = ctypes.c_ulonglong(result_time_gpu_va)

        probe_args = [ ctypes.byref(d_pp_vas_val), ctypes.byref(num_pp_pages_val), ctypes.byref(d_result_time_kernel_arg) ]
        probe_params = (ctypes.c_void_p * len(probe_args))(*[ctypes.cast(p, ctypes.c_void_p) for p in probe_args])

        flush_args = [ ctypes.byref(d_flush_vas_val), ctypes.byref(num_flush_pages_val) ]
        flush_params = (ctypes.c_void_p * len(flush_args))(*[ctypes.cast(p, ctypes.c_void_p) for p in flush_args])

        l2_flush_args = [ ctypes.byref(d_l2_flush_addr_val) ] # New: Argument for L2 flush kernel
        l2_flush_params = (ctypes.c_void_p * len(l2_flush_args))(*[ctypes.cast(p, ctypes.c_void_p) for p in l2_flush_args])

        # --- Measurement Loop ---
        print("Performing initial prime...")
        CU_CHECK(local_libcuda.cuLaunchKernel(probe_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, probe_params, None), "Launch Prime")
        CU_CHECK(local_libcuda.cuStreamSynchronize(None), "Sync Prime")

        print(f"Starting {NUM_SAMPLES} Hit/Miss measurements (with L2 flush)...")
        for i in range(NUM_SAMPLES):
            # Measure Hit
            CU_CHECK(local_libcuda.cuLaunchKernel(probe_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, probe_params, None), f"Launch Probe Hit {i}")
            CU_CHECK(local_libcuda.cuStreamSynchronize(None), f"Sync Probe Hit {i}")
            result_time_ptr_host_view = ctypes.cast(result_time_gpu_va, ctypes.POINTER(ctypes.c_ulonglong))
            hit_timings.append(result_time_ptr_host_view.contents.value)

            # Perform L2 Flush
            CU_CHECK(local_libcuda.cuLaunchKernel(l2_flush_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, l2_flush_params, None), f"Launch L2 Flush {i}") # New
            CU_CHECK(local_libcuda.cuStreamSynchronize(None), f"Sync L2 Flush {i}") # New

            # Measure Miss
            CU_CHECK(local_libcuda.cuLaunchKernel(probe_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, probe_params, None), f"Launch Probe Miss {i}")
            CU_CHECK(local_libcuda.cuStreamSynchronize(None), f"Sync Probe Miss {i}")
            result_time_ptr_host_view = ctypes.cast(result_time_gpu_va, ctypes.POINTER(ctypes.c_ulonglong))
            miss_timings.append(result_time_ptr_host_view.contents.value)

            if (i + 1) % 10 == 0: print(f"  Completed {i+1}/{NUM_SAMPLES} samples...")

        # --- Analysis ---
        print("Analyzing results...")
        hit_times_np = np.array(hit_timings)
        miss_times_np = np.array(miss_timings)
        hit_times_filt = hit_times_np[hit_times_np > 100] # Basic filter
        miss_times_filt = miss_times_np[miss_times_np > 100] # Basic filter

        if len(hit_times_filt) > 0 and len(miss_times_filt) > 0:
            print("\n--- Prime+Probe HIT Latency (Cycles) ---")
            print(f"Min:    {np.min(hit_times_filt)}")
            print(f"Max:    {np.max(hit_times_filt)}")
            print(f"Mean:   {np.mean(hit_times_filt):.2f}")
            print(f"Median: {np.median(hit_times_filt)}")
            print(f"StdDev: {np.std(hit_times_filt):.2f}")

            print("\n--- Prime+Probe MISS Latency (Cycles) ---")
            print(f"Min:    {np.min(miss_times_filt)}")
            print(f"Max:    {np.max(miss_times_filt)}")
            print(f"Mean:   {np.mean(miss_times_filt):.2f}")
            print(f"Median: {np.median(miss_times_filt)}")
            print(f"StdDev: {np.std(miss_times_filt):.2f}")

            hit_median = np.median(hit_times_filt)
            miss_median = np.median(miss_times_filt)
            miss_p10 = np.percentile(miss_times_filt, 10)
            threshold_probe = hit_median + (miss_median - hit_median) * 0.5 # Midpoint

            print(f"\nSuggested Prime+Probe Threshold T_probe: {threshold_probe:.0f}")
            print(f"(Separates median {hit_median:.0f} from {miss_median:.0f})")

            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6)); plt.hist(hit_times_filt, bins=50, alpha=0.7, label=f'P+P Hit (Median: {hit_median:.0f})'); plt.hist(miss_times_filt, bins=50, alpha=0.7, label=f'P+P Miss (Median: {miss_median:.0f})'); plt.axvline(threshold_probe, color='r', linestyle='--', label=f'Threshold ({threshold_probe:.0f})'); plt.xlabel('Total Probe Phase Cycles (%clock64)'); plt.ylabel('Frequency'); plt.title('Prime+Probe Baseline Latency'); plt.legend(); plt.grid(True); plt.savefig("pp_baseline_latency.png")
                print("\nSaved P+P latency histogram to pp_baseline_latency.png")
            except ImportError: print("\nInstall matplotlib to generate plot.")
            except Exception as plot_e: print(f"\nError plotting: {plot_e}")
        else:
            print("\nError: Not enough valid P+P timing samples collected.")
            print("Hit Samples:", hit_timings[:20])
            print("Miss Samples:", miss_timings[:20])

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally: # --- Cleanup ---
        print("\nCleaning up...")
        # Use .value for device pointers, use VA for managed result buffer
        if d_prime_probe_vas_ptr.value != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_prime_probe_vas_ptr.value), "Free PP VAs") 
            except Exception: pass
        if d_flush_vas_ptr.value != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_flush_vas_ptr.value), "Free Flush VAs") 
            except Exception: pass
        if result_time_gpu_va != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(result_time_gpu_va), "Free Result Time") 
            except Exception: pass
        if pool_gpu_va_start != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(pool_gpu_va_start), "Free Pool") 
            except Exception: pass
        if cu_context and cu_context.value: 
            try: CU_CHECK(local_libcuda.cuCtxDestroy_v2(cu_context), "CtxDestroy"); print("Context destroyed.") 
            except Exception: pass
        print("Cleanup finished.")

if __name__ == "__main__":
    main()