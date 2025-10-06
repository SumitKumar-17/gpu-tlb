import ctypes
import numpy as np
import time
import os
import sys
# Import functions from gpu_utils
from gpu_utils import (load_cuda_libs, CU_CHECK, CUDA_RT_CHECK, compile_kernels_to_ptx,
                       find_page_indices_in_buffer, initialize_chase_pages_va, # Use new init func
                       PAGE_SIZE) # Removed HUGE_CHUNK_SIZE

# --- Configuration ---
GPU_ID = 0
N_PAGES = 16 # Pages per set for probing
THRESHOLD_T = 1460 # SET THIS FROM YOUR BASELINE RUN
MAX_SAMPLES = 100000 # Max number of (t0, t1) pairs to collect
ALLOC_POOL_MB = 8 * 1024 # Allocate 8 GB
RESULTS_FILENAME = "receiver_results.npz" # Save raw data
PLOT_FILENAME = "receiver_timing.png" # Save plot
KERNELS_FILE = "channel_kernels.cu" # Source file
KERNELS_PTX = "channel_kernels.ptx" # Compiled PTX file

def main():
    # Load libs and get local handles
    driver_ok, local_libcuda, local_libcudart = load_cuda_libs()
    if not driver_ok: print("[Receiver] Failed loading libcuda."); exit(1)
    # Check if libcudart loaded, needed for sync
    if not local_libcudart: print("[Receiver] Error: libcudart handle is None, needed for sync."); exit(1)

    print(f"[Receiver] Starting. Threshold T = {THRESHOLD_T}")
    cu_context = None
    cu_module = None
    pool_gpu_va_start = 0 # For cleanup only
    d_page_vas0_ptr = ctypes.c_ulonglong(0) # Device ptr holding Set 0 VAs
    d_page_vas1_ptr = ctypes.c_ulonglong(0) # Device ptr holding Set 1 VAs
    results_buffer_gpu_va = 0
    results_buffer_gpu_ptr = ctypes.c_ulonglong(0) # Stores pointer object for results

    try:
        # --- Init CUDA ---
        CU_CHECK(local_libcuda.cuInit(0), "Receiver cuInit");
        cu_device = ctypes.c_int()
        CU_CHECK(local_libcuda.cuDeviceGet(ctypes.byref(cu_device), GPU_ID), "Receiver cuDeviceGet");
        cu_context_ptr = ctypes.c_void_p()
        CU_CHECK(local_libcuda.cuCtxCreate_v2(ctypes.byref(cu_context_ptr), 0, cu_device), "Receiver cuCtxCreate");
        cu_context = cu_context_ptr
        print("[Receiver] CUDA Context created.")

        # --- Allocate Memory ---
        print("[Receiver] Allocating memory...")
        pool_size_bytes = ALLOC_POOL_MB * 1024 * 1024
        pool_gpu_ptr = ctypes.c_ulonglong()
        flags = 1 # CU_MEM_ATTACH_GLOBAL
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(pool_gpu_ptr), pool_size_bytes, flags), "Receiver cuMemAllocManaged Pool")
        pool_gpu_va_start = pool_gpu_ptr.value
        if not pool_gpu_va_start: raise RuntimeError("Pool allocation failed")
        print(f"[Receiver] Pool GPU VA Start: {hex(pool_gpu_va_start)}")

        # Allocate results buffer (managed)
        results_buffer_size = MAX_SAMPLES * 2 * ctypes.sizeof(ctypes.c_ulonglong)
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(results_buffer_gpu_ptr), results_buffer_size, flags), "Receiver cuMemAllocManaged Results")
        results_buffer_gpu_va = results_buffer_gpu_ptr.value
        if not results_buffer_gpu_va: raise RuntimeError("Results buffer allocation failed")

        # --- Find Page Indices within Allocated Buffer ---
        indices0 = find_page_indices_in_buffer(local_libcuda, pool_gpu_va_start, pool_size_bytes, 0, N_PAGES)
        indices1 = find_page_indices_in_buffer(local_libcuda, pool_gpu_va_start, pool_size_bytes, 1, N_PAGES)
        if len(indices0) < N_PAGES or len(indices1) < N_PAGES:
             raise RuntimeError("Could not find enough pages for both sets.")
        # Calculate actual VAs needed for initialization and kernel args
        target_vas0 = [pool_gpu_va_start + i * PAGE_SIZE for i in indices0]
        target_vas1 = [pool_gpu_va_start + i * PAGE_SIZE for i in indices1]

        # --- Initialize Pages (passing VAs) ---
        initialize_chase_pages_va(local_libcuda, local_libcudart, target_vas0)
        initialize_chase_pages_va(local_libcuda, local_libcudart, target_vas1)

        # --- Allocate and Copy Target VAs ---
        VasListType = ctypes.c_ulonglong * N_PAGES
        h_vas0 = VasListType(*target_vas0); h_vas1 = VasListType(*target_vas1)
        CU_CHECK(local_libcuda.cuMemAlloc_v2(ctypes.byref(d_page_vas0_ptr), ctypes.sizeof(h_vas0)), "Receiver cuMemAlloc page VAs 0")
        CU_CHECK(local_libcuda.cuMemAlloc_v2(ctypes.byref(d_page_vas1_ptr), ctypes.sizeof(h_vas1)), "Receiver cuMemAlloc page VAs 1")
        CU_CHECK(local_libcuda.cuMemcpyHtoD_v2(d_page_vas0_ptr.value, ctypes.byref(h_vas0), ctypes.sizeof(h_vas0)), "Receiver cuMemcpyHtoD page VAs 0")
        CU_CHECK(local_libcuda.cuMemcpyHtoD_v2(d_page_vas1_ptr.value, ctypes.byref(h_vas1), ctypes.sizeof(h_vas1)), "Receiver cuMemcpyHtoD page VAs 1")
        print("[Receiver] Target VAs copied to device.")

        # --- Load PTX & Kernel ---
        cu_module_ptr = ctypes.c_void_p()
        with open(KERNELS_PTX, 'rb') as f: ptx_content = f.read()
        ptx_content_null_terminated = ctypes.create_string_buffer(ptx_content + b'\x00')
        CU_CHECK(local_libcuda.cuModuleLoadData(ctypes.byref(cu_module_ptr), ptx_content_null_terminated), "Receiver cuModuleLoadData")
        cu_module = cu_module_ptr
        receiver_kernel_func = ctypes.c_void_p()
        CU_CHECK(local_libcuda.cuModuleGetFunction(ctypes.byref(receiver_kernel_func), cu_module, b"receiver_probe_kernel"), "Receiver cuModuleGetFunction")

        # --- Prepare Launch Args (Revised Kernel) ---
        d_page_vas0_val = ctypes.c_ulonglong(d_page_vas0_ptr.value) # Ptr to list of VAs
        d_page_vas1_val = ctypes.c_ulonglong(d_page_vas1_ptr.value) # Ptr to list of VAs
        n_pages_val = ctypes.c_int(N_PAGES)
        results_buffer_val = ctypes.c_ulonglong(results_buffer_gpu_va) # Ptr to results buffer VA
        max_samples_val = ctypes.c_int(MAX_SAMPLES) # Max pairs

        launch_args = [
            ctypes.byref(d_page_vas0_val), ctypes.byref(d_page_vas1_val),
            ctypes.byref(n_pages_val), ctypes.byref(results_buffer_val),
            ctypes.byref(max_samples_val)]
        packed_args = (ctypes.c_void_p * len(launch_args))(*[ctypes.cast(p, ctypes.c_void_p) for p in launch_args])

        # --- Launch Kernel ---
        print("[Receiver] Launching probe kernel. Press Ctrl+C to stop and analyze.")
        start_time = time.time()
        CU_CHECK(local_libcuda.cuLaunchKernel(receiver_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, packed_args, None), "Receiver cuLaunchKernel")

        # --- Wait for Ctrl+C ---
        while True: time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Receiver] Ctrl+C detected. Stopping kernel and processing results...")
        try:
            # Sync using local_libcuda (Driver API stream sync)
            CU_CHECK(local_libcuda.cuStreamSynchronize(None), "Receiver cuStreamSynchronize")
            duration = time.time() - start_time
            print(f"[Receiver] Kernel synchronized after {duration:.2f}s.")

            # --- Get Results ---
            print("[Receiver] Reading results buffer...")
            # Array size should be max_samples * 2
            ResultsArrayType = ctypes.c_ulonglong * (MAX_SAMPLES * 2)
            results_ptr = ctypes.cast(results_buffer_gpu_va, ctypes.POINTER(ResultsArrayType))
            results_host_raw = np.frombuffer(results_ptr.contents, dtype=np.uint64).copy()

            # Process results
            last_nonzero = np.max(np.where(results_host_raw != 0)[0]) if np.any(results_host_raw != 0) else 0
            last_valid_index = last_nonzero + 1 if (last_nonzero + 1) % 2 == 0 else last_nonzero # Ensure pairs
            results_raw_valid = results_host_raw[:last_valid_index]
            results_pairs = results_raw_valid.reshape(-1, 2)
            num_samples_received = len(results_pairs)
            print(f"Read {num_samples_received} valid timing samples.")

            if num_samples_received > 0:
                 np.savez_compressed(RESULTS_FILENAME, timings=results_pairs, threshold=THRESHOLD_T)
                 print(f"Saved raw timings to {RESULTS_FILENAME}")
                 t0_misses = np.sum(results_pairs[:, 0] > THRESHOLD_T)
                 t1_misses = np.sum(results_pairs[:, 1] > THRESHOLD_T)
                 print(f"\n--- Quick Analysis ---")
                 print(f"T0 Misses (Expected if Sender used Set 0): {t0_misses} / {num_samples_received} ({t0_misses*100.0/num_samples_received:.1f}%)")
                 print(f"T1 Misses (Expected if Sender used Set 1): {t1_misses} / {num_samples_received} ({t1_misses*100.0/num_samples_received:.1f}%)")

                 # Plotting
                 try:
                     import matplotlib.pyplot as plt
                     plt.figure(figsize=(15, 5)); plt.plot(results_pairs[:, 0], label='t0', alpha=0.7); plt.plot(results_pairs[:, 1], label='t1', alpha=0.7); plt.axhline(THRESHOLD_T, color='r', linestyle='--', label=f'Threshold ({THRESHOLD_T})'); plt.title('Receiver Timing'); plt.xlabel('Sample Index'); plt.ylabel('Cycles'); plt.legend(); plt.ylim(bottom=0, top=max(2000, np.max(results_pairs)*1.1 if num_samples_received > 0 else 2000)); plt.grid(True); plt.savefig(PLOT_FILENAME)
                     print(f"Saved timing plot to {PLOT_FILENAME}")
                 except ImportError: print("matplotlib not found, skipping plot.")
                 except Exception as plot_e: print(f"Error plotting: {plot_e}")
            else: print("No valid samples recorded.")
        except Exception as e: print(f"\n[Receiver] Error during sync/results processing: {e}")

    except Exception as e: print(f"\n[Receiver] Error: {e}")
    finally: # --- Cleanup ---
        print("[Receiver] Cleaning up...")
        # Free memory (Use .value for device pointers)
        if d_page_vas0_ptr.value != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_page_vas0_ptr.value), "Receiver cuMemFree page VAs 0") 
            except Exception: pass
        if d_page_vas1_ptr.value != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_page_vas1_ptr.value), "Receiver cuMemFree page VAs 1") 
            except Exception: pass
        if results_buffer_gpu_va != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(results_buffer_gpu_va), "Receiver cuMemFree results") 
            except Exception: pass
        if pool_gpu_va_start != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(pool_gpu_va_start), "Receiver cuMemFree Pool") 
            except Exception: pass
        # Destroy context
        if cu_context and cu_context.value: 
            try: CU_CHECK(local_libcuda.cuCtxDestroy_v2(cu_context), "Receiver cuCtxDestroy"); print("[Receiver] Context destroyed.") 
            except Exception: pass
        print("[Receiver] Finished.")

if __name__ == "__main__":
    # Load libs globally once first for arg check safety, won't hurt
    load_cuda_libs()
    main() # Call main function