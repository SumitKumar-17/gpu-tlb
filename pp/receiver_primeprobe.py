import ctypes
import numpy as np
import time
import os
import sys
# Import functions from gpu_utils_pp
from gpu_utils_pp import (load_cuda_libs_pp, CU_CHECK, CUDA_RT_CHECK,
                          compile_kernels_to_ptx_pp, find_pages_for_set,
                          PAGE_SIZE, PRIME_PROBE_N, L1_L2_EVICT_COUNT) # Added L1_L2_EVICT_COUNT

# --- Configuration ---
GPU_ID = 0
# <<< CHANGE: Use new threshold from baseline >>>
THRESHOLD_T_PROBE = 1394 # Updated threshold
# <<< ----------------------------------------- >>>
MAX_SAMPLES = 50000   # Max number of Prime+Probe cycles
MONITORED_SET = 0     # Which L3 set index to Prime+Probe
ALLOC_POOL_MB_RECEIVER = 4 * 1024 # 4 GB pool
RESULTS_FILENAME = "receiver_pp_results.npz"
PLOT_FILENAME = "receiver_pp_timing.png"
KERNELS_FILE = "pp_baseline_kernels.cu" # Source file containing all kernels now
KERNELS_PTX = "pp_baseline_kernels.ptx" # Use the PTX compiled from the above

def main():
    # Load libs and get local handles
    driver_ok, local_libcuda, local_libcudart = load_cuda_libs_pp()
    if not driver_ok: print("[ReceiverPP] Failed loading libcuda."); exit(1)
    if not local_libcudart: print("[ReceiverPP] Error: libcudart handle needed for sync."); exit(1)

    print(f"[ReceiverPP] Starting. Monitoring Set {MONITORED_SET}. Threshold T_probe = {THRESHOLD_T_PROBE}")

    cu_context = None; cu_module = None; pool_gpu_va_start = 0
    d_pp_vas_ptr = ctypes.c_ulonglong(0) # Device ptr holding Prime+Probe VAs
    d_l1l2_evict_vas_ptr = ctypes.c_ulonglong(0) # <<< ADDED
    results_buffer_gpu_va = 0; results_buffer_gpu_ptr = ctypes.c_ulonglong(0)

    try:
        # --- Init CUDA ---
        CU_CHECK(local_libcuda.cuInit(0), "ReceiverPP cuInit"); # ... create context ...
        cu_device = ctypes.c_int(); CU_CHECK(local_libcuda.cuDeviceGet(ctypes.byref(cu_device), GPU_ID), "ReceiverPP cuDeviceGet");
        cu_context_ptr = ctypes.c_void_p(); CU_CHECK(local_libcuda.cuCtxCreate_v2(ctypes.byref(cu_context_ptr), 0, cu_device), "ReceiverPP cuCtxCreate");
        cu_context = cu_context_ptr
        print("[ReceiverPP] CUDA Context created.")

        # --- Allocate Memory ---
        print("[ReceiverPP] Allocating memory...")
        pool_size_bytes = ALLOC_POOL_MB_RECEIVER * 1024 * 1024
        pool_gpu_ptr = ctypes.c_ulonglong(); flags = 1
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(pool_gpu_ptr), pool_size_bytes, flags), "ReceiverPP cuMemAllocManaged Pool")
        pool_gpu_va_start = pool_gpu_ptr.value
        if not pool_gpu_va_start: raise RuntimeError("Pool allocation failed")
        print(f"[ReceiverPP] Pool GPU VA Start: {hex(pool_gpu_va_start)}")

        # Allocate results buffer (managed)
        results_buffer_size = MAX_SAMPLES * ctypes.sizeof(ctypes.c_ulonglong) # Store single time per sample
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(results_buffer_gpu_ptr), results_buffer_size, flags), "ReceiverPP cuMemAllocManaged Results")
        results_buffer_gpu_va = results_buffer_gpu_ptr.value
        if not results_buffer_gpu_va: raise RuntimeError("Results buffer allocation failed")

        # --- Find Pages ---
        prime_probe_vas = find_pages_for_set(local_libcuda, pool_gpu_va_start, pool_size_bytes, MONITORED_SET, PRIME_PROBE_N)
        if len(prime_probe_vas) < PRIME_PROBE_N: raise RuntimeError(f"Could not find enough pages for monitored set {MONITORED_SET}")
        # <<< ADDED: Find L1/L2 eviction pages >>>
        l1l2_evict_set_id = (MONITORED_SET + 10) % 256
        l1l2_evict_vas = find_pages_for_set(local_libcuda, pool_gpu_va_start, pool_size_bytes, l1l2_evict_set_id, L1_L2_EVICT_COUNT)
        if len(l1l2_evict_vas) < L1_L2_EVICT_COUNT: raise RuntimeError(f"Could not find enough pages for L1/L2 eviction set {l1l2_evict_set_id}")
        # <<< --------------------------------- >>>

        # --- Allocate and Copy VAs ---
        VasListTypePP = ctypes.c_ulonglong * PRIME_PROBE_N
        h_pp_vas = VasListTypePP(*prime_probe_vas)
        CU_CHECK(local_libcuda.cuMemAlloc_v2(ctypes.byref(d_pp_vas_ptr), ctypes.sizeof(h_pp_vas)), "ReceiverPP cuMemAlloc PP VAs")
        CU_CHECK(local_libcuda.cuMemcpyHtoD_v2(d_pp_vas_ptr.value, ctypes.byref(h_pp_vas), ctypes.sizeof(h_pp_vas)), "ReceiverPP cuMemcpyHtoD PP VAs")
        print(f"[ReceiverPP] {PRIME_PROBE_N} Prime+Probe VAs for set {MONITORED_SET} copied.")
        # <<< ADDED: Allocate and copy L1/L2 VAs >>>
        VasListTypeL1L2 = ctypes.c_ulonglong * L1_L2_EVICT_COUNT
        h_l1l2_vas = VasListTypeL1L2(*l1l2_evict_vas)
        CU_CHECK(local_libcuda.cuMemAlloc_v2(ctypes.byref(d_l1l2_evict_vas_ptr), ctypes.sizeof(h_l1l2_vas)), "ReceiverPP cuMemAlloc L1L2 VAs")
        CU_CHECK(local_libcuda.cuMemcpyHtoD_v2(d_l1l2_evict_vas_ptr.value, ctypes.byref(h_l1l2_vas), ctypes.sizeof(h_l1l2_vas)), "ReceiverPP cuMemcpy L1L2 VAs")
        print(f"[ReceiverPP] {L1_L2_EVICT_COUNT} L1/L2 Eviction VAs copied.")
        # <<< ---------------------------------- >>>

        # --- Load PTX & Kernel ---
        cu_module_ptr = ctypes.c_void_p()
        try:
            with open(KERNELS_PTX, 'rb') as f: ptx_content = f.read()
            ptx_content_null_terminated = ctypes.create_string_buffer(ptx_content + b'\x00')
            CU_CHECK(local_libcuda.cuModuleLoadData(ctypes.byref(cu_module_ptr), ptx_content_null_terminated), "ReceiverPP cuModuleLoadData")
            cu_module = cu_module_ptr
        except FileNotFoundError: raise RuntimeError(f"Kernel file {KERNELS_PTX} not found.")
        receiver_kernel_func = ctypes.c_void_p()
        CU_CHECK(local_libcuda.cuModuleGetFunction(ctypes.byref(receiver_kernel_func), cu_module, b"probe_time_kernel"), "ReceiverPP cuModuleGetFunction")

        # --- Prepare Launch Args ---
        d_pp_vas_val = ctypes.c_ulonglong(d_pp_vas_ptr.value)
        n_pages_val = ctypes.c_int(PRIME_PROBE_N)
        d_l1l2_evict_vas_val = ctypes.c_ulonglong(d_l1l2_evict_vas_ptr.value) # <<< ADDED
        n_l1l2_evict_val = ctypes.c_int(L1_L2_EVICT_COUNT) # <<< ADDED
        d_result_time_kernel_arg = ctypes.c_ulonglong(0) # Placeholder, set in loop

        # --- Launch Kernel Loop ---
        print("[ReceiverPP] Starting Prime+Probe loop. Press Ctrl+C to stop.")
        start_time = time.time()
        actual_samples = 0
        try:
            for i in range(MAX_SAMPLES):
                # Point result argument to the correct location
                current_result_gpu_va = results_buffer_gpu_va + i * ctypes.sizeof(ctypes.c_ulonglong)
                d_result_time_kernel_arg = ctypes.c_ulonglong(current_result_gpu_va) # VA for this sample

                # Args for probe_time_kernel: page_vas, num_pages, l1_l2_evict_vas, num_l1_l2_evict, result_time
                probe_args = [
                    ctypes.byref(d_pp_vas_val),
                    ctypes.byref(n_pages_val),
                    ctypes.byref(d_l1l2_evict_vas_val), # <<< ADDED
                    ctypes.byref(n_l1l2_evict_val),   # <<< ADDED
                    ctypes.byref(d_result_time_kernel_arg)
                ]
                packed_args = (ctypes.c_void_p * len(probe_args))(*[ctypes.cast(p, ctypes.c_void_p) for p in probe_args])

                CU_CHECK(local_libcuda.cuLaunchKernel(receiver_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, packed_args, None), f"ReceiverPP Launch {i}")
                CU_CHECK(local_libcuda.cuStreamSynchronize(None), f"ReceiverPP Sync {i}")
                actual_samples += 1

                # Optional small host delay
                time.sleep(0.0001)

        except KeyboardInterrupt: print("\n[ReceiverPP] Ctrl+C detected. Stopping.")
        # Let execution fall through to result processing

        duration = time.time() - start_time
        print(f"[ReceiverPP] Measurement loop finished/interrupted after {duration:.2f}s.")

        # --- Get and Process Results ---
        print("[ReceiverPP] Reading results buffer...")
        # Read only up to the number of samples actually launched
        ResultArrayType = ctypes.c_ulonglong * actual_samples
        results_ptr_host_view = ctypes.cast(results_buffer_gpu_va, ctypes.POINTER(ResultArrayType))
        probe_times_valid = np.frombuffer(results_ptr_host_view.contents, dtype=np.uint64).copy()
        num_samples_received = len(probe_times_valid)
        print(f"Read {num_samples_received} probe time samples.")

        if num_samples_received > 0:
            np.savez_compressed(RESULTS_FILENAME, timings=probe_times_valid, threshold=THRESHOLD_T_PROBE)
            print(f"Saved raw probe times to {RESULTS_FILENAME}")
            decoded_bits = (probe_times_valid > THRESHOLD_T_PROBE).astype(int)
            num_ones = np.sum(decoded_bits); num_zeros = len(decoded_bits) - num_ones
            print(f"\n--- Quick Analysis (Set {MONITORED_SET}) ---")
            print(f"Probe Times > Threshold ({THRESHOLD_T_PROBE}) ('1'): {num_ones} / {num_samples_received} ({num_ones*100.0/num_samples_received:.1f}%)")
            print(f"Probe Times <= Threshold ({THRESHOLD_T_PROBE}) ('0'): {num_zeros} / {num_samples_received} ({num_zeros*100.0/num_samples_received:.1f}%)")
            try: # Plotting
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15, 5)); plt.plot(probe_times_valid, label=f'Probe Time (Set {MONITORED_SET})', marker='.', linestyle='None', markersize=2); plt.axhline(THRESHOLD_T_PROBE, color='r', linestyle='--', label=f'Threshold ({THRESHOLD_T_PROBE})'); plt.title(f'Receiver Prime+Probe Timing (Set {MONITORED_SET})'); plt.xlabel('Sample Index'); plt.ylabel('Total Probe Cycles'); plt.legend(); plt.grid(True); plt.ylim(bottom=0); plt.savefig(PLOT_FILENAME)
                print(f"Saved timing plot to {PLOT_FILENAME}")
            except ImportError: print("matplotlib not found, skipping plot.")
            except Exception as plot_e: print(f"Error plotting: {plot_e}")
        else: print("No valid samples recorded.")

    except Exception as e: print(f"\n[ReceiverPP] Error: {e}"); import traceback; traceback.print_exc()
    finally: # --- Cleanup ---
        print("[ReceiverPP] Cleaning up...")
        # Free device memory
        if d_pp_vas_ptr.value != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_pp_vas_ptr.value), "ReceiverPP Free PP VAs") 
            except Exception: pass
        if d_l1l2_evict_vas_ptr.value != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_l1l2_evict_vas_ptr.value), "ReceiverPP Free L1L2 VAs") 
            except Exception: pass
        # Free managed memory
        if results_buffer_gpu_va != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(results_buffer_gpu_va), "ReceiverPP Free Results") 
            except Exception: pass
        if pool_gpu_va_start != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(pool_gpu_va_start), "ReceiverPP Free Pool") 
            except Exception: pass
        # Destroy context
        if cu_context and cu_context.value: 
            try: CU_CHECK(local_libcuda.cuCtxDestroy_v2(cu_context), "ReceiverPP CtxDestroy"); print("[ReceiverPP] Context destroyed.") 
            except Exception: pass
        print("[ReceiverPP] Finished.")

if __name__ == "__main__":
    load_cuda_libs_pp()
    main()