import ctypes
import numpy as np
import time
import os
import sys
# Import functions from gpu_utils_pp
from gpu_utils_pp import (load_cuda_libs_pp, CU_CHECK, compile_kernels_to_ptx_pp,
                          find_pages_for_set, PAGE_SIZE, PRIME_PROBE_N) # Use PRIME_PROBE_N for consistency?

# --- Configuration ---
GPU_ID = 0
N_PAGES_SENDER = PRIME_PROBE_N # Use same N as receiver uses for probing (e.g., 17)
ALLOC_POOL_MB_SENDER = 2 * 1024 # 2 GB pool for sender's pages
KERNELS_FILE = "pp_baseline_kernels.cu" # Source file containing all kernels now
KERNELS_PTX = "pp_baseline_kernels.ptx" # Use the PTX compiled from the above
TARGET_SET_COVERT = 0 # The L3 set receiver monitors (e.g., Set 0)
NON_TARGET_SET_COVERT = 5 # A different L3 set to access for sending '0'

def main(target_set_arg):
    # Load libs and get local handles
    driver_ok, local_libcuda, local_libcudart = load_cuda_libs_pp()
    if not driver_ok: print("[SenderPP] Failed loading libcuda."); exit(1)

    try:
        bit_to_send = int(target_set_arg) # Argument now IS the bit
        if bit_to_send not in [0, 1]: raise ValueError()
        target_set = TARGET_SET_COVERT if bit_to_send == 1 else NON_TARGET_SET_COVERT
    except ValueError: print("Error: Argument must be 0 or 1 (the bit to send)"); exit(1)

    print(f"[SenderPP] Starting. Sending Bit '{bit_to_send}' (Contending Set {target_set}).")

    cu_context = None; cu_module = None
    pool_gpu_va_start = 0; d_page_vas_ptr = ctypes.c_ulonglong(0)

    try:
        # --- Init CUDA ---
        CU_CHECK(local_libcuda.cuInit(0), "SenderPP cuInit"); cu_device = ctypes.c_int()
        CU_CHECK(local_libcuda.cuDeviceGet(ctypes.byref(cu_device), GPU_ID), "SenderPP cuDeviceGet"); cu_context_ptr = ctypes.c_void_p()
        CU_CHECK(local_libcuda.cuCtxCreate_v2(ctypes.byref(cu_context_ptr), 0, cu_device), "SenderPP cuCtxCreate"); cu_context = cu_context_ptr
        print("[SenderPP] CUDA Context created.")

        # --- Allocate Memory ---
        print("[SenderPP] Allocating memory...")
        pool_size_bytes = ALLOC_POOL_MB_SENDER * 1024 * 1024
        pool_gpu_ptr = ctypes.c_ulonglong()
        flags = 1
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(pool_gpu_ptr), pool_size_bytes, flags), "SenderPP cuMemAllocManaged Pool")
        pool_gpu_va_start = pool_gpu_ptr.value
        if not pool_gpu_va_start: raise RuntimeError("Pool allocation failed")
        print(f"[SenderPP] Pool GPU VA Start: {hex(pool_gpu_va_start)}")

        # --- Find Pages for the chosen set ---
        page_vas = find_pages_for_set(local_libcuda, pool_gpu_va_start, pool_size_bytes, target_set, N_PAGES_SENDER)
        if len(page_vas) < N_PAGES_SENDER: raise RuntimeError(f"Could not find enough pages for target set {target_set}")

        # --- Allocate and Copy Target VAs ---
        VasListType = ctypes.c_ulonglong * N_PAGES_SENDER
        h_vas = VasListType(*page_vas)
        CU_CHECK(local_libcuda.cuMemAlloc_v2(ctypes.byref(d_page_vas_ptr), ctypes.sizeof(h_vas)), "SenderPP cuMemAlloc page VAs")
        CU_CHECK(local_libcuda.cuMemcpyHtoD_v2(d_page_vas_ptr.value, ctypes.byref(h_vas), ctypes.sizeof(h_vas)), "SenderPP cuMemcpyHtoD page VAs")
        print(f"[SenderPP] {N_PAGES_SENDER} Target VAs for set {target_set} copied to device.")

        # --- Load PTX & Kernel ---
        cu_module_ptr = ctypes.c_void_p()
        try:
            with open(KERNELS_PTX, 'rb') as f: ptx_content = f.read()
            ptx_content_null_terminated = ctypes.create_string_buffer(ptx_content + b'\x00')
            CU_CHECK(local_libcuda.cuModuleLoadData(ctypes.byref(cu_module_ptr), ptx_content_null_terminated), "SenderPP cuModuleLoadData")
            cu_module = cu_module_ptr
        except FileNotFoundError: raise RuntimeError(f"Kernel file {KERNELS_PTX} not found.")

        sender_kernel_func = ctypes.c_void_p()
        # <<< CHANGE: Use new kernel name >>>
        CU_CHECK(local_libcuda.cuModuleGetFunction(ctypes.byref(sender_kernel_func), cu_module, b"sender_contention_kernel_pp"), "SenderPP cuModuleGetFunction")
        # <<< ----------------------------- >>>


        # --- Prepare Launch Args ---
        d_page_vas_val = ctypes.c_ulonglong(d_page_vas_ptr.value) # Pass VA of VA list
        n_pages_val = ctypes.c_int(N_PAGES_SENDER)
        launch_args = [ ctypes.byref(d_page_vas_val), ctypes.byref(n_pages_val) ]
        packed_args = (ctypes.c_void_p * len(launch_args))(*[ctypes.cast(p, ctypes.c_void_p) for p in launch_args])

        # --- Launch Kernel ---
        print(f"[SenderPP] Launching contention kernel for Set {target_set}. Press Ctrl+C to stop.")
        CU_CHECK(local_libcuda.cuLaunchKernel(sender_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, packed_args, None), "SenderPP cuLaunchKernel")

        # --- Run Loop (Wait for Ctrl+C) ---
        while True: time.sleep(1)

    except KeyboardInterrupt: print("\n[SenderPP] Ctrl+C detected.")
    except Exception as e: print(f"\n[SenderPP] Error: {e}")
    finally: # --- Cleanup ---
        print("[SenderPP] Cleaning up...")
        if d_page_vas_ptr.value != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_page_vas_ptr.value), "SenderPP Free VAs") 
            except Exception: pass
        if pool_gpu_va_start != 0: 
            try: CU_CHECK(local_libcuda.cuMemFree_v2(pool_gpu_va_start), "SenderPP Free Pool") 
            except Exception: pass
        if cu_context and cu_context.value: 
            try: CU_CHECK(local_libcuda.cuCtxDestroy_v2(cu_context), "SenderPP CtxDestroy"); print("[SenderPP] Context destroyed.") 
            except Exception: pass
        print("[SenderPP] Finished.")

if __name__ == "__main__":
    load_cuda_libs_pp() # Load libs globally once first for arg check safety
    if len(sys.argv) != 2:
        print("Usage: python sender_primeprobe.py <bit>")
        print("  <bit>: 0 or 1")
        sys.exit(1)
    main(sys.argv[1])