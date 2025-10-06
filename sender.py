import ctypes
import numpy as np
import time
import os
import sys
# Import functions from gpu_utils
from gpu_utils import (load_cuda_libs, CU_CHECK, compile_kernels_to_ptx,
                       find_page_indices_in_buffer, PAGE_SIZE)

# --- Configuration ---
GPU_ID = 0
N_PAGES = 16
ALLOC_POOL_MB = 8 * 1024 # Allocate 8 GB
KERNELS_FILE = "channel_kernels.cu" # Source file
KERNELS_PTX = "channel_kernels.ptx" # Compiled PTX file

def main(target_set_arg):
    # Load libs and get local handles
    driver_ok, local_libcuda, local_libcudart = load_cuda_libs()
    if not driver_ok:
        print("[Sender] Failed to load essential CUDA Driver library.")
        exit(1)
    # Use local_libcuda for all subsequent Driver API calls

    try:
        target_set = int(target_set_arg)
        if target_set not in [0, 1]:
            print("Error: Target set must be 0 or 1"); exit(1)
    except ValueError:
        print("Error: Target set must be 0 or 1"); exit(1)

    print(f"[Sender] Starting. Target Set: {target_set}")

    cu_context = None
    cu_module = None
    pool_gpu_va_start = 0 # For cleanup only
    d_page_vas_ptr = ctypes.c_ulonglong(0) # Device ptr holding target VAs
    d_stop_flag_ptr = ctypes.c_ulonglong(0) # Device ptr for stop flag

    try:
        # --- Init CUDA ---
        CU_CHECK(local_libcuda.cuInit(0), "Sender cuInit")
        cu_device = ctypes.c_int()
        CU_CHECK(local_libcuda.cuDeviceGet(ctypes.byref(cu_device), GPU_ID), "Sender cuDeviceGet")
        cu_context_ptr = ctypes.c_void_p()
        CU_CHECK(local_libcuda.cuCtxCreate_v2(ctypes.byref(cu_context_ptr), 0, cu_device), "Sender cuCtxCreate")
        cu_context = cu_context_ptr
        print("[Sender] CUDA Context created.")

        # --- Allocate Memory ---
        print("[Sender] Allocating memory...")
        pool_size_bytes = ALLOC_POOL_MB * 1024 * 1024
        pool_gpu_ptr = ctypes.c_ulonglong()
        flags = 1 # CU_MEM_ATTACH_GLOBAL
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(pool_gpu_ptr), pool_size_bytes, flags), "Sender cuMemAllocManaged Pool")
        pool_gpu_va_start = pool_gpu_ptr.value # Keep for cleanup consistency
        if not pool_gpu_va_start: raise RuntimeError("Pool allocation failed")
        print(f"[Sender] Pool GPU VA Start: {hex(pool_gpu_va_start)}")

        # --- Find Page Indices within Allocated Buffer ---
        page_indices = find_page_indices_in_buffer(local_libcuda, pool_gpu_va_start, pool_size_bytes, target_set, N_PAGES)
        if len(page_indices) < N_PAGES:
            raise RuntimeError(f"Could not find enough pages for target set {target_set}")
        # Calculate the actual VAs from indices
        target_vas = [pool_gpu_va_start + i * PAGE_SIZE for i in page_indices]

        # --- Allocate and Copy Target VAs ---
        VasListType = ctypes.c_ulonglong * N_PAGES
        h_vas = VasListType(*target_vas) # List of actual target VAs
        CU_CHECK(local_libcuda.cuMemAlloc_v2(ctypes.byref(d_page_vas_ptr), ctypes.sizeof(h_vas)), "Sender cuMemAlloc page VAs")
        CU_CHECK(local_libcuda.cuMemcpyHtoD_v2(d_page_vas_ptr.value, ctypes.byref(h_vas), ctypes.sizeof(h_vas)), "Sender cuMemcpyHtoD page VAs")
        print("[Sender] Target VAs copied to device.")
        
        # --- Allocate Stop Flag ---
        stop_flag_size = ctypes.sizeof(ctypes.c_int)
        CU_CHECK(local_libcuda.cuMemAlloc_v2(ctypes.byref(d_stop_flag_ptr), stop_flag_size), "Sender cuMemAlloc stop flag")
        h_stop_flag = ctypes.c_int(0) # Initialize to 0 (continue running)
        CU_CHECK(local_libcuda.cuMemcpyHtoD_v2(d_stop_flag_ptr.value, ctypes.byref(h_stop_flag), stop_flag_size), "Sender cuMemcpyHtoD stop flag")
        print("[Sender] Stop flag initialized on device.")

        # --- Load PTX & Kernel ---
        cu_module_ptr = ctypes.c_void_p()
        with open(KERNELS_PTX, 'rb') as f: ptx_content = f.read()
        ptx_content_null_terminated = ctypes.create_string_buffer(ptx_content + b'\x00')
        CU_CHECK(local_libcuda.cuModuleLoadData(ctypes.byref(cu_module_ptr), ptx_content_null_terminated), "Sender cuModuleLoadData")
        cu_module = cu_module_ptr
        sender_kernel_func = ctypes.c_void_p()
        CU_CHECK(local_libcuda.cuModuleGetFunction(ctypes.byref(sender_kernel_func), cu_module, b"sender_contention_kernel"), "Sender cuModuleGetFunction")

        # --- Prepare Launch Args (Revised Kernel) ---
        d_page_vas_val = ctypes.c_ulonglong(d_page_vas_ptr.value) # Pass VA of VA list
        n_pages_val = ctypes.c_int(N_PAGES)
        d_stop_flag_val = ctypes.c_ulonglong(d_stop_flag_ptr.value) # Pass VA of stop flag
        launch_args = [
            ctypes.byref(d_page_vas_val), # Pointer to device ptr value
            ctypes.byref(n_pages_val),
            ctypes.byref(d_stop_flag_val) # Pointer to stop flag
        ]
        packed_args = (ctypes.c_void_p * len(launch_args))(*[ctypes.cast(p, ctypes.c_void_p) for p in launch_args])

        # --- Launch Kernel ---
        print(f"[Sender] Launching contention kernel for Set {target_set}. Press Ctrl+C to stop.")
        CU_CHECK(local_libcuda.cuLaunchKernel(sender_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, packed_args, None), "Sender cuLaunchKernel")

        # --- Run Loop (Wait for Ctrl+C) ---
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Sender] Ctrl+C detected. Setting stop flag...")
        # Set stop flag to signal kernel to exit
        if d_stop_flag_ptr.value != 0:
            h_stop_flag = ctypes.c_int(1) # Set to 1 to stop
            CU_CHECK(local_libcuda.cuMemcpyHtoD_v2(d_stop_flag_ptr.value, ctypes.byref(h_stop_flag), ctypes.sizeof(h_stop_flag)), "Sender cuMemcpyHtoD stop flag")
            # Wait a bit for kernel to finish
            time.sleep(0.5)
        print("[Sender] Exiting.")
    except Exception as e:
        print(f"\n[Sender] Error: {e}")
    finally: # --- Cleanup (using local_libcuda) ---
        print("[Sender] Cleaning up...")
        if d_stop_flag_ptr.value != 0:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_stop_flag_ptr.value), "Sender cuMemFree stop flag")
            except Exception: pass
        if d_page_vas_ptr.value != 0:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_page_vas_ptr.value), "Sender cuMemFree page VAs")
            except Exception: pass
        if pool_gpu_va_start != 0:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(pool_gpu_va_start), "Sender cuMemFree Pool")
            except Exception: pass
        if cu_context and cu_context.value:
            try: CU_CHECK(local_libcuda.cuCtxDestroy_v2(cu_context), "Sender cuCtxDestroy"); print("[Sender] Context destroyed.")
            except Exception: pass
        print("[Sender] Finished.")

if __name__ == "__main__":
    # Load libs globally once first just for argument check safety, won't hurt
    load_cuda_libs()
    if len(sys.argv) != 2:
        print("Usage: python sender.py <set_id>")
        print("  <set_id>: 0 or 1")
        sys.exit(1)
    main(sys.argv[1]) # Pass argument to main