import subprocess
import ctypes
import numpy as np
import time
import os
import random
from l3_hash_rtx3060 import get_l3_set_index_rtx3060_64k # Import the hash function

# --- Configuration ---
GPU_ID = 0
N_PAGES = 11         # Number of pages per sequence (like paper)
PAGE_SIZE = 64 * 1024 # Use 64KB pages
ALLOC_POOL_SIZE_MB = 256 # Size of initial memory pool to find pages

# Channel Parameters (Tune these!)
DELAY_MS_SENDER = 20  # Host sleep delay (ms) for sender between sending bits
THRESHOLD_T = 1504    # Timing threshold from baseline (e.g., 1460 cycles)
NUM_BITS = 8          # Number of bits to transmit (reduced for testing)
RECEIVER_SAMPLES_PER_BIT = 10 # How many (t0, t1) samples receiver tries per expected bit duration

# Kernel file and PTX output
KERNELS_FILE = "channel_kernels.cu" # Use the new kernel file
KERNELS_PTX = "channel_kernels.ptx"

# --- Globals for CUDA Lib Handles ---
libcuda = None
libcudart = None

# --- CUDA API Helper Functions ---
def load_cuda_libs():
    global libcuda, libcudart
    # Only load if not already loaded in this process
    if libcuda is None:
        try:
            libcuda = ctypes.CDLL("libcuda.so")
            print(f"libcuda.so loaded in process {os.getpid()}.")
            # Define Driver API prototypes
            libcuda.cuInit.argtypes = [ctypes.c_uint]; libcuda.cuInit.restype = int
            libcuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]; libcuda.cuDeviceGet.restype = int
            libcuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]; libcuda.cuCtxCreate_v2.restype = int
            libcuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]; libcuda.cuCtxDestroy_v2.restype = int
            libcuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]; libcuda.cuModuleLoadData.restype = int
            libcuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]; libcuda.cuModuleGetFunction.restype = int
            libcuda.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
            libcuda.cuLaunchKernel.restype = int
            libcuda.cuStreamSynchronize.argtypes = [ctypes.c_void_p]; libcuda.cuStreamSynchronize.restype = int
            libcuda.cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t, ctypes.c_uint]; libcuda.cuMemAllocManaged.restype = int
            libcuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]; libcuda.cuMemAlloc_v2.restype = int
            libcuda.cuMemFree_v2.argtypes = [ctypes.c_ulonglong]; libcuda.cuMemFree_v2.restype = int
            libcuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_size_t]; libcuda.cuMemcpyHtoD_v2.restype = int
            libcuda.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_ulonglong]; libcuda.cuPointerGetAttribute.restype = int
        except OSError:
            print(f"Error: libcuda.so not found in process {os.getpid()}."); libcuda = None

    if libcudart is None:
        try:
            libcudart = ctypes.CDLL("libcudart.so")
            print(f"libcudart.so loaded in process {os.getpid()}.")
            # Define Runtime API prototypes
            libcudart.cudaGetErrorString.restype = ctypes.c_char_p; libcudart.cudaGetErrorString.argtypes = [ctypes.c_int]
            libcudart.cudaDeviceSynchronize.restype = int
            libcudart.cudaSetDevice.argtypes = [ctypes.c_int]; libcudart.cudaSetDevice.restype = int
        except OSError:
            print(f"Warning: libcudart.so not found in process {os.getpid()}.")
            libcudart = None

    # Return True if essential lib (libcuda) was loaded
    return libcuda is not None

def CU_CHECK(err_code, func_name):
    if err_code != 0: raise RuntimeError(f"CUDA Driver API Error in {func_name}: Code {err_code}")

def CUDA_RT_CHECK(err_code, func_name):
     if err_code != 0:
         err_str = f"Error Code {err_code}"
         try:
              if libcudart: err_str = libcudart.cudaGetErrorString(err_code).decode('utf-8')
         except: pass
         raise RuntimeError(f"CUDA Runtime API Error in {func_name}: {err_str}")

# --- Compile Kernels ---
def compile_kernels_to_ptx():
    print(f"Compiling {KERNELS_FILE} to {KERNELS_PTX}...")
    try:
        subprocess.run(['nvcc', '-O3', '--ptx', '-o', KERNELS_PTX, KERNELS_FILE, '-gencode', 'arch=compute_86,code=sm_86'], check=True)
        print("PTX compilation successful.")
        return True
    except Exception as e:
        print(f"PTX Compilation failed: {e}")
        return False

# --- Helper: Find Suitable Pages (Driver API Version) ---
def find_eviction_pages_driver(n_pages_needed, page_size, pool_size_mb):
    print(f"Finding {n_pages_needed} pages for two distinct L3 sets (Driver Alloc)...")
    if not libcuda: print("Driver library not loaded."); return None, None, None, 0

    pool_size_bytes = pool_size_mb * 1024 * 1024
    pool_gpu_ptr = ctypes.c_ulonglong() # Driver API returns CUdeviceptr (ulonglong)
    flags = 1 # CU_MEM_ATTACH_GLOBAL
    try:
        CU_CHECK(libcuda.cuMemAllocManaged(ctypes.byref(pool_gpu_ptr), pool_size_bytes, flags), "find_eviction cuMemAllocManaged")
    except Exception as e:
        print(f"cuMemAllocManaged failed: {e}")
        return None, None, None, 0

    pool_gpu_ptr_val = pool_gpu_ptr.value
    if not pool_gpu_ptr_val: print("Error: cuMemAllocManaged returned null pointer."); return None, None, None, 0
    pool_gpu_va_start = pool_gpu_ptr_val # This IS the GPU VA
    print(f"Allocated managed pool. GPU VA Start: {hex(pool_gpu_va_start)}")

    # Iterate potential pages using GPU VA, calculate L3 set, find indices
    page_indices = []; page_l3_sets = []; num_pages_in_pool = pool_size_bytes // page_size
    print("Iterating through GPU VA range based on pool start...")
    for i in range(num_pages_in_pool):
        current_gpu_va = pool_gpu_va_start + i * page_size
        try: l3_set = get_l3_set_index_rtx3060_64k(current_gpu_va); page_indices.append(i); page_l3_sets.append(l3_set)
        except Exception as e: print(f"Warning: Error calculating L3 set for index {i}, VA {hex(current_gpu_va)}: {e}"); continue
    if not page_indices: print("Error: Failed to process any potential pages."); libcuda.cuMemFree_v2(pool_gpu_va_start); return None, None, None, 0

    # Find two distinct sets
    sets_found = {}; set0_indices = None; set1_indices = None; set0_l3_idx = -1; set1_l3_idx = -1
    for idx, l3_set in zip(page_indices, page_l3_sets):
        if l3_set not in sets_found: sets_found[l3_set] = []
        if len(sets_found[l3_set]) < n_pages_needed: sets_found[l3_set].append(idx)
    sorted_sets = sorted(sets_found.items())
    for l3_set, index_list in sorted_sets:
        if len(index_list) >= n_pages_needed:
            if set0_indices is None: set0_indices = index_list[:n_pages_needed]; set0_l3_idx = l3_set
            elif set1_indices is None and l3_set != set0_l3_idx : set1_indices = index_list[:n_pages_needed]; set1_l3_idx = l3_set; break
    if set0_indices and set1_indices:
        print(f"Found page indices: Set {set0_l3_idx} ({len(set0_indices)} indices), Set {set1_l3_idx} ({len(set1_indices)} indices)")
        return pool_gpu_va_start, set0_indices, set1_indices, pool_size_bytes
    else: print("Error: Could not find two distinct L3 sets."); libcuda.cuMemFree_v2(pool_gpu_va_start); return None, None, None, 0

# --- Helper Function: Initialize Pages for Pointer Chasing (Driver API Context) ---
def initialize_chase_pages_driver(pool_gpu_va_start, page_indices, n_pages):
    if not libcuda: raise RuntimeError("libcuda needed for pointer attribute")
    if not libcudart: raise RuntimeError("libcudart needed for sync")

    print(f"Initializing {len(page_indices)} pages for pointer chasing...")
    chase_length = 8
    page_size = PAGE_SIZE
    sizeof_ulonglong = ctypes.sizeof(ctypes.c_ulonglong)

    # Get Host pointer corresponding to the start GPU VA
    pool_host_ptr_attr = ctypes.c_void_p()
    CU_CHECK(libcuda.cuPointerGetAttribute(ctypes.byref(pool_host_ptr_attr), 4, pool_gpu_va_start), "initialize_chase cuPointerGetAttribute HOST_POINTER")
    if not pool_host_ptr_attr.value: raise RuntimeError("Could not get host pointer for pool in init")
    pool_host_ptr_val = pool_host_ptr_attr.value

    # Initialize pages using the host pointer
    for page_idx_in_pool in page_indices:
        page_host_ptr = ctypes.c_void_p(pool_host_ptr_val + page_idx_in_pool * page_size)
        page_gpu_va = pool_gpu_va_start + page_idx_in_pool * page_size

        PageType = ctypes.c_ulonglong * (page_size // sizeof_ulonglong)
        page_view = ctypes.cast(page_host_ptr, ctypes.POINTER(PageType))

        for k in range(chase_length):
            next_element_index = (k + 1) % chase_length
            next_element_gpu_va = page_gpu_va + next_element_index * sizeof_ulonglong
            try:
                page_view.contents[k] = next_element_gpu_va
            except IndexError:
                print(f"Warning: Index {k} out of bounds for page view during init. Page index {page_idx_in_pool}")
                break

    # Sync after writing all pages
    CUDA_RT_CHECK(libcudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize Chase Init")
    print("Pointer chase pages initialized.")

# --- Simplified Test ---
def run_simple_test():
    if not load_cuda_libs(): exit(1)
    if not libcuda or not libcudart: exit(1)

    print("Running simplified covert channel test...")
    if not compile_kernels_to_ptx(): exit(1)

    cu_context = None
    pool_gpu_va_start = 0

    try:
        CU_CHECK(libcuda.cuInit(0), "cuInit")
        cu_device = ctypes.c_int()
        CU_CHECK(libcuda.cuDeviceGet(ctypes.byref(cu_device), GPU_ID), "cuDeviceGet")
        cu_context_ptr = ctypes.c_void_p()
        CU_CHECK(libcuda.cuCtxCreate_v2(ctypes.byref(cu_context_ptr), 0, cu_device), "cuCtxCreate")
        cu_context = cu_context_ptr
        print("CUDA Context created.")

        pool_gpu_va_start, pages0_indices, pages1_indices, pool_size_bytes_actual = find_eviction_pages_driver(N_PAGES, PAGE_SIZE, ALLOC_POOL_SIZE_MB)
        if pool_gpu_va_start is None: raise RuntimeError("Page finding failed")

        all_indices = list(set(pages0_indices + pages1_indices))
        initialize_chase_pages_driver(pool_gpu_va_start, all_indices, len(all_indices))

        # Load PTX & Get kernel functions
        cu_module_ptr = ctypes.c_void_p()
        with open(KERNELS_PTX, 'rb') as f: ptx_content = f.read()
        ptx_content_null_terminated = ctypes.create_string_buffer(ptx_content + b'\x00')
        CU_CHECK(libcuda.cuModuleLoadData(ctypes.byref(cu_module_ptr), ptx_content_null_terminated), "cuModuleLoadData")
        cu_module = cu_module_ptr
        
        sender_kernel_func = ctypes.c_void_p()
        receiver_kernel_func = ctypes.c_void_p()
        CU_CHECK(libcuda.cuModuleGetFunction(ctypes.byref(sender_kernel_func), cu_module, b"sender_contention_kernel"), "cuModuleGetFunction sender")
        CU_CHECK(libcuda.cuModuleGetFunction(ctypes.byref(receiver_kernel_func), cu_module, b"receiver_probe_kernel"), "cuModuleGetFunction receiver")

        print("Kernels loaded successfully.")

        # Test simple receiver kernel launch
        print("Testing receiver kernel...")
        flags = 1 # CU_MEM_ATTACH_GLOBAL
        
        # Allocate memory for receiver
        results_buffer_gpu_ptr = ctypes.c_ulonglong(0)
        d_page_vas0_ptr = ctypes.c_ulonglong(0)
        d_page_vas1_ptr = ctypes.c_ulonglong(0)
        
        max_samples = 100  # Small number for testing
        results_buffer_size = max_samples * 2 * ctypes.sizeof(ctypes.c_ulonglong)
        CU_CHECK(libcuda.cuMemAllocManaged(ctypes.byref(results_buffer_gpu_ptr), results_buffer_size, flags), "cuMemAllocManaged results")
        CU_CHECK(libcuda.cuMemAllocManaged(ctypes.byref(d_page_vas0_ptr), N_PAGES * ctypes.sizeof(ctypes.c_ulonglong), flags), "cuMemAllocManaged page_vas0")
        CU_CHECK(libcuda.cuMemAllocManaged(ctypes.byref(d_page_vas1_ptr), N_PAGES * ctypes.sizeof(ctypes.c_ulonglong), flags), "cuMemAllocManaged page_vas1")

        # Convert page indices to actual page VAs
        PageVAsType = ctypes.c_ulonglong * N_PAGES
        h_page_vas0 = PageVAsType()
        h_page_vas1 = PageVAsType()
        for i in range(N_PAGES):
            h_page_vas0[i] = pool_gpu_va_start + pages0_indices[i] * PAGE_SIZE
            h_page_vas1[i] = pool_gpu_va_start + pages1_indices[i] * PAGE_SIZE
        
        CU_CHECK(libcuda.cuMemcpyHtoD_v2(d_page_vas0_ptr.value, ctypes.byref(h_page_vas0), ctypes.sizeof(h_page_vas0)), "cuMemcpyHtoD page_vas0")
        CU_CHECK(libcuda.cuMemcpyHtoD_v2(d_page_vas1_ptr.value, ctypes.byref(h_page_vas1), ctypes.sizeof(h_page_vas1)), "cuMemcpyHtoD page_vas1")

        # Prepare launch args for receiver
        d_page_vas0_val = ctypes.c_ulonglong(d_page_vas0_ptr.value)
        d_page_vas1_val = ctypes.c_ulonglong(d_page_vas1_ptr.value)
        n_pages_val = ctypes.c_int(N_PAGES)
        results_buffer_val = ctypes.c_ulonglong(results_buffer_gpu_ptr.value)
        max_samples_val = ctypes.c_int(max_samples)

        launch_args = [
            ctypes.byref(d_page_vas0_val),
            ctypes.byref(d_page_vas1_val),
            ctypes.byref(n_pages_val),
            ctypes.byref(results_buffer_val),
            ctypes.byref(max_samples_val)
        ]
        packed_args = (ctypes.c_void_p * len(launch_args))(*[ctypes.cast(p, ctypes.c_void_p) for p in launch_args])

        print("Launching receiver kernel for 2 seconds...")
        start_time = time.time()
        CU_CHECK(libcuda.cuLaunchKernel(receiver_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, packed_args, None), "cuLaunchKernel receiver")
        
        # Let it run for a short time
        time.sleep(2)
        
        print("Synchronizing...")
        CU_CHECK(libcuda.cuStreamSynchronize(None), "cuStreamSynchronize")
        end_time = time.time()
        duration = end_time - start_time
        print(f"Receiver kernel completed in {duration:.2f}s")

        # Read results
        ResultsArrayType = ctypes.c_ulonglong * (max_samples * 2)
        results_ptr = ctypes.cast(results_buffer_gpu_ptr.value, ctypes.POINTER(ResultsArrayType))
        results_host = np.frombuffer(results_ptr.contents, dtype=np.uint64).copy()
        
        # Find valid results
        nonzero_count = np.count_nonzero(results_host)
        print(f"Found {nonzero_count} non-zero timing values")
        
        if nonzero_count > 0:
            valid_results = results_host[results_host != 0][:20]  # First 20 non-zero values
            print(f"Sample timing values: {valid_results}")
            print("SUCCESS: Receiver kernel is working and collecting timing data!")
        else:
            print("WARNING: No timing data collected. This might indicate a kernel issue.")

        # Cleanup
        print("Cleaning up...")
        if d_page_vas0_ptr.value != 0:
            try: CU_CHECK(libcuda.cuMemFree_v2(d_page_vas0_ptr.value), "cuMemFree page_vas0")
            except Exception: pass
        if d_page_vas1_ptr.value != 0:
            try: CU_CHECK(libcuda.cuMemFree_v2(d_page_vas1_ptr.value), "cuMemFree page_vas1")
            except Exception: pass
        if results_buffer_gpu_ptr.value != 0:
            try: CU_CHECK(libcuda.cuMemFree_v2(results_buffer_gpu_ptr.value), "cuMemFree results")
            except Exception: pass

    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        if pool_gpu_va_start != 0:
            try: CU_CHECK(libcuda.cuMemFree_v2(pool_gpu_va_start), "cuMemFree Pool")
            except Exception: pass
        if cu_context and cu_context.value:
            try: CU_CHECK(libcuda.cuCtxDestroy_v2(cu_context), "cuCtxDestroy")
            except Exception: pass
        print("Test completed.")

if __name__ == "__main__":
    run_simple_test()
