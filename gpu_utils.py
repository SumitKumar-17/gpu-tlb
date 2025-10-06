import ctypes
import os
import subprocess
import numpy as np
from l3_hash_rtx3060 import get_l3_set_index_rtx3060_64k

# --- Globals for Lib Handles ---
libcuda = None
libcudart = None

# --- Constants ---
PAGE_SIZE = 64 * 1024
CHASE_LENGTH = 8 # Length of pointer chain for timing
# Removed HUGE_CHUNK_SIZE, BASE_ADDR, STRIDE_SIZE

# --- API Check Functions ---
def CU_CHECK(err_code, func_name):
    """Checks CUDA Driver API error code."""
    if err_code != 0: # CUDA_SUCCESS = 0
        raise RuntimeError(f"CUDA Driver API Error in {func_name}: Code {err_code}")

def CUDA_RT_CHECK(err_code, func_name):
    """Checks CUDA Runtime API error code."""
    if err_code != 0: # cudaSuccess = 0
        err_str = f"Error Code {err_code}"
        try:
            if libcudart: # Check if runtime lib was loaded
                libcudart.cudaGetErrorString.restype = ctypes.c_char_p
                libcudart.cudaGetErrorString.argtypes = [ctypes.c_int]
                err_str = libcudart.cudaGetErrorString(err_code).decode('utf-8')
        except Exception: pass
        raise RuntimeError(f"CUDA Runtime API Error in {func_name}: {err_str}")

# --- Library Loading ---
def load_cuda_libs():
    """Loads CUDA Driver and Runtime libraries and returns handles."""
    global libcuda, libcudart
    local_driver_lib = None; local_runtime_lib = None
    driver_ok = False

    # Load Driver API
    if libcuda is None:
        try:
            local_driver_lib = ctypes.CDLL("libcuda.so")
            print(f"libcuda.so loaded in process {os.getpid()}.")
            # Define Driver API prototypes
            local_driver_lib.cuInit.argtypes = [ctypes.c_uint]; local_driver_lib.cuInit.restype = int
            local_driver_lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]; local_driver_lib.cuDeviceGet.restype = int
            local_driver_lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]; local_driver_lib.cuCtxCreate_v2.restype = int
            local_driver_lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]; local_driver_lib.cuCtxDestroy_v2.restype = int
            local_driver_lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]; local_driver_lib.cuModuleLoadData.restype = int
            local_driver_lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]; local_driver_lib.cuModuleGetFunction.restype = int
            local_driver_lib.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
            local_driver_lib.cuLaunchKernel.restype = int
            local_driver_lib.cuStreamSynchronize.argtypes = [ctypes.c_void_p]; local_driver_lib.cuStreamSynchronize.restype = int
            local_driver_lib.cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t, ctypes.c_uint]; local_driver_lib.cuMemAllocManaged.restype = int
            local_driver_lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]; local_driver_lib.cuMemAlloc_v2.restype = int
            local_driver_lib.cuMemFree_v2.argtypes = [ctypes.c_ulonglong]; local_driver_lib.cuMemFree_v2.restype = int
            local_driver_lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_size_t]; local_driver_lib.cuMemcpyHtoD_v2.restype = int
            local_driver_lib.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_ulonglong]; local_driver_lib.cuPointerGetAttribute.restype = int
            libcuda = local_driver_lib # Assign to global
            driver_ok = True
        except OSError: print(f"Error: libcuda.so not found in process {os.getpid()}."); libcuda = None; driver_ok = False
    else: local_driver_lib = libcuda; driver_ok = True # Already loaded

    # Load Runtime API
    if libcudart is None:
        try:
            local_runtime_lib = ctypes.CDLL("libcudart.so")
            print(f"libcudart.so loaded in process {os.getpid()}.")
            local_runtime_lib.cudaGetErrorString.restype = ctypes.c_char_p; local_runtime_lib.cudaGetErrorString.argtypes = [ctypes.c_int]
            local_runtime_lib.cudaDeviceSynchronize.restype = int
            libcudart = local_runtime_lib # Assign to global
        except OSError: print(f"Warning: libcudart.so not found in process {os.getpid()}."); libcudart = None
    else: local_runtime_lib = libcudart # Already loaded

    return driver_ok, local_driver_lib, local_runtime_lib

# --- Compilation ---
def compile_kernels_to_ptx(cu_file, ptx_file):
    """Compiles a .cu file to PTX targeting sm_86."""
    print(f"Compiling {cu_file} to {ptx_file}...")
    try:
        subprocess.run(['nvcc', '-O3', '--ptx', '-o', ptx_file, cu_file, '-gencode', 'arch=compute_86,code=sm_86'], check=True)
        print("PTX compilation successful.")
        return True
    except Exception as e: print(f"PTX Compilation failed: {e}"); return False

# --- Page Finding (Indices within buffer) ---
def find_page_indices_in_buffer(local_libcuda, pool_gpu_va_start, pool_size_bytes, target_set, n_pages_needed):
    """Finds indices of pages within an allocated buffer that hash to the target_set."""
    if not local_libcuda: raise RuntimeError("libcuda handle required for find_page_indices")
    print(f"Scanning buffer {hex(pool_gpu_va_start)} for {n_pages_needed} pages hashing to set {target_set}...")
    page_indices = []
    num_pages_in_pool = pool_size_bytes // PAGE_SIZE

    for i in range(num_pages_in_pool):
        current_gpu_va = pool_gpu_va_start + i * PAGE_SIZE
        try:
            l3_set = get_l3_set_index_rtx3060_64k(current_gpu_va)
            if l3_set == target_set:
                page_indices.append(i) # Store index relative to pool start
                if len(page_indices) == n_pages_needed:
                    print(f"Found {n_pages_needed} indices for set {target_set}.")
                    return page_indices
        except Exception: continue

    print(f"Warning: Only found {len(page_indices)}/{n_pages_needed} pages for set {target_set}.")
    return page_indices # Return whatever was found

# --- Pointer Chase Initialization (Accepts list of actual VAs) ---
def initialize_chase_pages_va(local_libcuda, local_libcudart, page_va_list):
    """Initializes pages specified by a list of VAs for pointer chasing."""
    if not local_libcuda: raise RuntimeError("libcuda handle needed for pointer attribute")
    if not local_libcudart: raise RuntimeError("libcudart handle needed for sync")

    print(f"Initializing {len(page_va_list)} pages (by VA) for pointer chasing...")
    sizeof_ulonglong = ctypes.sizeof(ctypes.c_ulonglong)

    for page_gpu_va in page_va_list:
        page_host_ptr_attr = ctypes.c_void_p()
        try:
            CU_CHECK(local_libcuda.cuPointerGetAttribute(ctypes.byref(page_host_ptr_attr), 4, page_gpu_va), # 4=HOST_POINTER
                     f"initialize_chase cuPointerGetAttribute VA {hex(page_gpu_va)}")
            if not page_host_ptr_attr.value: print(f"Warning: Could not get host pointer for VA {hex(page_gpu_va)}, skipping init."); continue
            page_host_ptr = page_host_ptr_attr.value # Get the integer address
            PageType = ctypes.c_ulonglong * (PAGE_SIZE // sizeof_ulonglong)
            page_view = ctypes.cast(page_host_ptr, ctypes.POINTER(PageType))
            for k in range(CHASE_LENGTH):
                next_element_index = (k + 1) % CHASE_LENGTH
                next_element_gpu_va = page_gpu_va + next_element_index * sizeof_ulonglong
                try: page_view.contents[k] = next_element_gpu_va
                except IndexError: print(f"Warning: Chase init index error VA {hex(page_gpu_va)}, k={k}"); break
                except Exception as write_e: print(f"Warning: Error writing chase init VA {hex(page_gpu_va)}, k={k}: {write_e}"); break
        except Exception as e: print(f"Warning: Failed getting host pointer or writing for VA {hex(page_gpu_va)} ({e}), skipping init.")

    CUDA_RT_CHECK(local_libcudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize Chase Init")
    print("Pointer chase page initialization attempt complete.")