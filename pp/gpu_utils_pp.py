import ctypes
import os
import subprocess
import numpy as np
from l3_hash_rtx3060 import get_l3_set_index_rtx3060_64k # Assumes this file exists

# --- Globals for Lib Handles ---
libcuda = None
libcudart = None

# --- Constants ---
PAGE_SIZE = 64 * 1024
# PRIME_PROBE_N = 9 # Number of pages in eviction set (Ways + 1 for 8-way L3)
PRIME_PROBE_N = 17 # Let's try a slightly larger set based on L1/L2 capacity observed before
L3_FILL_COUNT = 20000 # For flushing L3 during miss measurement
L1_L2_EVICT_COUNT = 35 # Number of pages for L1/L2 eviction set (e.g., >= 16 for L1d)

# --- API Check Functions ---
def CU_CHECK(err_code, func_name):
    """Checks CUDA Driver API error code."""
    if err_code != 0: raise RuntimeError(f"CUDA Driver API Error in {func_name}: Code {err_code}")

def CUDA_RT_CHECK(err_code, func_name):
    """Checks CUDA Runtime API error code."""
    if err_code != 0:
        err_str = f"Error Code {err_code}"
        try:
            if libcudart:
                libcudart.cudaGetErrorString.restype = ctypes.c_char_p
                libcudart.cudaGetErrorString.argtypes = [ctypes.c_int]
                err_str = libcudart.cudaGetErrorString(err_code).decode('utf-8')
        except Exception: pass
        raise RuntimeError(f"CUDA Runtime API Error in {func_name}: {err_str}")

# --- Library Loading ---
def load_cuda_libs_pp():
    """Loads CUDA Driver and Runtime libraries and returns handles."""
    global libcuda, libcudart
    local_driver_lib = None; local_runtime_lib = None
    driver_ok = False

    # Load Driver API
    if libcuda is None:
        try:
            local_driver_lib = ctypes.CDLL("libcuda.so")
            print(f"libcuda.so loaded in process {os.getpid()}.")
            # Define Driver API prototypes needed
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
            # NO cuPointerGetAttribute needed if not initializing chase pages
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
def compile_kernels_to_ptx_pp(cu_file, ptx_file):
    """Compiles a .cu file to PTX targeting sm_86."""
    print(f"Compiling {cu_file} to {ptx_file}...")
    try:
        subprocess.run(['nvcc', '-O3', '--ptx', '-o', ptx_file, cu_file, '-gencode', 'arch=compute_86,code=sm_86'], check=True, capture_output=True, text=True)
        print("PTX compilation successful.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"PTX Compilation failed:\nSTDERR:\n{e.stderr}\nSTDOUT:\n{e.stdout}")
        return False
    except FileNotFoundError:
        print("Error: 'nvcc' command not found. Is CUDA Toolkit installed and in PATH?")
        return False
    except Exception as e:
        print(f"PTX Compilation failed with unexpected error: {e}")
        return False


# --- Page Finding (VAs within buffer by Hashing) ---
def find_pages_for_set(local_libcuda, pool_gpu_va_start, pool_size_bytes, target_set, n_pages_needed):
    """Finds VAs of pages within an allocated buffer that hash to the target_set."""
    if not local_libcuda: raise RuntimeError("libcuda handle required")
    print(f"Scanning buffer {hex(pool_gpu_va_start)} for {n_pages_needed} pages hashing to set {target_set}...")
    target_vas = []
    num_pages_in_pool = pool_size_bytes // PAGE_SIZE
    for i in range(num_pages_in_pool):
        current_gpu_va = pool_gpu_va_start + i * PAGE_SIZE
        try:
            l3_set = get_l3_set_index_rtx3060_64k(current_gpu_va)
            if l3_set == target_set:
                target_vas.append(current_gpu_va)
                if len(target_vas) == n_pages_needed:
                    print(f"Found {n_pages_needed} VAs for set {target_set}.")
                    return target_vas
        except Exception: continue # Skip hashing errors

    print(f"Warning: Only found {len(target_vas)}/{n_pages_needed} VAs for set {target_set}.")
    return target_vas