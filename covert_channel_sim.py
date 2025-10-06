import subprocess
import multiprocessing
from multiprocessing import set_start_method # Ensure this is imported
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
STRIDE_SIZE = 1 * 1024 * 1024 # Stride used during allocation
ALLOC_POOL_SIZE_MB = 256 # Size of initial memory pool to find pages

# Channel Parameters (Tune these!)
DELAY_MS_SENDER = 20  # Host sleep delay (ms) for sender between sending bits
THRESHOLD_T = 1504    # Timing threshold from baseline (e.g., 1460 cycles)
NUM_BITS = 64         # Number of bits to transmit
RECEIVER_SAMPLES_PER_BIT = 30 # How many (t0, t1) samples receiver tries per expected bit duration

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
            libcudart.cudaMallocManaged.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]; libcudart.cudaMallocManaged.restype = int
            libcudart.cudaFree.argtypes = [ctypes.c_void_p]; libcudart.cudaFree.restype = int
            libcudart.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]; libcudart.cudaHostGetDevicePointer.restype = int
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
    # Assumes Driver API (libcuda) is initialized in the calling scope (parent)
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

    # Find two distinct sets (same logic as before)
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
        # Return the STARTING GPU VA of the pool and the lists of indices
        return pool_gpu_va_start, set0_indices, set1_indices, pool_size_bytes
    else: print("Error: Could not find two distinct L3 sets."); libcuda.cuMemFree_v2(pool_gpu_va_start); return None, None, None, 0


# --- Helper Function: Initialize Pages for Pointer Chasing (Driver API Context) ---
def initialize_chase_pages_driver(pool_gpu_va_start, page_indices, n_pages):
    """ Writes pointer chain setup into managed memory using host pointer obtained via Driver API """
    if not libcuda: raise RuntimeError("libcuda needed for pointer attribute")
    if not libcudart: raise RuntimeError("libcudart needed for sync") # Still use Runtime sync for simplicity

    print(f"Initializing {len(page_indices)} pages for pointer chasing...") # Use actual number of pages passed
    chase_length = 8
    page_size = PAGE_SIZE
    sizeof_ulonglong = ctypes.sizeof(ctypes.c_ulonglong)

    # Get Host pointer corresponding to the start GPU VA
    pool_host_ptr_attr = ctypes.c_void_p()
    CU_CHECK(libcuda.cuPointerGetAttribute(ctypes.byref(pool_host_ptr_attr), 4, pool_gpu_va_start), # 4=HOST_POINTER
             "initialize_chase cuPointerGetAttribute HOST_POINTER")
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
                page_view.contents[k] = next_element_gpu_va # Write GPU VA
            except IndexError:
                print(f"Warning: Index {k} out of bounds for page view during init. Page index {page_idx_in_pool}")
                break # Avoid writing out of bounds

    # Sync after writing all pages
    CUDA_RT_CHECK(libcudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize Chase Init")
    print("Pointer chase pages initialized.")


# --- Worker Functions ---

# --- Worker Functions ---

def sender_worker(bit_q, pool_gpu_va_start, pages0_indices, pages1_indices, n_pages, delay_ms, stop_event, gpu_id):
    # Load libs within worker
    if not load_cuda_libs() or not libcuda:
        print("[Sender] Error: Failed to load CUDA libraries in worker.")
        return
    print("[Sender] Process started")
    sender_cu_context = None; cu_module = None
    stop_flag_gpu_ptr = ctypes.c_ulonglong(0) # Use managed alloc now
    d_indices_ptr = ctypes.c_ulonglong(0)     # Use managed alloc now
    sender_kernel_func = ctypes.c_void_p()

    try:
        # Init Driver API
        CU_CHECK(libcuda.cuInit(0), "Sender cuInit")
        sender_cu_device = ctypes.c_int()
        CU_CHECK(libcuda.cuDeviceGet(ctypes.byref(sender_cu_device), gpu_id), "Sender cuDeviceGet")
        sender_cu_context_ptr = ctypes.c_void_p()
        # Create context, don't explicitly push
        CU_CHECK(libcuda.cuCtxCreate_v2(ctypes.byref(sender_cu_context_ptr), 0, sender_cu_device), "Sender cuCtxCreate")
        sender_cu_context = sender_cu_context_ptr
        print("[Sender] CUDA Context created.")

        # Load PTX & Kernel
        cu_module_ptr = ctypes.c_void_p()
        with open(KERNELS_PTX, 'rb') as f: ptx_content = f.read()
        ptx_content_null_terminated = ctypes.create_string_buffer(ptx_content + b'\x00')
        CU_CHECK(libcuda.cuModuleLoadData(ctypes.byref(cu_module_ptr), ptx_content_null_terminated), "Sender cuModuleLoadData")
        cu_module = cu_module_ptr
        CU_CHECK(libcuda.cuModuleGetFunction(ctypes.byref(sender_kernel_func), cu_module, b"sender_contention_kernel"), "Sender cuModuleGetFunction")

        # Allocate GPU Memory using Managed Alloc
        flags = 1 # CU_MEM_ATTACH_GLOBAL
        CU_CHECK(libcuda.cuMemAllocManaged(ctypes.byref(stop_flag_gpu_ptr), ctypes.sizeof(ctypes.c_int), flags), "Sender cuMemAllocManaged stop_flag")
        CU_CHECK(libcuda.cuMemAllocManaged(ctypes.byref(d_indices_ptr), n_pages * ctypes.sizeof(ctypes.c_int), flags), "Sender cuMemAllocManaged indices")

        # Initialize Stop Flag
        stop_val_cpu = np.array([0], dtype=np.int32)
        CU_CHECK(libcuda.cuMemcpyHtoD_v2(stop_flag_gpu_ptr.value, # Dest GPU Pointer Value
                                         stop_val_cpu.ctypes.data_as(ctypes.c_void_p), # Src Host Pointer
                                         stop_val_cpu.nbytes), # Size
                 "Sender cuMemcpyHtoD stop_flag")

        # Prepare Kernel Args (values used repeatedly)
        pool_base_va_val = ctypes.c_ulonglong(pool_gpu_va_start)
        d_indices_val = ctypes.c_ulonglong(d_indices_ptr.value) # This is the GPU VA of the indices buffer
        n_pages_val = ctypes.c_int(n_pages)
        stop_flag_kernel_arg = ctypes.c_ulonglong(stop_flag_gpu_ptr.value) # Kernel needs the GPU VA

        print("[Sender] Setup complete, starting send loop...")
        # --- Send Loop ---
        while not stop_event.is_set():
            try:
                bit = bit_q.get(timeout=0.1)
                indices_to_use = pages0_indices if bit == 0 else pages1_indices
                # Create ctypes array on host to copy from
                IndexType = ctypes.c_int * n_pages
                h_indices = IndexType(*indices_to_use)

                # Copy the correct index list to device (managed) memory
                CU_CHECK(libcuda.cuMemcpyHtoD_v2(d_indices_ptr.value, # Dest GPU Pointer Value
                                                 ctypes.byref(h_indices), # Src Host Pointer (byref for ctypes)
                                                 ctypes.sizeof(h_indices)), # Size
                         f"Sender memcpy indices bit {bit}")

                # Prepare launch params
                launch_args = [
                    ctypes.byref(pool_base_va_val),
                    ctypes.byref(d_indices_val), # Pointer to device ptr value
                    ctypes.byref(n_pages_val),
                    ctypes.byref(stop_flag_kernel_arg) # Pointer to device ptr value
                ]
                packed_args = (ctypes.c_void_p * len(launch_args))(*[ctypes.cast(p, ctypes.c_void_p) for p in launch_args])

                CU_CHECK(libcuda.cuLaunchKernel(sender_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, packed_args, None), f"Sender launch bit {bit}")
                time.sleep(delay_ms / 1000.0) # Host sleep controls duration

            except multiprocessing.queues.Empty: continue
            except Exception as loop_e: print(f"[Sender] Error in loop: {loop_e}"); break

        # --- Post-Loop: Set Stop Flag ---
        print("[Sender] Stop event received. Setting final stop flag.")
        stop_val_cpu = np.array([1], dtype=np.int32)
        try:
             # Use numpy data pointer for memcpy source
             CU_CHECK(libcuda.cuMemcpyHtoD_v2(stop_flag_gpu_ptr.value, # Dest GPU Pointer Value
                                              stop_val_cpu.ctypes.data_as(ctypes.c_void_p), # Src Host Pointer
                                              stop_val_cpu.nbytes), # Size
                      "Sender cuMemcpyHtoD stop_flag_final")
        except Exception as final_e: print(f"[Sender] Error setting final stop flag: {final_e}")

    except Exception as setup_e: print(f"[Sender] Error: {setup_e}")
    finally: # --- Cleanup ---
        print("[Sender] Cleaning up...")
        # Use .value when calling cuMemFree_v2 and correct exception syntax
        if d_indices_ptr.value != 0:
            try:
                CU_CHECK(libcuda.cuMemFree_v2(d_indices_ptr.value), "Sender cuMemFree indices")
            except Exception: # Corrected
                pass
        if stop_flag_gpu_ptr.value != 0:
            try:
                CU_CHECK(libcuda.cuMemFree_v2(stop_flag_gpu_ptr.value), "Sender cuMemFree stop_flag")
            except Exception: # Corrected
                pass
        # Context destroy takes the handle
        if sender_cu_context and sender_cu_context.value:
            try:
                CU_CHECK(libcuda.cuCtxDestroy_v2(sender_cu_context), "Sender cuCtxDestroy")
                print("[Sender] CUDA Context destroyed.")
            except Exception as ctx_e:
                 print(f"[Sender] Error destroying context: {ctx_e}")
        print("[Sender] Process finished.")


def receiver_worker(result_q, pool_gpu_va_start, pages0_indices, pages1_indices, n_pages, num_bits, samples_per_bit, stop_event, gpu_id):
    # Load libs within worker
    if not load_cuda_libs() or not libcuda:
        print("[Receiver] Error: Failed to load CUDA libraries in worker.")
        return
    print("[Receiver] Process started")
    receiver_cu_context = None; cu_module = None
    stop_flag_gpu_ptr = ctypes.c_ulonglong(0) # Use managed alloc
    results_buffer_gpu_va = 0 # Keep results managed, store VA
    results_buffer_gpu_ptr = ctypes.c_ulonglong(0) # Store pointer object
    d_indices0_ptr = ctypes.c_ulonglong(0) # Use managed alloc
    d_indices1_ptr = ctypes.c_ulonglong(0) # Use managed alloc
    receiver_kernel_func = ctypes.c_void_p(); results_host = None

    try:
        # Init Driver API
        CU_CHECK(libcuda.cuInit(0), "Receiver cuInit"); receiver_cu_device = ctypes.c_int()
        CU_CHECK(libcuda.cuDeviceGet(ctypes.byref(receiver_cu_device), gpu_id), "Receiver cuDeviceGet"); receiver_cu_context_ptr = ctypes.c_void_p()
        # Create context, don't explicitly push
        CU_CHECK(libcuda.cuCtxCreate_v2(ctypes.byref(receiver_cu_context_ptr), 0, receiver_cu_device), "Receiver cuCtxCreate"); receiver_cu_context = receiver_cu_context_ptr
        print("[Receiver] CUDA Context created.")

        # Load PTX & Kernel
        cu_module_ptr = ctypes.c_void_p();
        with open(KERNELS_PTX, 'rb') as f: ptx_content = f.read()
        ptx_content_null_terminated = ctypes.create_string_buffer(ptx_content + b'\x00')
        CU_CHECK(libcuda.cuModuleLoadData(ctypes.byref(cu_module_ptr), ptx_content_null_terminated), "Receiver cuModuleLoadData")
        cu_module = cu_module_ptr
        CU_CHECK(libcuda.cuModuleGetFunction(ctypes.byref(receiver_kernel_func), cu_module, b"receiver_probe_kernel"), "Receiver cuModuleGetFunction")

        # Allocate GPU Memory using Managed Alloc
        flags = 1 # CU_MEM_ATTACH_GLOBAL
        max_total_samples = num_bits * samples_per_bit * 2 * 2
        results_buffer_size = max_total_samples * ctypes.sizeof(ctypes.c_ulonglong)
        CU_CHECK(libcuda.cuMemAllocManaged(ctypes.byref(results_buffer_gpu_ptr), results_buffer_size, flags), "Receiver cuMemAllocManaged results")
        results_buffer_gpu_va = results_buffer_gpu_ptr.value
        if not results_buffer_gpu_va: raise RuntimeError("Failed results buffer alloc")

        CU_CHECK(libcuda.cuMemAllocManaged(ctypes.byref(stop_flag_gpu_ptr), ctypes.sizeof(ctypes.c_int), flags), "Receiver cuMemAllocManaged stop_flag")
        CU_CHECK(libcuda.cuMemAllocManaged(ctypes.byref(d_indices0_ptr), n_pages * ctypes.sizeof(ctypes.c_int), flags), "Receiver cuMemAllocManaged indices0")
        CU_CHECK(libcuda.cuMemAllocManaged(ctypes.byref(d_indices1_ptr), n_pages * ctypes.sizeof(ctypes.c_int), flags), "Receiver cuMemAllocManaged indices1")

        # Initialize Stop Flag & Copy Index Lists
        stop_val_cpu = np.array([0], dtype=np.int32)
        CU_CHECK(libcuda.cuMemcpyHtoD_v2(stop_flag_gpu_ptr.value, # Use .value
                                         stop_val_cpu.ctypes.data_as(ctypes.c_void_p),
                                         stop_val_cpu.nbytes),
                 "Receiver cuMemcpyHtoD stop_flag")
        IndexType = ctypes.c_int * n_pages
        h_indices0 = IndexType(*pages0_indices); h_indices1 = IndexType(*pages1_indices)
        CU_CHECK(libcuda.cuMemcpyHtoD_v2(d_indices0_ptr.value, ctypes.byref(h_indices0), ctypes.sizeof(h_indices0)), "Receiver cuMemcpyHtoD indices0") # Use .value
        CU_CHECK(libcuda.cuMemcpyHtoD_v2(d_indices1_ptr.value, ctypes.byref(h_indices1), ctypes.sizeof(h_indices1)), "Receiver cuMemcpyHtoD indices1") # Use .value

        # Prepare Kernel Launch Args
        pool_base_va_val = ctypes.c_ulonglong(pool_gpu_va_start)
        d_indices0_val = ctypes.c_ulonglong(d_indices0_ptr.value)
        d_indices1_val = ctypes.c_ulonglong(d_indices1_ptr.value)
        n_pages_val = ctypes.c_int(n_pages)
        results_buffer_val = ctypes.c_ulonglong(results_buffer_gpu_va) # Pass GPU VA
        max_samples_val = ctypes.c_int(max_total_samples // 2)
        stop_flag_kernel_arg = ctypes.c_ulonglong(stop_flag_gpu_ptr.value) # Pass GPU VA

        launch_args = [
            ctypes.byref(pool_base_va_val), ctypes.byref(d_indices0_val),
            ctypes.byref(d_indices1_val), ctypes.byref(n_pages_val),
            ctypes.byref(results_buffer_val), ctypes.byref(max_samples_val),
            ctypes.byref(stop_flag_kernel_arg) ]
        packed_args = (ctypes.c_void_p * len(launch_args))(*[ctypes.cast(p, ctypes.c_void_p) for p in launch_args])

        # Launch Kernel & Wait
        print("[Receiver] Launching probe kernel...")
        start_time = time.time()
        CU_CHECK(libcuda.cuLaunchKernel(receiver_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, packed_args, None), "Receiver cuLaunchKernel")
        stop_event.wait()
        print("[Receiver] Stop event received. Setting final stop flag...")
        stop_val_cpu = np.array([1], dtype=np.int32)
        CU_CHECK(libcuda.cuMemcpyHtoD_v2(stop_flag_gpu_ptr.value, # Use .value
                                         stop_val_cpu.ctypes.data_as(ctypes.c_void_p),
                                         stop_val_cpu.nbytes),
                 "Receiver cuMemcpyHtoD stop_flag_final")
        CU_CHECK(libcuda.cuStreamSynchronize(None), "Receiver cuStreamSynchronize")
        end_time = time.time()
        duration = end_time - start_time
        print(f"[Receiver] Kernel finished in {duration:.2f}s. Preparing results...")

        # Get Results
        ResultsArrayType = ctypes.c_ulonglong * max_total_samples
        results_ptr = ctypes.cast(results_buffer_gpu_va, ctypes.POINTER(ResultsArrayType))
        results_host = np.frombuffer(results_ptr.contents, dtype=np.uint64).copy()

        result_q.put((results_host, duration))
        print("[Receiver] Results sent to queue.")

    except Exception as setup_e: print(f"[Receiver] Error: {setup_e}")
    finally: # --- Cleanup ---
        print("[Receiver] Cleaning up...")
        # Use .value when calling cuMemFree_v2 and correct exception syntax
        if d_indices0_ptr.value != 0:
            try:
                CU_CHECK(libcuda.cuMemFree_v2(d_indices0_ptr.value), "Receiver cuMemFree indices0")
            except Exception: # Corrected
                pass
        if d_indices1_ptr.value != 0:
            try:
                CU_CHECK(libcuda.cuMemFree_v2(d_indices1_ptr.value), "Receiver cuMemFree indices1")
            except Exception: # Corrected
                pass
        if results_buffer_gpu_va != 0: # results_buffer_gpu_va already holds the value
            try:
                CU_CHECK(libcuda.cuMemFree_v2(results_buffer_gpu_va), "Receiver cuMemFree results")
            except Exception: # Corrected
                pass
        if stop_flag_gpu_ptr.value != 0: # Use updated pointer name
            try:
                CU_CHECK(libcuda.cuMemFree_v2(stop_flag_gpu_ptr.value), "Receiver cuMemFree stop_flag")
            except Exception: # Corrected
                pass
        # Context destroy takes the handle
        if receiver_cu_context and receiver_cu_context.value:
             try:
                 CU_CHECK(libcuda.cuCtxDestroy_v2(receiver_cu_context), "Receiver cuCtxDestroy")
                 print("[Receiver] CUDA Context destroyed.")
             except Exception as ctx_e:
                  print(f"[Receiver] Error destroying context: {ctx_e}")
        print("[Receiver] Process finished.")
# --- Main Execution ---
if __name__ == "__main__":
    try: set_start_method('spawn', force=True); print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e: print(f"Warning: Could not set start method: {e}")

    if not load_cuda_libs(): exit(1)
    if not libcuda or not libcudart: exit(1)

    print("Ensure ASLR is disabled: 'sudo sysctl kernel.randomize_va_space=0'")
    time.sleep(1)

    if not compile_kernels_to_ptx(): exit(1)

    parent_cu_context = None; pool_gpu_va_start = 0
    pages0_indices = None; pages1_indices = None; pool_size_bytes_actual = 0
    initialization_ok = False

    try:
        CU_CHECK(libcuda.cuInit(0), "Parent cuInit"); parent_cu_device = ctypes.c_int()
        CU_CHECK(libcuda.cuDeviceGet(ctypes.byref(parent_cu_device), GPU_ID), "Parent cuDeviceGet"); parent_cu_context_ptr = ctypes.c_void_p()
        CU_CHECK(libcuda.cuCtxCreate_v2(ctypes.byref(parent_cu_context_ptr), 0, parent_cu_device), "Parent cuCtxCreate"); parent_cu_context = parent_cu_context_ptr
        print("Parent Driver Context created.")

        pool_gpu_va_start, pages0_indices, pages1_indices, pool_size_bytes_actual = find_eviction_pages_driver(N_PAGES, PAGE_SIZE, ALLOC_POOL_SIZE_MB)
        if pool_gpu_va_start is None: raise RuntimeError("Page finding failed")

        all_indices = list(set(pages0_indices + pages1_indices))
        initialize_chase_pages_driver(pool_gpu_va_start, all_indices, len(all_indices))
        initialization_ok = True

    except Exception as e: print(f"Error during parent setup: {e}"); initialization_ok = False

    if not initialization_ok:
         print("Parent initialization failed. Cleaning up...")
         if parent_cu_context and parent_cu_context.value:
             try:
                 CU_CHECK(libcuda.cuCtxDestroy_v2(parent_cu_context), "Parent CtxDestroy on Error")
             except Exception: # Corrected
                 pass
         if pool_gpu_va_start != 0:
             try:
                 CU_CHECK(libcuda.cuMemFree_v2(pool_gpu_va_start), "Parent MemFree on Error")
             except Exception: # Corrected
                 pass
         exit(1)

    bit_queue = multiprocessing.Queue(); result_queue = multiprocessing.Queue(); stop_event = multiprocessing.Event()
    random_bits = [random.randint(0, 1) for _ in range(NUM_BITS)]
    print(f"Generated {NUM_BITS} random bits: {random_bits[:20]}...")

    sender = multiprocessing.Process(target=sender_worker, args=(bit_queue, pool_gpu_va_start, pages0_indices, pages1_indices, N_PAGES, DELAY_MS_SENDER, stop_event, GPU_ID))
    receiver = multiprocessing.Process(target=receiver_worker, args=(result_queue, pool_gpu_va_start, pages0_indices, pages1_indices, N_PAGES, NUM_BITS, RECEIVER_SAMPLES_PER_BIT, stop_event, GPU_ID))

    print("Starting processes...")
    receiver.start(); time.sleep(2.5); sender.start(); time.sleep(0.5)

    print("Sending bits...")
    for i, bit in enumerate(random_bits): bit_queue.put(bit); time.sleep(DELAY_MS_SENDER / 1000.0 * 1.1)
    print("All bits sent. Waiting for receiver..."); time.sleep(3); stop_event.set()
    sender.join(timeout=10); receiver.join(timeout=15)
    if sender.is_alive(): print("Warning: Sender timeout."); sender.terminate(); sender.join()
    if receiver.is_alive(): print("Warning: Receiver timeout."); receiver.terminate(); receiver.join()

    print("Processing results...")
    try:
        results_raw, total_duration = result_queue.get(timeout=5)
        last_nonzero = np.max(np.where(results_raw != 0)[0]) if np.any(results_raw != 0) else 0
        results_raw_valid = results_raw[:last_nonzero+1]
        if len(results_raw_valid) % 2 != 0: results_raw_valid = results_raw_valid[:-1]
        results_pairs = results_raw_valid.reshape(-1, 2)
        num_samples_received = len(results_pairs)
        print(f"Received {num_samples_received} valid timing samples.")
        if num_samples_received == 0: raise ValueError("No valid samples received")

        decoded_bits = []; samples_per_window = max(1, num_samples_received // NUM_BITS)
        print(f"Decoding with window size: {samples_per_window}")
        for i in range(NUM_BITS):
             start_idx = i * samples_per_window; end_idx = min((i + 1) * samples_per_window, num_samples_received)
             if start_idx >= end_idx: decoded_bits.append(-1); continue
             window = results_pairs[start_idx:end_idx]; t0 = window[:, 0]; t1 = window[:, 1]
             m0 = np.sum(t0 > THRESHOLD_T); m1 = np.sum(t1 > THRESHOLD_T); h0 = len(window) - m0; h1 = len(window) - m1
             if m0 > h0 and m0 > m1 + max(1, len(window)*0.1): decoded_bits.append(0)
             elif m1 > h1 and m1 > m0 + max(1, len(window)*0.1): decoded_bits.append(1)
             else: decoded_bits.append(0 if m0 >= m1 else 1)

        correct = 0; valid = 0
        for i in range(min(len(random_bits), len(decoded_bits))):
            if decoded_bits[i] != -1: valid += 1; correct += (random_bits[i] == decoded_bits[i])
        accuracy = (correct / valid) * 100 if valid > 0 else 0
        bandwidth = (valid / total_duration) if total_duration > 0 else 0

        print("\n--- Results ---")
        print(f"Sent {len(random_bits)} bits."); print(f"Decoded {valid} bits."); print(f"Correct: {correct}/{valid}")
        print(f"Accuracy: {accuracy:.2f}%"); print(f"Receiver Duration: {total_duration:.2f} s")
        print(f"Estimated Bandwidth: {bandwidth:.2f} bps ({bandwidth/1000:.2f} kbps)")

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 5)); plt.plot(results_pairs[:, 0], label='t0', alpha=0.7); plt.plot(results_pairs[:, 1], label='t1', alpha=0.7); plt.axhline(THRESHOLD_T, color='r', linestyle='--', label=f'Threshold ({THRESHOLD_T})'); plt.title('Covert Channel Receiver Timing'); plt.xlabel('Sample Index'); plt.ylabel('Cycles'); plt.legend(); plt.ylim(bottom=0, top=max(2000, np.max(results_pairs)*1.1 if num_samples_received > 0 else 2000)); plt.grid(True); plt.savefig("covert_channel_timing.png")
            print("Saved timing plot to covert_channel_timing.png")
        except ImportError: print("matplotlib not found, skipping plot.")
        except Exception as plot_e: print(f"Error plotting: {plot_e}")

    except (multiprocessing.queues.Empty, ValueError, IndexError) as qe: print(f"Error: No results or processing failed: {qe}")
    except Exception as e: print(f"An unexpected error processing results: {e}")

    print("Cleaning up parent...")
    if pool_gpu_va_start != 0:
        try:
            CU_CHECK(libcuda.cuMemFree_v2(pool_gpu_va_start), "Parent cuMemFree Pool")
            print("Memory pool freed.")
        except Exception as e: # Corrected
             print(f"Error during pool cleanup: {e}")
    if parent_cu_context and parent_cu_context.value:
        try:
            CU_CHECK(libcuda.cuCtxDestroy_v2(parent_cu_context), "Parent cuCtxDestroy")
            print("Parent context destroyed.")
        except Exception as e: # Corrected
             print(f"Error cleaning parent context: {e}")
    print("Done.")