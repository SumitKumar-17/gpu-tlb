import subprocess
import multiprocessing
import ctypes
import numpy as np
import time
import os

# --- Configuration ---
GPU_ID = 0
PAGE_SIZE = 64 * 1024
STRIDE_SIZE = 1 * 1024 * 1024 # Stride between allocated pages (like examples)
TARGET_PAGE_COUNT = 1       # The single page we time
L1_L2_EVICT_COUNT = 32      # Number of pages to access before target (perturb L1/L2) >= 16 for L1d
# <<< --- Increased L3_FILL_COUNT --- >>>
L3_FILL_COUNT = 20000        # Number of pages to flush L3 (Increased)
# <<< ------------------------------- >>>
TOTAL_PAGES_NEEDED = TARGET_PAGE_COUNT + L1_L2_EVICT_COUNT + L3_FILL_COUNT
NUM_TIMING_SAMPLES = 500    # How many times to measure hit/miss latency

KERNELS_FILE = "baseline_kernels.cu"
KERNELS_PTX = "baseline_kernels.ptx"

# --- Globals for CUDA Lib Handles ---
libcuda = None
libcudart = None # Keep for error strings maybe

# --- CUDA API Helper Functions ---
def load_cuda_libs():
    global libcuda, libcudart
    try:
        libcuda = ctypes.CDLL("libcuda.so")
        print("libcuda.so loaded.")
        # Define Driver API prototypes needed
        libcuda.cuInit.argtypes = [ctypes.c_uint]; libcuda.cuInit.restype = int
        libcuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]; libcuda.cuDeviceGet.restype = int
        libcuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]; libcuda.cuCtxCreate_v2.restype = int
        libcuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]; libcuda.cuCtxDestroy_v2.restype = int
        libcuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]; libcuda.cuModuleLoadData.restype = int
        libcuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]; libcuda.cuModuleGetFunction.restype = int
        libcuda.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        libcuda.cuLaunchKernel.restype = int
        libcuda.cuStreamSynchronize.argtypes = [ctypes.c_void_p]; libcuda.cuStreamSynchronize.restype = int
        libcuda.cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t, ctypes.c_uint] # Returns CUdeviceptr (ulonglong)
        libcuda.cuMemAllocManaged.restype = int
        libcuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t] # For device-only ptr arrays
        libcuda.cuMemAlloc_v2.restype = int
        libcuda.cuMemFree_v2.argtypes = [ctypes.c_ulonglong] # Takes CUdeviceptr
        libcuda.cuMemFree_v2.restype = int
        libcuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_size_t]
        libcuda.cuMemcpyHtoD_v2.restype = int
        # Add definition for cuPointerGetAttribute used for host pointer lookup
        libcuda.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_ulonglong]
        libcuda.cuPointerGetAttribute.restype = int
    except OSError:
        print("Error: libcuda.so not found.")
        libcuda = None
        return False

    # Still load libcudart for error strings AND synchronization
    try:
        libcudart = ctypes.CDLL("libcudart.so")
        libcudart.cudaGetErrorString.restype = ctypes.c_char_p
        libcudart.cudaGetErrorString.argtypes = [ctypes.c_int]
        libcudart.cudaDeviceSynchronize.restype = int # Add this definition
        libcudart.cudaSetDevice.argtypes = [ctypes.c_int] # Add this definition
        libcudart.cudaSetDevice.restype = int
    except OSError:
        libcudart = None
        print("Warning: libcudart.so not found, cannot get detailed error strings or use Runtime Sync.")
        # If libcudart is needed for sync, this might be a problem
        if libcuda: # Check if driver loaded
             print("Proceeding with Driver API only (sync might be needed via Driver API).")
        else:
             return False # Cannot proceed if neither loaded

    return True

# --- Driver API Check ---
def CU_CHECK(err_code, func_name):
    if err_code != 0: # CUDA_SUCCESS = 0
        # TODO: Add Driver API error string lookup if possible (cuGetErrorString)
        raise RuntimeError(f"CUDA Driver API Error in {func_name}: Code {err_code}")

# --- Runtime API Check ---
def CUDA_RT_CHECK(err_code, func_name):
     if err_code != 0: # cudaSuccess = 0
         err_str = f"Error Code {err_code}"
         try:
              if libcudart:
                   err_str = libcudart.cudaGetErrorString(err_code).decode('utf-8')
         except: pass
         raise RuntimeError(f"CUDA Runtime API Error in {func_name}: {err_str}")


# --- Compile Kernels ---
def compile_kernels_to_ptx():
    print(f"Compiling {KERNELS_FILE} to {KERNELS_PTX}...")
    try:
        # Use specific compute capability for RTX 3060 (Ampere)
        subprocess.run(['nvcc', '-O3', '--ptx', '-o', KERNELS_PTX, KERNELS_FILE, '-gencode', 'arch=compute_86,code=sm_86'], check=True)
        print("PTX compilation successful.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"PTX Compilation failed: {e}")
        return False
    except FileNotFoundError:
        print("Error: 'nvcc' command not found. Is CUDA Toolkit installed and in PATH?")
        return False


# --- Main Execution ---
if __name__ == "__main__":
    if not load_cuda_libs() or not libcuda: # Ensure Driver API lib loaded
        exit(1)

    if not compile_kernels_to_ptx():
         exit(1)

    cu_context = None
    cu_module = None
    pool_gpu_ptr_val = 0 # Use 0 as null indicator
    results_buffer_gpu_va = 0
    d_l1_l2_list_ptr = ctypes.c_ulonglong(0)
    d_l3_fill_list_ptr = ctypes.c_ulonglong(0)
    pool_host_ptr = ctypes.c_void_p() # Store host pointer for init

    try:
        # --- Initialize CUDA Driver API ---
        CU_CHECK(libcuda.cuInit(0), "cuInit")
        cu_device = ctypes.c_int()
        CU_CHECK(libcuda.cuDeviceGet(ctypes.byref(cu_device), GPU_ID), "cuDeviceGet")
        cu_context_ptr = ctypes.c_void_p()
        CU_CHECK(libcuda.cuCtxCreate_v2(ctypes.byref(cu_context_ptr), 0, cu_device), "cuCtxCreate")
        cu_context = cu_context_ptr
        print("CUDA Driver Context created.")

        # --- Load PTX Module ---
        cu_module_ptr = ctypes.c_void_p()
        with open(KERNELS_PTX, 'rb') as f: ptx_content = f.read()
        ptx_content_null_terminated = ctypes.create_string_buffer(ptx_content + b'\x00')
        CU_CHECK(libcuda.cuModuleLoadData(ctypes.byref(cu_module_ptr), ptx_content_null_terminated), "cuModuleLoadData")
        cu_module = cu_module_ptr
        print("PTX Module loaded.")

        # --- Get Kernel Function Handles ---
        timing_kernel_func = ctypes.c_void_p()
        flush_kernel_func = ctypes.c_void_p()
        CU_CHECK(libcuda.cuModuleGetFunction(ctypes.byref(timing_kernel_func), cu_module, b"timing_kernel"), "cuModuleGetFunction timing")
        CU_CHECK(libcuda.cuModuleGetFunction(ctypes.byref(flush_kernel_func), cu_module, b"flush_kernel"), "cuModuleGetFunction flush")
        print("Kernel functions retrieved.")

        # --- Allocate Memory (DRIVER API cuMemAllocManaged) ---
        print("Allocating managed memory using Driver API...")
        total_size = TOTAL_PAGES_NEEDED * STRIDE_SIZE # Uses updated TOTAL_PAGES_NEEDED
        pool_gpu_ptr = ctypes.c_ulonglong()
        flags = 1 # CU_MEM_ATTACH_GLOBAL
        CU_CHECK(libcuda.cuMemAllocManaged(ctypes.byref(pool_gpu_ptr), total_size, flags), "cuMemAllocManaged Pool")
        pool_gpu_ptr_val = pool_gpu_ptr.value
        if not pool_gpu_ptr_val: raise RuntimeError("cuMemAllocManaged returned NULL")
        print(f"Managed Pool GPU VA Start: {hex(pool_gpu_ptr_val)}")

        # Also allocate results buffer
        results_buffer_size = NUM_TIMING_SAMPLES * ctypes.sizeof(ctypes.c_ulonglong) * 2
        results_buffer_gpu_ptr = ctypes.c_ulonglong()
        CU_CHECK(libcuda.cuMemAllocManaged(ctypes.byref(results_buffer_gpu_ptr), results_buffer_size, flags), "cuMemAllocManaged Results")
        results_buffer_gpu_va = results_buffer_gpu_ptr.value
        if not results_buffer_gpu_va: raise RuntimeError("cuMemAllocManaged for results returned NULL")

        # --- Get Host Pointer for Pool (Needed for Initialization) ---
        pool_host_ptr_attr = ctypes.c_void_p()
        CU_CHECK(libcuda.cuPointerGetAttribute(ctypes.byref(pool_host_ptr_attr), 4, pool_gpu_ptr_val), # 4=HOST_POINTER
                 "cuPointerGetAttribute HOST_POINTER Pool")
        if not pool_host_ptr_attr.value: raise RuntimeError("Could not get host pointer for pool")
        pool_host_ptr = pool_host_ptr_attr
        print(f"Managed Pool Host Ptr (for init): {hex(pool_host_ptr.value)}")

        # --- Calculate VAs ---
        target_page_va = pool_gpu_ptr_val + 0 * STRIDE_SIZE
        l1_l2_evict_start_idx = TARGET_PAGE_COUNT
        l3_fill_start_idx = TARGET_PAGE_COUNT + L1_L2_EVICT_COUNT
        l1_l2_evict_pages_va_list = [pool_gpu_ptr_val + (l1_l2_evict_start_idx + i) * STRIDE_SIZE for i in range(L1_L2_EVICT_COUNT)]
        l3_fill_pages_va_list = [pool_gpu_ptr_val + (l3_fill_start_idx + i) * STRIDE_SIZE for i in range(L3_FILL_COUNT)]

        # --- Allocate and Copy VA Lists ---
        VA_List_Type = ctypes.c_ulonglong * L1_L2_EVICT_COUNT
        h_l1_l2_list = VA_List_Type(*l1_l2_evict_pages_va_list)
        d_l1_l2_list_ptr = ctypes.c_ulonglong(0) # Initialize
        CU_CHECK(libcuda.cuMemAlloc_v2(ctypes.byref(d_l1_l2_list_ptr), ctypes.sizeof(h_l1_l2_list)), "cuMemAlloc L1L2 List")
        CU_CHECK(libcuda.cuMemcpyHtoD_v2(d_l1_l2_list_ptr, ctypes.byref(h_l1_l2_list), ctypes.sizeof(h_l1_l2_list)), "cuMemcpyHtoD L1L2 List")

        VA_List_Type_Fill = ctypes.c_ulonglong * L3_FILL_COUNT
        h_l3_fill_list = VA_List_Type_Fill(*l3_fill_pages_va_list)
        d_l3_fill_list_ptr = ctypes.c_ulonglong(0) # Initialize
        CU_CHECK(libcuda.cuMemAlloc_v2(ctypes.byref(d_l3_fill_list_ptr), ctypes.sizeof(h_l3_fill_list)), "cuMemAlloc L3Fill List")
        CU_CHECK(libcuda.cuMemcpyHtoD_v2(d_l3_fill_list_ptr, ctypes.byref(h_l3_fill_list), ctypes.sizeof(h_l3_fill_list)), "cuMemcpyHtoD L3Fill List")

        # --- Initialize Target Page for Pointer Chasing ---
        print("Initializing target page for pointer chasing...")
        target_page_host_ptr = ctypes.c_void_p(pool_host_ptr.value + 0 * STRIDE_SIZE)
        chase_length = 8
        TargetPageElements = PAGE_SIZE // ctypes.sizeof(ctypes.c_ulonglong)
        TargetPageType = ctypes.c_ulonglong * TargetPageElements
        target_page_view = ctypes.cast(target_page_host_ptr, ctypes.POINTER(TargetPageType))

        for k in range(chase_length):
            next_element_index = (k + 1) % chase_length
            next_element_gpu_va = target_page_va + next_element_index * ctypes.sizeof(ctypes.c_ulonglong)
            target_page_view.contents[k] = next_element_gpu_va

        # Sync device using RUNTIME API
        if not libcudart: raise RuntimeError("libcudart needed for sync after host write")
        # Need to set device for runtime context before sync? Maybe not needed if driver context is primary.
        # Let's try without setting device first. If it fails, add:
        # CUDA_RT_CHECK(libcudart.cudaSetDevice(GPU_ID), "cudaSetDevice Target Init Sync")
        CUDA_RT_CHECK(libcudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize Target Init")
        print("Target page initialized.")

        print("Memory allocation and VA setup complete.")

        # --- Prepare Kernel Arguments (Values) ---
        target_page_va_val = ctypes.c_ulonglong(target_page_va)
        d_l1_l2_list_val = ctypes.c_ulonglong(d_l1_l2_list_ptr.value)
        num_l1_l2_val = ctypes.c_int(L1_L2_EVICT_COUNT)
        results_buffer_val = ctypes.c_ulonglong(results_buffer_gpu_va)
        buffer_offset_val = ctypes.c_int(0) # Placeholder, will be updated
        num_samples_val = ctypes.c_int(NUM_TIMING_SAMPLES)
        d_l3_fill_list_val = ctypes.c_ulonglong(d_l3_fill_list_ptr.value)
        num_l3_fill_val = ctypes.c_int(L3_FILL_COUNT)


        # --- Run Warmup ---
        print("Running warmup...")
        buffer_offset_val_warmup = ctypes.c_int(0)
        num_samples_val_warmup = ctypes.c_int(10)
        timing_args_ptrs_warmup = [
            ctypes.byref(target_page_va_val), ctypes.byref(d_l1_l2_list_val),
            ctypes.byref(num_l1_l2_val), ctypes.byref(results_buffer_val),
            ctypes.byref(buffer_offset_val_warmup), ctypes.byref(num_samples_val_warmup) ]
        timing_params_warmup = (ctypes.c_void_p * len(timing_args_ptrs_warmup))(*[ctypes.cast(p, ctypes.c_void_p) for p in timing_args_ptrs_warmup])
        CU_CHECK(libcuda.cuLaunchKernel(timing_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, timing_params_warmup, None), "cuLaunchKernel Warmup")
        CU_CHECK(libcuda.cuStreamSynchronize(None), "cuStreamSynchronize Warmup")
        print("Warmup complete.")


        # --- Measure Hit Latency ---
        print("Measuring HIT latency...")
        buffer_offset_val_hit = ctypes.c_int(0)
        timing_args_ptrs_hit = [
            ctypes.byref(target_page_va_val), ctypes.byref(d_l1_l2_list_val),
            ctypes.byref(num_l1_l2_val), ctypes.byref(results_buffer_val),
            ctypes.byref(buffer_offset_val_hit), ctypes.byref(num_samples_val) ]
        timing_params_hit = (ctypes.c_void_p * len(timing_args_ptrs_hit))(*[ctypes.cast(p, ctypes.c_void_p) for p in timing_args_ptrs_hit])
        CU_CHECK(libcuda.cuLaunchKernel(timing_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, timing_params_hit, None), "cuLaunchKernel Hit")
        CU_CHECK(libcuda.cuStreamSynchronize(None), "cuStreamSynchronize Hit")
        print("Hit measurement complete.")


        # --- Run L3 Flush ---
        print("Running L3 Flush...")
        flush_args_ptrs = [
            ctypes.byref(d_l3_fill_list_val), ctypes.byref(num_l3_fill_val) ]
        flush_params = (ctypes.c_void_p * len(flush_args_ptrs))(*[ctypes.cast(p, ctypes.c_void_p) for p in flush_args_ptrs])
        CU_CHECK(libcuda.cuLaunchKernel(flush_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, flush_params, None), "cuLaunchKernel Flush")
        CU_CHECK(libcuda.cuStreamSynchronize(None), "cuStreamSynchronize Flush")
        print("L3 Flush complete.")


        # --- Measure Miss Latency ---
        print("Measuring MISS latency...")
        buffer_offset_val_miss = ctypes.c_int(NUM_TIMING_SAMPLES)
        timing_args_ptrs_miss = [
            ctypes.byref(target_page_va_val), ctypes.byref(d_l1_l2_list_val),
            ctypes.byref(num_l1_l2_val), ctypes.byref(results_buffer_val),
            ctypes.byref(buffer_offset_val_miss), ctypes.byref(num_samples_val) ]
        timing_params_miss = (ctypes.c_void_p * len(timing_args_ptrs_miss))(*[ctypes.cast(p, ctypes.c_void_p) for p in timing_args_ptrs_miss])
        CU_CHECK(libcuda.cuLaunchKernel(timing_kernel_func, 1, 1, 1, 1, 1, 1, 0, None, timing_params_miss, None), "cuLaunchKernel Miss")
        CU_CHECK(libcuda.cuStreamSynchronize(None), "cuStreamSynchronize Miss")
        print("Miss measurement complete.")


        # --- Retrieve and Analyze Results ---
        print("Analyzing results...")
        ResultsArrayType = ctypes.c_ulonglong * (NUM_TIMING_SAMPLES * 2)
        # Cast the GPU VA of the results buffer to access from host
        results_ptr = ctypes.cast(results_buffer_gpu_va, ctypes.POINTER(ResultsArrayType))
        all_results = np.frombuffer(results_ptr.contents, dtype=np.uint64).copy()

        hit_times = all_results[:NUM_TIMING_SAMPLES]
        miss_times = all_results[NUM_TIMING_SAMPLES:]
        hit_times_filtered = hit_times[hit_times > 0][5:] # Basic filtering
        miss_times_filtered = miss_times[miss_times > 0][5:] # Basic filtering

        if len(hit_times_filtered) > 0 and len(miss_times_filtered) > 0:
            print("\n--- HIT Latency (Cycles) ---")
            print(f"Min:    {np.min(hit_times_filtered)}")
            print(f"Max:    {np.max(hit_times_filtered)}")
            print(f"Mean:   {np.mean(hit_times_filtered):.2f}")
            print(f"Median: {np.median(hit_times_filtered)}")
            print(f"StdDev: {np.std(hit_times_filtered):.2f}")

            print("\n--- MISS Latency (Cycles) ---")
            print(f"Min:    {np.min(miss_times_filtered)}")
            print(f"Max:    {np.max(miss_times_filtered)}")
            print(f"Mean:   {np.mean(miss_times_filtered):.2f}")
            print(f"Median: {np.median(miss_times_filtered)}")
            print(f"StdDev: {np.std(miss_times_filtered):.2f}")

            hit_median = np.median(hit_times_filtered)
            miss_median = np.median(miss_times_filtered)
            miss_p10 = np.percentile(miss_times_filtered, 10)
            threshold_mid = hit_median + (miss_median - hit_median) / 2
            threshold_p10 = miss_p10 * 0.95
            print(f"\nSuggested Threshold T (Midpoint): {threshold_mid:.0f}")
            print(f"Suggested Threshold T (P10-Based): {threshold_p10:.0f}")
            print(f"----> Use a value between {hit_median:.0f} and {miss_p10:.0f} for T in the covert channel script.")

            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6)); plt.hist(hit_times_filtered, bins=50, alpha=0.7, label=f'Hit Latency (Median: {hit_median:.0f})'); plt.hist(miss_times_filtered, bins=50, alpha=0.7, label=f'Miss Latency (Median: {miss_median:.0f})'); plt.axvline(threshold_mid, color='r', linestyle='--', label=f'Mid Threshold ({threshold_mid:.0f})'); plt.axvline(threshold_p10, color='g', linestyle=':', label=f'P10 Threshold ({threshold_p10:.0f})'); plt.xlabel('Cycles (%clock64)'); plt.ylabel('Frequency'); plt.title('Baseline TLB Hit/Miss Latency Distribution'); plt.legend(); plt.grid(True); plt.savefig("baseline_latency.png")
                print("\nSaved latency histogram to baseline_latency.png")
            except ImportError: print("\nInstall matplotlib (pip install matplotlib) to generate latency plot.")
            except Exception as plot_e: print(f"\nError generating plot: {plot_e}")
        else:
            print("\nError: Not enough valid timing samples collected.")
            print("Hit samples raw:", hit_times[:20])
            print("Miss samples raw:", miss_times[:20])


    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # --- Cleanup (Use Driver API) ---
        print("\nCleaning up...")
        # Free device-only memory first
        if libcuda and cu_context: # Check if driver context was created
             if d_l1_l2_list_ptr.value != 0:
                 try: CU_CHECK(libcuda.cuMemFree_v2(d_l1_l2_list_ptr), "cuMemFree L1L2 List")
                 except Exception as e: print(f"Cleanup L1L2 Error: {e}")
             if d_l3_fill_list_ptr.value != 0:
                 try: CU_CHECK(libcuda.cuMemFree_v2(d_l3_fill_list_ptr), "cuMemFree L3Fill List")
                 except Exception as e: print(f"Cleanup L3Fill Error: {e}")
        # Free managed memory
        if pool_gpu_ptr_val != 0: # Check if pool was allocated
            try: CU_CHECK(libcuda.cuMemFree_v2(pool_gpu_ptr_val), "cuMemFree Pool")
            except Exception as e: print(f"Cleanup Pool Error: {e}")
        if results_buffer_gpu_va != 0: # Check if results buffer was allocated
            try: CU_CHECK(libcuda.cuMemFree_v2(results_buffer_gpu_va), "cuMemFree Results")
            except Exception as e: print(f"Cleanup Results Error: {e}")
        # Destroy context
        if cu_context and cu_context.value:
            try: CU_CHECK(libcuda.cuCtxDestroy_v2(cu_context), "cuCtxDestroy")
            except Exception as e: print(f"Cleanup Context Error: {e}")
        print("Cleanup finished.")