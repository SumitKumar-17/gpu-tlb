import ctypes
import numpy as np
import time
import os
import sys

# Import functions and constants from gpu_utils_pp
from gpu_utils_pp import (load_cuda_libs_pp, CU_CHECK, CUDA_RT_CHECK,
                          compile_kernels_to_ptx_pp, find_pages_for_set,
                          PAGE_SIZE, PRIME_PROBE_N, L1_L2_EVICT_COUNT)

# --- Configuration ---
GPU_ID = 0
TARGET_SET_ID = 0  # Which L3 set index to probe (for probe_time_kernel)
# PRIME_PROBE_N is imported, defines how many pages in TARGET_SET_ID are probed
# L1_L2_EVICT_COUNT is imported, for probe_time_kernel's L1/L2 eviction stage

ALLOC_POOL_MB = 8 * 1024  # 8 GB pool for VA finding and matrices
KERNELS_FILE = "pp_baseline_kernels.cu"  # Must contain probe_time, matrix_mul, matrix_add
KERNELS_PTX = "pp_baseline_kernels.ptx"

TARGET_SAMPLING_INTERVAL = 0.010  # 10 milliseconds

# Workload Configuration
MATRIX_N = 256  # Dimension for square matrices (N x N).
NUM_MATMUL_OPS_1 = 100
NUM_MATADD_OPS = 100
NUM_MATMUL_OPS_2 = 100

# CUDA constants
CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 1  # From cuda.h CUmem_advise_enum


def main():
    # Load libs and get local handles
    driver_ok, local_libcuda, local_libcudart = load_cuda_libs_pp()
    if not driver_ok:
        print("Failed loading libcuda.")
        exit(1)
    if not local_libcudart:
        print("Error: libcudart not loaded (needed for sync and prefetch).")
        exit(1)

    # --- Compile Kernels ---
    if not compile_kernels_to_ptx_pp(KERNELS_FILE, KERNELS_PTX):
        exit(1)

    cu_context = None
    cu_module = None
    pool_gpu_va_start = 0
    pool_gpu_ptr = ctypes.c_ulonglong(0)

    d_prime_probe_vas_ptr = ctypes.c_ulonglong(0)
    d_l1l2_evict_vas_ptr = ctypes.c_ulonglong(0)
    result_time_gpu_ptr = ctypes.c_ulonglong(0)
    result_time_gpu_va = 0

    probe_timings_during_workload = []

    d_matrix_A_ptr = ctypes.c_ulonglong(0)
    d_matrix_B_ptr = ctypes.c_ulonglong(0)
    d_matrix_C_ptr = ctypes.c_ulonglong(0)
    d_matrix_D_ptr = ctypes.c_ulonglong(0)
    d_matrix_E_ptr = ctypes.c_ulonglong(0)

    try:
        # --- Init CUDA & Context ---
        CU_CHECK(local_libcuda.cuInit(0), "cuInit")
        cu_device = ctypes.c_int()
        CU_CHECK(local_libcuda.cuDeviceGet(ctypes.byref(cu_device), GPU_ID), "cuDeviceGet")
        print(f"Device ID from cuDeviceGet: {cu_device.value}")  # Debug print for device ID

        cu_context_ptr = ctypes.c_void_p()
        CU_CHECK(local_libcuda.cuCtxCreate_v2(ctypes.byref(cu_context_ptr), 0, cu_device), "cuCtxCreate")
        cu_context = cu_context_ptr
        print("CUDA Context created.")

        # --- Load PTX Module & Kernels ---
        cu_module_ptr = ctypes.c_void_p()
        with open(KERNELS_PTX, 'rb') as f:
            ptx_content = f.read()
        ptx_content_null_terminated = ctypes.create_string_buffer(ptx_content + b'\x00')
        CU_CHECK(local_libcuda.cuModuleLoadData(ctypes.byref(cu_module_ptr), ptx_content_null_terminated), "cuModuleLoadData")
        cu_module = cu_module_ptr

        probe_kernel_func = ctypes.c_void_p()
        matmul_kernel_func = ctypes.c_void_p()
        matrix_add_kernel_func = ctypes.c_void_p()

        CU_CHECK(local_libcuda.cuModuleGetFunction(ctypes.byref(probe_kernel_func), cu_module, b"probe_time_kernel"), "cuModuleGetFunction probe_time_kernel")
        CU_CHECK(local_libcuda.cuModuleGetFunction(ctypes.byref(matmul_kernel_func), cu_module, b"matrix_mul_kernel"), "cuModuleGetFunction matrix_mul_kernel")
        CU_CHECK(local_libcuda.cuModuleGetFunction(ctypes.byref(matrix_add_kernel_func), cu_module, b"matrix_add_kernel"), "cuModuleGetFunction matrix_add_kernel")
        print("Kernels loaded.")

        # --- Allocate Memory Pool for VA finding ---
        print("Allocating VA pool memory...")
        pool_size_bytes = ALLOC_POOL_MB * 1024 * 1024
        flags = 1  # CU_MEM_ATTACH_GLOBAL for managed memory
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(pool_gpu_ptr), pool_size_bytes, flags), "cuMemAllocManaged Pool")
        pool_gpu_va_start = pool_gpu_ptr.value
        if not pool_gpu_va_start:
            raise RuntimeError("Pool allocation failed (VA is 0)")
        print(f"Pool GPU VA Start: {hex(pool_gpu_va_start)}")

        # --- Setup VAs for probe_time_kernel ---
        if PRIME_PROBE_N <= 0:
            raise RuntimeError("PRIME_PROBE_N must be > 0 to define target VAs for probe_time_kernel.")

        prime_probe_vas_list = find_pages_for_set(local_libcuda, pool_gpu_va_start, pool_size_bytes, TARGET_SET_ID,
                                                   PRIME_PROBE_N)
        if len(prime_probe_vas_list) < PRIME_PROBE_N:
            raise RuntimeError(
                f"Could not find enough pages ({len(prime_probe_vas_list)}/{PRIME_PROBE_N}) for probe target set {TARGET_SET_ID}")
        print(f"Found {len(prime_probe_vas_list)} pages for probe target set {TARGET_SET_ID}.")

        VasListTypePP = ctypes.c_ulonglong * PRIME_PROBE_N
        h_pp_vas = VasListTypePP(*prime_probe_vas_list)
        CU_CHECK(local_libcuda.cuMemAlloc_v2(ctypes.byref(d_prime_probe_vas_ptr), ctypes.sizeof(h_pp_vas)), "cuMemAlloc_v2 PP VAs")
        CU_CHECK(local_libcuda.cuMemcpyHtoD_v2(d_prime_probe_vas_ptr.value, ctypes.byref(h_pp_vas),
                                            ctypes.sizeof(h_pp_vas)), "cuMemcpyHtoD_v2 PP VAs")

        actual_l1l2_evict_count = 0
        if L1_L2_EVICT_COUNT > 0:
            l1l2_evict_set_id = (TARGET_SET_ID + 10) % 256
            l1l2_evict_vas_list = find_pages_for_set(local_libcuda, pool_gpu_va_start, pool_size_bytes,
                                                       l1l2_evict_set_id, L1_L2_EVICT_COUNT)
            actual_l1l2_evict_count = len(l1l2_evict_vas_list)
            if actual_l1l2_evict_count < L1_L2_EVICT_COUNT:
                print(
                    f"Warning: Could not find enough pages for L1/L2 eviction set {l1l2_evict_set_id}. Found {actual_l1l2_evict_count}/{L1_L2_EVICT_COUNT}.")
            if actual_l1l2_evict_count > 0:
                VasListTypeL1L2 = ctypes.c_ulonglong * actual_l1l2_evict_count
                h_l1l2_vas = VasListTypeL1L2(*l1l2_evict_vas_list)
                CU_CHECK(local_libcuda.cuMemAlloc_v2(ctypes.byref(d_l1l2_evict_vas_ptr), ctypes.sizeof(h_l1l2_vas)), "cuMemAlloc_v2 L1L2 VAs")
                CU_CHECK(local_libcuda.cuMemcpyHtoD_v2(d_l1l2_evict_vas_ptr.value, ctypes.byref(h_l1l2_vas),
                                                    ctypes.sizeof(h_l1l2_vas)), "cuMemcpyHtoD_v2 L1L2 VAs")
                print(f"Found and copied {actual_l1l2_evict_count} pages for L1/L2 eviction.")
            else:
                d_l1l2_evict_vas_ptr.value = 0
                print("Warning: No L1/L2 eviction pages will be used by probe_time_kernel.")
        else:
            d_l1l2_evict_vas_ptr.value = 0
            print("L1_L2_EVICT_COUNT is 0, no L1/L2 eviction by probe_time_kernel.")

        num_l1l2_evict_val = ctypes.c_int(actual_l1l2_evict_count)

        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(result_time_gpu_ptr), ctypes.sizeof(ctypes.c_ulonglong), flags), "cuMemAllocManaged Result Time")
        result_time_gpu_va = result_time_gpu_ptr.value
        if not result_time_gpu_va:
            raise RuntimeError("Result time buffer alloc failed")
        ctypes.memset(result_time_gpu_va, 0, ctypes.sizeof(ctypes.c_ulonglong))

        MATRIX_ELEMENTS = MATRIX_N * MATRIX_N
        MATRIX_SIZE_BYTES = MATRIX_ELEMENTS * ctypes.sizeof(ctypes.c_float)

        print(f"Allocating memory for {MATRIX_N}x{MATRIX_N} matrices ({MATRIX_SIZE_BYTES / (1024 * 1024):.2f} MB each)...")
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(d_matrix_A_ptr), MATRIX_SIZE_BYTES, flags), "cuMemAllocManaged MatA")
        if not d_matrix_A_ptr.value:
            raise RuntimeError("MatA allocation failed (pointer is null)")
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(d_matrix_B_ptr), MATRIX_SIZE_BYTES, flags), "cuMemAllocManaged MatB")
        if not d_matrix_B_ptr.value:
            raise RuntimeError("MatB allocation failed (pointer is null)")
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(d_matrix_C_ptr), MATRIX_SIZE_BYTES, flags), "cuMemAllocManaged MatC")
        if not d_matrix_C_ptr.value:
            raise RuntimeError("MatC allocation failed (pointer is null)")
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(d_matrix_D_ptr), MATRIX_SIZE_BYTES, flags), "cuMemAllocManaged MatD")
        if not d_matrix_D_ptr.value:
            raise RuntimeError("MatD allocation failed (pointer is null)")
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(d_matrix_E_ptr), MATRIX_SIZE_BYTES, flags), "cuMemAllocManaged MatE")
        if not d_matrix_E_ptr.value:
            raise RuntimeError("MatE allocation failed (pointer is null)")

        d_A_va = d_matrix_A_ptr.value
        d_B_va = d_matrix_B_ptr.value
        d_C_va = d_matrix_C_ptr.value
        d_D_va = d_matrix_D_ptr.value
        d_E_va = d_matrix_E_ptr.value

        h_A_np = np.random.rand(MATRIX_N, MATRIX_N).astype(np.float32)
        h_B_np = np.random.rand(MATRIX_N, MATRIX_N).astype(np.float32)
        h_D_np = np.random.rand(MATRIX_N, MATRIX_N).astype(np.float32)

        ctypes.memmove(d_A_va, h_A_np.ctypes.data, MATRIX_SIZE_BYTES)
        ctypes.memmove(d_B_va, h_B_np.ctypes.data, MATRIX_SIZE_BYTES)
        ctypes.memmove(d_D_va, h_D_np.ctypes.data, MATRIX_SIZE_BYTES)

        # --- Temporarily commenting out cuMemAdvise and cudaMemPrefetchAsync for debugging ---
        print("Skipping cuMemAdvise and cudaMemPrefetchAsync for debugging purposes.")
        # CU_CHECK(local_libcuda.cuMemAdvise(d_A_va, MATRIX_SIZE_BYTES, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, cu_device.value), "cuMemAdvise MatA Preferred Location")
        # CU_CHECK(local_libcuda.cuMemAdvise(d_B_va, MATRIX_SIZE_BYTES, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, cu_device.value), "cuMemAdvise MatB Preferred Location")
        # CU_CHECK(local_libcuda.cuMemAdvise(d_D_va, MATRIX_SIZE_BYTES, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, cu_device.value), "cuMemAdvise MatD Preferred Location")

        # CUDA_RT_CHECK(local_libcudart.cudaMemPrefetchAsync(d_A_va, MATRIX_SIZE_BYTES, cu_device.value, None), "cudaMemPrefetchAsync MatA")
        # CUDA_RT_CHECK(local_libcudart.cudaMemPrefetchAsync(d_B_va, MATRIX_SIZE_BYTES, cu_device.value, None), "cudaMemPrefetchAsync MatB")
        # CUDA_RT_CHECK(local_libcudart.cudaMemPrefetchAsync(d_D_va, MATRIX_SIZE_BYTES, cu_device.value, None), "cudaMemPrefetchAsync MatD")
        # CUDA_RT_CHECK(local_libcudart.cudaDeviceSynchronize())  # Sync would be needed if prefetch was active
        # --- End of temporarily commented out section ---

        print("Device memory for workload setup complete (advise/prefetch skipped).")

        num_pp_pages_val = ctypes.c_int(PRIME_PROBE_N)
        d_pp_vas_val = ctypes.c_ulonglong(d_prime_probe_vas_ptr.value)
        d_l1l2_evict_vas_val = ctypes.c_ulonglong(d_l1l2_evict_vas_ptr.value)
        d_result_time_kernel_arg = ctypes.c_ulonglong(result_time_gpu_va)

        THREADS_PER_BLOCK_XY = 16
        matmul_grid_dim_x = (MATRIX_N + THREADS_PER_BLOCK_XY - 1) // THREADS_PER_BLOCK_XY
        matmul_grid_dim_y = (MATRIX_N + THREADS_PER_BLOCK_XY - 1) // THREADS_PER_BLOCK_XY

        THREADS_PER_BLOCK_ADD = 256
        matadd_grid_dim_x = (MATRIX_ELEMENTS + THREADS_PER_BLOCK_ADD - 1) // THREADS_PER_BLOCK_ADD

        mat_N_val = ctypes.c_int(MATRIX_N)
        mat_elements_val = ctypes.c_int(MATRIX_ELEMENTS)

        total_ops_to_run = NUM_MATMUL_OPS_1 + NUM_MATADD_OPS + NUM_MATMUL_OPS_2
        print(f"Starting workload ({total_ops_to_run} ops) and L3 TLB monitoring...")

        matmul1_done = 0
        matadd_done = 0
        matmul2_done = 0
        samples_collected = 0

        total_start_time = time.perf_counter()

        while matmul1_done < NUM_MATMUL_OPS_1 or \
                matadd_done < NUM_MATADD_OPS or \
                matmul2_done < NUM_MATMUL_OPS_2:

            iteration_start_time = time.perf_counter()
            workload_kernel_launched = False

            if matmul1_done < NUM_MATMUL_OPS_1:
                args_matmul = [
                    ctypes.byref(ctypes.c_ulonglong(d_A_va)),
                    ctypes.byref(ctypes.c_ulonglong(d_B_va)),
                    ctypes.byref(ctypes.c_ulonglong(d_C_va)),
                    ctypes.byref(mat_N_val)
                ]
                CU_CHECK(local_libcuda.cuLaunchKernel(
                    matmul_kernel_func, matmul_grid_dim_x, matmul_grid_dim_y, 1,
                    THREADS_PER_BLOCK_XY, THREADS_PER_BLOCK_XY, 1,
                    0, None,
                    (ctypes.c_void_p * len(args_matmul))(*[ctypes.cast(arg, ctypes.c_void_p) for arg in args_matmul]),
                    None
                ), "cuLaunchKernel matmul1")
                matmul1_done += 1
                workload_kernel_launched = True
            elif matadd_done < NUM_MATADD_OPS:
                args_add = [
                    ctypes.byref(ctypes.c_ulonglong(d_C_va)),
                    ctypes.byref(ctypes.c_ulonglong(d_D_va)),
                    ctypes.byref(ctypes.c_ulonglong(d_E_va)),
                    ctypes.byref(mat_elements_val)
                ]
                CU_CHECK(local_libcuda.cuLaunchKernel(
                    matrix_add_kernel_func, matadd_grid_dim_x, 1, 1,
                    THREADS_PER_BLOCK_ADD, 1, 1,
                    0, None,
                    (ctypes.c_void_p * len(args_add))(*[ctypes.cast(arg, ctypes.c_void_p) for arg in args_add]),
                    None
                ), "cuLaunchKernel matadd")
                matadd_done += 1
                workload_kernel_launched = True
            elif matmul2_done < NUM_MATMUL_OPS_2:
                args_matmul = [
                    ctypes.byref(ctypes.c_ulonglong(d_A_va)),
                    ctypes.byref(ctypes.c_ulonglong(d_B_va)),
                    ctypes.byref(ctypes.c_ulonglong(d_C_va)),
                    ctypes.byref(mat_N_val)
                ]
                CU_CHECK(local_libcuda.cuLaunchKernel(
                    matmul_kernel_func, matmul_grid_dim_x, matmul_grid_dim_y, 1,
                    THREADS_PER_BLOCK_XY, THREADS_PER_BLOCK_XY, 1,
                    0, None,
                    (ctypes.c_void_p * len(args_matmul))(*[ctypes.cast(arg, ctypes.c_void_p) for arg in args_matmul]),
                    None
                ), "cuLaunchKernel matmul2")
                matmul2_done += 1
                workload_kernel_launched = True

            if workload_kernel_launched:
                CUDA_RT_CHECK(local_libcudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

            probe_args = [
                ctypes.byref(d_pp_vas_val),
                ctypes.byref(num_pp_pages_val),
                ctypes.byref(d_l1l2_evict_vas_val),
                ctypes.byref(num_l1l2_evict_val),
                ctypes.byref(d_result_time_kernel_arg)
            ]
            CU_CHECK(local_libcuda.cuLaunchKernel(
                probe_kernel_func, 1, 1, 1, 1, 1, 1,
                0, None,
                (ctypes.c_void_p * len(probe_args))(*[ctypes.cast(arg, ctypes.c_void_p) for arg in probe_args]),
                None
            ), "cuLaunchKernel probe")
            CUDA_RT_CHECK(local_libcudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize probe")

            current_probe_time_buffer = (ctypes.c_ulonglong * 1)()
            ctypes.memmove(ctypes.byref(current_probe_time_buffer), result_time_gpu_va,
                           ctypes.sizeof(ctypes.c_ulonglong))
            probe_timings_during_workload.append(current_probe_time_buffer[0])
            samples_collected += 1

            iteration_end_time = time.perf_counter()
            elapsed_this_iteration = iteration_end_time - iteration_start_time
            sleep_duration = TARGET_SAMPLING_INTERVAL - elapsed_this_iteration
            if sleep_duration > 0:
                time.sleep(sleep_duration)

            if samples_collected % 50 == 0:
                ops_done = matmul1_done + matadd_done + matmul2_done
                print(
                    f"Sample {samples_collected}: Ops done {ops_done}/{total_ops_to_run}. Last probe: {current_probe_time_buffer[0]} cycles. Iter time: {elapsed_this_iteration * 1000:.2f} ms.")

        total_end_time = time.perf_counter()
        print(f"Finished collecting {samples_collected} samples in {total_end_time - total_start_time:.2f} seconds.")

        if probe_timings_during_workload:
            avg_timing = np.mean(probe_timings_during_workload)
            min_timing = np.min(probe_timings_during_workload)
            max_timing = np.max(probe_timings_during_workload)
            std_timing = np.std(probe_timings_during_workload)
            median_timing = np.median(probe_timings_during_workload)
            print("\nProbe Timings during Workload (GPU cycles):")
            print(f"Total Samples: {len(probe_timings_during_workload)}")
            print(f"Min timing:    {min_timing} cycles")
            print(f"Max timing:    {max_timing} cycles")
            print(f"Average timing: {avg_timing:.2f} cycles")
            print(f"Median timing:  {median_timing:.2f} cycles")
            print(f"Stddev timing:  {std_timing:.2f} cycles")

            try:
                results_filename = f"tlb_timings_workload_N{MATRIX_N}_set{TARGET_SET_ID}.npy"
                np.save(results_filename, np.array(probe_timings_during_workload))
                print(f"Results saved to {results_filename}")
            except Exception as e_save:
                print(f"Error saving results: {e_save}")
        else:
            print("No timings were collected.")

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
    finally:
        print("Cleaning up CUDA resources...")
        # Add context check before freeing
        if local_libcuda and cu_context:
            current_ctx_ptr_before_free = ctypes.c_void_p()
            err_get_ctx = local_libcuda.cuCtxGetCurrent(ctypes.byref(current_ctx_ptr_before_free))
            if err_get_ctx == 0:
                # print(f"Current context beforefree operations: {hex(current_ctx_ptr_before_free.value)}. Original context: {hex(cu_context.value if isinstance(cu_context, ctypes.c_void_p) else 0)}")
                if current_ctx_ptr_before_free.value != (
                        cu_context.value if isinstance(cu_context, ctypes.c_void_p) else 0) and current_ctx_ptr_before_free.value != 0:  # if 0, it means no context is current
                    print("Warning: Context may have changed or become non-current before explicit cleanup!")
            else:
                print(f"Error getting current context before free operations: {err_get_ctx}")

        if d_matrix_A_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_matrix_A_ptr.value), "Free Flush VAs") 
            except Exception: pass

        if d_matrix_B_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_matrix_B_ptr.value), "Free Flush VAs") 
            except Exception: pass

        if d_matrix_C_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_matrix_C_ptr.value), "Free Flush VAs") 
            except Exception: pass

        if d_matrix_D_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_matrix_D_ptr.value), "Free Flush VAs") 
            except Exception: pass

        if d_matrix_E_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_matrix_E_ptr.value), "Free Flush VAs") 
            except Exception: pass

        if d_prime_probe_vas_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_prime_probe_vas_ptr.value), "Free Flush VAs") 
            except Exception: pass

        if d_l1l2_evict_vas_ptr.value != 0:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_l1l2_evict_vas_ptr.value), "Free Flush VAs") 
            except Exception: pass

        if result_time_gpu_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(result_time_gpu_ptr.value), "Free Flush VAs") 
            except Exception: pass

        if pool_gpu_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(pool_gpu_ptr.value), "Free Flush VAs") 
            except Exception: pass

        if cu_module:
            CU_CHECK(local_libcuda.cuModuleUnload(cu_module), "cuModuleUnload")
        if cu_context:
            CU_CHECK(local_libcuda.cuCtxDestroy_v2(cu_context), "cuCtxDestroy")
        print("Cleanup complete.")


if __name__ == "__main__":
    if PRIME_PROBE_N <= 0:
        print("Error: PRIME_PROBE_N (from gpu_utils_pp) must be greater than 0.")
        exit(1)
    main()

