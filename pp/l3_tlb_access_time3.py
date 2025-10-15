import ctypes
import numpy as np
import time

# Import functions and constants from gpu_utils_pp
from gpu_utils_pp import (load_cuda_libs_pp, CU_CHECK, CUDA_RT_CHECK,
                          compile_kernels_to_ptx_pp)

# --- Configuration ---
GPU_ID = 0
TARGET_SET_ID = 0
ALLOC_POOL_MB = 8 * 1024
KERNELS_FILE = "pp_baseline_kernels.cu"
KERNELS_PTX = "pp_baseline_kernels.ptx"
TARGET_SAMPLING_INTERVAL = 0.010
probe_timings_during_workload = []
# Workload Configuration
MATRIX_N = 512
NUM_MATMUL_OPS_1 = 100
NUM_MATADD_OPS = 100
NUM_MATMUL_OPS_2 = 100

# CUDA constants
CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 1

# New Constants
MAX_THREADS_PER_BLOCK = 256  # Should match the #define in the CUDA code
MAX_BLOCKS = 2048  # Maximum number of blocks
TOTAL_OPS = NUM_MATMUL_OPS_1 + NUM_MATADD_OPS + NUM_MATMUL_OPS_2
THREAD_TIMING_BUFFER_SIZE_PER_OP = MAX_THREADS_PER_BLOCK * MAX_BLOCKS
THREAD_TIMING_BUFFER_SIZE = THREAD_TIMING_BUFFER_SIZE_PER_OP * TOTAL_OPS  # Buffer for all ops

def main():
    # Load libs
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
    d_l1l2_evict_vas_ptr = ctypes.c_ulonglong(0)
    result_time_gpu_ptr = ctypes.c_ulonglong(0)
    result_time_gpu_va = 0
    thread_timings_during_workload = []  # Store thread timings after transfer
    d_matrix_A_ptr = ctypes.c_ulonglong(0)
    d_matrix_B_ptr = ctypes.c_ulonglong(0)
    d_matrix_C_ptr = ctypes.c_ulonglong(0)
    d_matrix_D_ptr = ctypes.c_ulonglong(0)
    d_matrix_E_ptr = ctypes.c_ulonglong(0)

    # --- Thread Timing Buffer ---
    d_thread_timing_buffer_ptr = ctypes.c_ulonglong(0)  # Device pointer
    h_thread_timings = np.zeros(THREAD_TIMING_BUFFER_SIZE, dtype=np.uint64)  # Host buffer
    h_thread_timings_ctypes = (ctypes.c_ulonglong * THREAD_TIMING_BUFFER_SIZE).from_buffer(h_thread_timings)

    try:
        # --- Init CUDA & Context ---
        CU_CHECK(local_libcuda.cuInit(0), "cuInit")
        cu_device = ctypes.c_int()
        CU_CHECK(local_libcuda.cuDeviceGet(ctypes.byref(cu_device), GPU_ID), "cuDeviceGet")
        print(f"Device ID from cuDeviceGet: {cu_device.value}")

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

        
        matmul_kernel_func = ctypes.c_void_p()
        matrix_add_kernel_func = ctypes.c_void_p()

        CU_CHECK(local_libcuda.cuModuleGetFunction(ctypes.byref(matmul_kernel_func), cu_module, b"matrix_mul_kernel"), "cuModuleGetFunction matrix_mul_kernel")
        CU_CHECK(local_libcuda.cuModuleGetFunction(ctypes.byref(matrix_add_kernel_func), cu_module, b"matrix_add_kernel"), "cuModuleGetFunction matrix_add_kernel")
        print("Kernels loaded.")

        # --- Allocate Memory Pool for VA finding ---
        print("Allocating VA pool memory...")
        pool_size_bytes = ALLOC_POOL_MB * 1024 * 1024
        flags = 1
        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(pool_gpu_ptr), pool_size_bytes, flags), "cuMemAllocManaged Pool")
        pool_gpu_va_start = pool_gpu_ptr.value
        if not pool_gpu_va_start:
            raise RuntimeError("Pool allocation failed (VA is 0)")
        print(f"Pool GPU VA Start: {hex(pool_gpu_va_start)}")


        CU_CHECK(local_libcuda.cuMemAllocManaged(ctypes.byref(result_time_gpu_ptr), ctypes.sizeof(ctypes.c_ulonglong), flags), "cuMemAllocManaged Result Time")
        result_time_gpu_va = result_time_gpu_ptr.value
        if not result_time_gpu_va:
            raise RuntimeError("Result time buffer alloc failed")
        ctypes.memset(result_time_gpu_va, 0, ctypes.sizeof(ctypes.c_ulonglong))

        # --- Allocate Matrices ---
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

        print("Skipping cuMemAdvise and cudaMemPrefetchAsync for debugging purposes.")
        print("Device memory for workload setup complete (advise/prefetch skipped).")

 

        THREADS_PER_BLOCK_XY = 16
        matmul_grid_dim_x = (MATRIX_N + THREADS_PER_BLOCK_XY - 1) // THREADS_PER_BLOCK_XY
        matmul_grid_dim_y = (MATRIX_N + THREADS_PER_BLOCK_XY - 1) // THREADS_PER_BLOCK_XY

        THREADS_PER_BLOCK_ADD = 256
        matadd_grid_dim_x = (MATRIX_ELEMENTS + THREADS_PER_BLOCK_ADD - 1) // THREADS_PER_BLOCK_ADD

        mat_N_val = ctypes.c_int(MATRIX_N)
        mat_elements_val = ctypes.c_int(MATRIX_ELEMENTS)

        total_ops_to_run = TOTAL_OPS
        print(f"Starting workload ({total_ops_to_run} ops) and L3 TLB monitoring...")

        matmul1_done = 0
        matadd_done = 0
        matmul2_done = 0
        samples_collected = 0
        current_op_index = 0  # Tracks the current operation for timing buffer offset

        total_start_time = time.perf_counter()

        # --- Allocate the thread timing buffer on the device ---
        CU_CHECK(local_libcuda.cuMemAlloc_v2(ctypes.byref(d_thread_timing_buffer_ptr), THREAD_TIMING_BUFFER_SIZE * ctypes.sizeof(ctypes.c_ulonglong)), "cuMemAlloc_v2 thread timing buffer")
        d_thread_timing_buffer_ptr_value = d_thread_timing_buffer_ptr.value
        print(f"Thread timing buffer pointer: {hex(d_thread_timing_buffer_ptr_value)}")

        while matmul1_done < NUM_MATMUL_OPS_1 or \
              matadd_done < NUM_MATADD_OPS or \
              matmul2_done < NUM_MATMUL_OPS_2:

            iteration_start_time = time.perf_counter()
            workload_null_launched = False
            blocks_x = 0
            blocks_y = 0
            kernel_type = ""

            # Calculate offset for this operation's timings
            timing_offset = current_op_index * THREAD_TIMING_BUFFER_SIZE_PER_OP
            timing_offset_val = ctypes.c_int(timing_offset)

            if matmul1_done < NUM_MATMUL_OPS_1:
                blocks_x = matmul_grid_dim_x
                blocks_y = matmul_grid_dim_y
                kernel_type = "matmul1"
                args_matmul = [
                    ctypes.byref(ctypes.c_ulonglong(d_A_va)),
                    ctypes.byref(ctypes.c_ulonglong(d_B_va)),
                    ctypes.byref(ctypes.c_ulonglong(d_C_va)),
                    ctypes.byref(mat_N_val),
                    ctypes.byref(ctypes.c_ulonglong(d_thread_timing_buffer_ptr_value)),
                    ctypes.byref(timing_offset_val)  # Pass offset
                ]
                CU_CHECK(local_libcuda.cuLaunchKernel(
                    matmul_kernel_func, blocks_x, blocks_y, 1,
                    THREADS_PER_BLOCK_XY, THREADS_PER_BLOCK_XY, 1,
                    0, None,
                    (ctypes.c_void_p * len(args_matmul))(*[ctypes.cast(arg, ctypes.c_void_p) for arg in args_matmul]),
                    None
                ), "cuLaunchKernel matmul1")
                matmul1_done += 1
                workload_kernel_launched = True
            elif matadd_done < NUM_MATADD_OPS:
                blocks_x = matadd_grid_dim_x
                blocks_y = 1
                kernel_type = "matadd"
                args_add = [
                    ctypes.byref(ctypes.c_ulonglong(d_C_va)),
                    ctypes.byref(ctypes.c_ulonglong(d_D_va)),
                    ctypes.byref(ctypes.c_ulonglong(d_E_va)),
                    ctypes.byref(mat_elements_val),
                    ctypes.byref(ctypes.c_ulonglong(d_thread_timing_buffer_ptr_value)),
                    ctypes.byref(timing_offset_val)  # Pass offset
                ]
                CU_CHECK(local_libcuda.cuLaunchKernel(
                    matrix_add_kernel_func, blocks_x, blocks_y, 1,
                    THREADS_PER_BLOCK_ADD, 1, 1,
                    0, None,
                    (ctypes.c_void_p * len(args_add))(*[ctypes.cast(arg, ctypes.c_void_p) for arg in args_add]),
                    None
                ),"cuLaunchKernel matadd")
                matadd_done += 1
                workload_kernel_launched = True
            elif matmul2_done < NUM_MATMUL_OPS_2:
                blocks_x = matmul_grid_dim_x
                blocks_y = matmul_grid_dim_y
                kernel_type = "matmul2"
                args_matmul = [
                    ctypes.byref(ctypes.c_ulonglong(d_A_va)),
                    ctypes.byref(ctypes.c_ulonglong(d_B_va)),
                    ctypes.byref(ctypes.c_ulonglong(d_C_va)),
                    ctypes.byref(mat_N_val),
                    ctypes.byref(ctypes.c_ulonglong(d_thread_timing_buffer_ptr_value)),
                    ctypes.byref(timing_offset_val)  # Pass offset
                ]
                CU_CHECK(local_libcuda.cuLaunchKernel(
                    matmul_kernel_func, blocks_x, blocks_y, 1,
                    THREADS_PER_BLOCK_XY, THREADS_PER_BLOCK_XY, 1,
                    0, None,
                    (ctypes.c_void_p * len(args_matmul))(*[ctypes.cast(arg, ctypes.c_void_p) for arg in args_matmul]),
                    None
                ), "cuLaunchKernel matmul2")
                matmul2_done += 1
                workload_kernel_launched = True

            if workload_kernel_launched:
                CUDA_RT_CHECK(local_libcudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
                op_offset = current_op_index * THREAD_TIMING_BUFFER_SIZE_PER_OP
                host_ptr = ctypes.c_void_p(ctypes.addressof(h_thread_timings_ctypes) + op_offset * ctypes.sizeof(ctypes.c_ulonglong))
                CU_CHECK(local_libcuda.cuMemcpyDtoH(
                    host_ptr,
                    d_thread_timing_buffer_ptr_value + (op_offset * ctypes.sizeof(ctypes.c_ulonglong)),
                    THREAD_TIMING_BUFFER_SIZE_PER_OP * ctypes.sizeof(ctypes.c_ulonglong)
                ), f"cuMemcpyDtoH thread timings op {current_op_index}")

                # Access the copied timings via NumPy view
                start_idx = op_offset
                end_idx = start_idx + THREAD_TIMING_BUFFER_SIZE_PER_OP
                op_timings = h_thread_timings[start_idx:end_idx]
                non_zero_timings = op_timings[op_timings != 0]

                # Print the contents
                print(f"\nThread Timings for {kernel_type} Operation {current_op_index}:")
                if non_zero_timings.size > 0:
                    print(f"Number of non-zero timings: {non_zero_timings.size}")
                    print(f"Timings (cycles): {non_zero_timings.tolist()}")
                    print(f"Average: {np.mean(non_zero_timings):.2f} cycles")
                    print(f"Min: {np.min(non_zero_timings)} cycles")
                    print(f"Max: {np.max(non_zero_timings)} cycles")
                else:
                    print("No non-zero timings recorded.")

                # --- Transfer thread timings to host after each kernel launch ---
                # op_offset = current_op_index * THREAD_TIMING_BUFFER_SIZE_PER_OP
                # host_ptr = ctypes.c_void_p(ctypes.addressof(h_thread_timings_ctypes) + op_offset * ctypes.sizeof(ctypes.c_ulonglong))
                # CU_CHECK(local_libcuda.cuMemcpyDtoH(
                #     host_ptr,
                #     d_thread_timing_buffer_ptr_value + (op_offset * ctypes.sizeof(ctypes.c_ulonglong)),
                #     THREAD_TIMING_BUFFER_SIZE_PER_OP * ctypes.sizeof(ctypes.c_ulonglong)
                # ), f"cuMemcpyDtoH thread timings op {current_op_index}")
                # CU_CHECK(local_libcuda.cuMemcpyDtoH(
                #     ctypes.cast(h_thread_timings.ctypes.data, ctypes.c_void_p) + current_op_index * THREAD_TIMING_BUFFER_SIZE_PER_OP * ctypes.sizeof(ctypes.c_ulonglong),
                #     d_thread_timing_buffer_ptr_value + current_op_index * THREAD_TIMING_BUFFER_SIZE_PER_OP * ctypes.sizeof(ctypes.c_ulonglong),
                #     THREAD_TIMING_BUFFER_SIZE_PER_OP * ctypes.sizeof(ctypes.c_ulonglong)
                # ), "cuMemcpyDtoH thread timings")

                # --- Clear the device buffer after transfer ---
                # CU_CHECK(local_libcuda.cuMemsetD64(
                #     d_thread_timing_buffer_ptr_value + current_op_index * THREAD_TIMING_BUFFER_SIZE_PER_OP * ctypes.sizeof(ctypes.c_ulonglong),
                #     0,
                #     THREAD_TIMING_BUFFER_SIZE_PER_OP // 8  # Size in 64-bit words
                # ), "cuMemsetD64 to clear buffer")

                current_op_index += 1  # Increment operation index
            result_time_ptr_host_view = ctypes.cast(d_thread_timing_buffer_ptr_value, ctypes.POINTER(ctypes.c_ulonglong))
            print(result_time_ptr_host_view.contents.value)
            samples_collected += 1

            iteration_end_time = time.perf_counter()
            elapsed_this_iteration = iteration_end_time - iteration_start_time
            sleep_duration = TARGET_SAMPLING_INTERVAL - elapsed_this_iteration
            if sleep_duration > 0:
                time.sleep(sleep_duration)

            
        total_end_time = time.perf_counter()
        print(f"Finished collecting {samples_collected} samples in {total_end_time - total_start_time:.2f} seconds.")

        # --- Process thread timings ---
        for op_idx in range(total_ops_to_run):
            start_idx = op_idx * THREAD_TIMING_BUFFER_SIZE_PER_OP
            end_idx = (op_idx + 1) * THREAD_TIMING_BUFFER_SIZE_PER_OP
            op_timings = h_thread_timings[start_idx:end_idx]
            kernel_type = ""
            if op_idx < NUM_MATMUL_OPS_1:
                kernel_type = "matmul1"
            elif op_idx < NUM_MATADD_OPS + NUM_MATMUL_OPS_1:
                kernel_type = "matadd"
            else:
                kernel_type = "matmul2"

            # Store non-zero timings with kernel type
            non_zero_timings = op_timings[op_timings != 0]
            if non_zero_timings.size > 0:
                thread_timings = {
                    'kernel_type': kernel_type,
                    'timings': non_zero_timings.copy(),
                    'iteration': op_idx
                }
                thread_timings_during_workload.append(thread_timings)

                # Print timing statistics
                print(f"\n{kernel_type} Kernel Execution Times (operation {op_idx}):")
                print(f"Number of threads with timings: {non_zero_timings.size}")
                print(f"Average: {np.mean(non_zero_timings):.2f} cycles")
                print(f"Min: {np.min(non_zero_timings)} cycles")
                print(f"Max: {np.max(non_zero_timings)} cycles")

        
        # --- Process and save thread timings ---
        if thread_timings_during_workload:
            print("\nThread Timings Summary:")
            for kernel_type in ['matmul1', 'matadd', 'matmul2']:
                kernel_timings = [t['timings'] for t in thread_timings_during_workload if t['kernel_type'] == kernel_type]
                kernel_timings = [t for t in kernel_timings if t.size > 0]
                if kernel_timings:
                    all_timings = np.concatenate(kernel_timings)
                    print(f"\n{kernel_type} timings:")
                    print(f"Total samples: {len(kernel_timings)}")
                    print(f"Total threads measured: {all_timings.size}")
                    print(f"Average: {np.mean(all_timings):.2f} cycles")
                    print(f"Min: {np.min(all_timings)} cycles")
                    print(f"Max: {np.max(all_timings)} cycles")
                    print(f"Stddev: {np.std(all_timings):.2f} cycles")

            try:
                thread_results_filename = f"thread_timings_workload_N{MATRIX_N}_set{TARGET_SET_ID}.npy"
                np.save(thread_results_filename, thread_timings_during_workload)
                print(f"Thread timings saved to {thread_results_filename}")
            except Exception as e_save:
                print(f"Error saving thread timings: {e_save}")
        else:
            print("No thread timings were collected.")

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
    finally:
        print("Cleaning up CUDA resources...")
        if local_libcuda and cu_context:
            current_ctx_ptr_before_free = ctypes.c_void_p()
            err_get_ctx = local_libcuda.cuCtxGetCurrent(ctypes.byref(current_ctx_ptr_before_free))
            if err_get_ctx == 0:
                if current_ctx_ptr_before_free.value != 0:
                    print("Warning: Context may have changed or become non-current before explicit cleanup!")
            else:
                print(f"Error getting current context before free operations: {err_get_ctx}")

        if d_matrix_A_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_matrix_A_ptr.value), "Free MatA")
            except Exception: pass
        if d_matrix_B_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_matrix_B_ptr.value), "Free MatB")
            except Exception: pass
        if d_matrix_C_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_matrix_C_ptr.value), "Free MatC")
            except Exception: pass
        if d_matrix_D_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_matrix_D_ptr.value), "Free MatD")
            except Exception: pass
        if d_matrix_E_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_matrix_E_ptr.value), "Free MatE")
            except Exception: pass
        
            except Exception: pass
        if pool_gpu_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(pool_gpu_ptr.value), "Free Pool")
            except Exception: pass
        if d_thread_timing_buffer_ptr.value:
            try: CU_CHECK(local_libcuda.cuMemFree_v2(d_thread_timing_buffer_ptr.value), "Free Thread Timing Buffer")
            except Exception: pass

        if cu_module:
            CU_CHECK(local_libcuda.cuModuleUnload(cu_module), "cuModuleUnload")
        if cu_context:
            CU_CHECK(local_libcuda.cuCtxDestroy_v2(cu_context), "cuCtxDestroy")
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
