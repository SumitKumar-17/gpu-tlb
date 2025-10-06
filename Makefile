# Simple Makefile for TLB Covert Channel Experiment

# Compiler
NVCC = nvcc

# CUDA architecture for RTX 3060
ARCH = -gencode arch=compute_86,code=sm_86

# Compiler flags
NVCCFLAGS = -O3 --ptxas-options=-v

# Source and Target Files
KERNEL_SRC = channel_kernels.cu
KERNEL_PTX = channel_kernels.ptx

BASELINE_KERNEL_SRC = baseline_kernels.cu
BASELINE_KERNEL_PTX = baseline_kernels.ptx

# Default target
all: $(KERNEL_PTX) $(BASELINE_KERNEL_PTX)

# Rule to compile channel kernels to PTX
$(KERNEL_PTX): $(KERNEL_SRC) gpu_utils.py # Depend on gpu_utils in case constants change
	$(NVCC) $(NVCCFLAGS) $(ARCH) --ptx -o $@ $<

# Rule to compile baseline kernels to PTX
$(BASELINE_KERNEL_PTX): $(BASELINE_KERNEL_SRC) gpu_utils.py # Depend on gpu_utils
	$(NVCC) $(NVCCFLAGS) $(ARCH) --ptx -o $@ $<

# Phony targets
.PHONY: all clean run_baseline run_sender0 run_sender1 run_receiver

run_baseline: $(BASELINE_KERNEL_PTX) run_timing.py gpu_utils.py l3_hash_rtx3060.py
	@echo "Running baseline timing..."
	python3 run_timing.py

run_sender0: $(KERNEL_PTX) sender.py gpu_utils.py l3_hash_rtx3060.py
	@echo "Starting sender for Set 0..."
	python3 sender.py 0

run_sender1: $(KERNEL_PTX) sender.py gpu_utils.py l3_hash_rtx3060.py
	@echo "Starting sender for Set 1..."
	python3 sender.py 1

run_receiver: $(KERNEL_PTX) receiver.py gpu_utils.py l3_hash_rtx3060.py
	@echo "Starting receiver..."
	python3 receiver.py

clean:
	@echo "Cleaning up..."
	rm -f $(KERNEL_PTX) $(BASELINE_KERNEL_PTX) *.pyc __pycache__/* receiver_results.npz *.png *.npz baseline_latency.png
	@echo "Done."