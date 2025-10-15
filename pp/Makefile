# Simple Makefile for TLB Covert Channel Experiment

# Compiler
NVCC = nvcc

# CUDA architecture for RTX 3060
ARCH = -gencode arch=compute_86,code=sm_86

# Compiler flags
NVCCFLAGS = -O3 --ptxas-options=-v

# Use consolidated kernel file
KERNEL_SRC = pp_baseline_kernels.cu
KERNEL_PTX = pp_baseline_kernels.ptx

# Default target
all: $(KERNEL_PTX)

# Rule to compile kernels to PTX
$(KERNEL_PTX): $(KERNEL_SRC) gpu_utils_pp.py # Depend on utils
	$(NVCC) $(NVCCFLAGS) $(ARCH) --ptx -o $@ $<

# Phony targets
.PHONY: all clean compile run_pp_baseline run_sender_pp0 run_sender_pp1 run_receiver_pp

compile: all

run_pp_baseline: $(KERNEL_PTX) run_probe_baseline.py gpu_utils_pp.py l3_hash_rtx3060.py
	@echo "Running Prime+Probe baseline timing..."
	python3 run_probe_baseline.py

# These targets use the new sender/receiver script names and depend on the consolidated PTX
run_sender_pp0: $(KERNEL_PTX) sender_primeprobe.py gpu_utils_pp.py l3_hash_rtx3060.py
	@echo "Starting Prime+Probe sender for Bit 0 (Contending Set ${NON_TARGET_SET_COVERT})..."
	python3 sender_primeprobe.py 0

run_sender_pp1: $(KERNEL_PTX) sender_primeprobe.py gpu_utils_pp.py l3_hash_rtx3060.py
	@echo "Starting Prime+Probe sender for Bit 1 (Contending Set ${TARGET_SET_COVERT})..."
	python3 sender_primeprobe.py 1

run_receiver_pp: $(KERNEL_PTX) receiver_primeprobe.py gpu_utils_pp.py l3_hash_rtx3060.py
	@echo "Starting Prime+Probe receiver..."
	python3 receiver_primeprobe.py

clean:
	@echo "Cleaning up..."
	rm -f $(KERNEL_PTX) *.pyc __pycache__/* receiver_*.npz *.png *.npz pp_baseline_latency.png
	@echo "Done."