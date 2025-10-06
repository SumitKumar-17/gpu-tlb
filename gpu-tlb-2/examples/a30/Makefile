NVCC = nvcc
FLAGS = -arch=sm_80 -Xptxas -O0 -g -G

FILES = fill.cu l1-dtlb-cap.cu l1-itlb-cap.cu l2-utlb-set.cu l2-utlb-set-static.cu utlb-topology.cu l3-utlb-set-static.cu tlb-timing.cu receiver.cu sender.cu covert_channel.cu
BINS = $(FILES:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(FLAGS) $< -o $@

clean:
	rm -f $(BINS)

.PHONY: all clean

