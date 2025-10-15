NVCC = nvcc
FLAGS = -arch=sm_86 -Xptxas -O0 --compiler-options '-std=c++11' --compiler-bindir /usr/bin/gcc-10 -lstdc++

FILES = fill.cu l1-dtlb-cap.cu l1-itlb-cap.cu l2-utlb-set.cu l2-utlb-hash.cu l3-utlb-set.cu l3-utlb-hash.cu 
BINS = $(FILES:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(FLAGS) $< -o $@

clean:
	rm -f $(BINS)

.PHONY: all clean
