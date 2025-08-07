GCC = gcc
G++ = g++
NVCC = nvcc

# CUDA architecture flags
ARCH_T4 = -gencode arch=compute_75,code=sm_75

vector_add:
	$(NVCC) $(ARCH_T4) -o vector_add vector_add.cu
