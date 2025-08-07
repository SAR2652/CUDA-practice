#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda.h>
#include <stdio.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
    { \
        printf("CUDA Error: %s (at %s:%d)\n", cudaGetErrorString(err), \
        __FILE__, __LINE__); \
        return -1; \
    } \
}

#endif