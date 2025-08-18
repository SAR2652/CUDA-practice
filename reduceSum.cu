#include <stdio.h>
#include <cstdlib>
#include "cuda_utils.h"


__global__ void reduceSum(int* input, int* output, int N)
{
    // This array lives in dynamic shared memory, its size will be provided
    // at runtime, not at compile time.
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    // first index under consideration
    sdata[tid] = (i < N) ? input[i]: 0;

    // Synchronize threads so that all have their first tid initialized
    __syncthreads();

    for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }

}

int main()
{
    int N = 1 << 20;
    size_t size = N * sizeof(int);

    int* h_in = new int[N];
    for(int i = 0; i < N; i++)
    {
        h_in[i] = 1;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    int *h_out = new int[blocksPerGrid];
    int dynamicSharedMemorySize = threadsPerBlock * sizeof(int);

    int *d_in, *d_out;
    CHECK_CUDA(cudaMalloc((void **) &d_in, size));
    CHECK_CUDA(cudaMalloc((void **) &d_out,
    ((N + threadsPerBlock - 1) / threadsPerBlock) * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));
    reduceSum<<<blocksPerGrid, threadsPerBlock, dynamicSharedMemorySize>>>(
        d_in, d_out, N
    );

    CHECK_CUDA(cudaMemcpy(h_out, d_out, blocksPerGrid * sizeof(int),
        cudaMemcpyHostToDevice));

    unsigned long long sum = 0;
    for(int i = 0; i < blocksPerGrid; i++)
    {
        sum += h_out[i];
    }

    printf("Sum = %lld", sum);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;

    return 0;
}