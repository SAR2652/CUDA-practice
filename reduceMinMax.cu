#include <stdio.h>
#include <limits.h>
#include <random>
#include <cstdlib>
#include <algorithm>
#include "cuda_utils.h"

__global__ void reduceMinMax(int* input, int* output_min, int* output_max,
                             int N)
{
    // You can allocate only one shared variable
    extern __shared__ int sdata[];

    // Therefore allocate 2X dynamic shared memory size and manually split it
    int* sdata_min = sdata;
    int* sdata_max = sdata + blockDim.x;

    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Assign all values in a block to shared memory
    sdata_min[tid] = (i < N) ? input[i]: INT_MAX;
    sdata_max[tid] = (i < N) ? input[i]: INT_MIN;

    __syncthreads();

    for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            sdata_min[tid] = min(sdata_min[tid], sdata_min[tid + stride]);
            sdata_max[tid] = max(sdata_max[tid], sdata_max[tid + stride]);
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        output_min[blockIdx.x] = sdata_min[0];
        output_max[blockIdx.x] = sdata_max[0];
    }
}


int main()
{
    int N = 1 << 20;
    size_t size = N * sizeof(int);

    int *h_in = new int[N];

    std::random_device rd;
    std::mt19937 gen(rd());
    
    for(int i = 0 ; i < N; i++)
    {
        h_in[i] = gen();
    }

    int threadsPerBlock = 256;
    int dynamicSharedMemorySize = threadsPerBlock * sizeof(int);

    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int *h_out = new int[blocksPerGrid];

    int *d_in, *d_out_min, *d_out_max;
    CHECK_CUDA(cudaMalloc((void **) &d_in, size));
    CHECK_CUDA(cudaMalloc((void **) &d_out_min, blocksPerGrid * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **) &d_out_max, blocksPerGrid * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));
    reduceMinMax<<<blocksPerGrid, threadsPerBlock,
    2 * dynamicSharedMemorySize>>>(d_in, d_out_min, d_out_max, N);

    int *h_out_min = new int[blocksPerGrid];
    int *h_out_max = new int[blocksPerGrid];

    CHECK_CUDA(cudaMemcpy(h_out_min, d_out_min, blocksPerGrid * sizeof(int),
               cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_max, d_out_max, blocksPerGrid * sizeof(int),
               cudaMemcpyDeviceToHost));

    int minimum = INT32_MAX;
    int maximum = INT32_MIN;

    for(int i = 0; i < blocksPerGrid; i++)
    {
        if(h_out_min[i] < minimum)
        {
            minimum = h_out_min[i];
        }

        if(h_out_max[i] > maximum)
        {
            maximum = h_out_max[i];
        }
    }

    printf("Minimum = %d\n", minimum);
    printf("Maximum = %d\n", maximum);

    return 0;
}