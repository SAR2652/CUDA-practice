#include <stdio.h>
#include <cuda.h> 

#define CHECK_CUDA(call)
{
    cudaError_t err = call;
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s (at %s:%d)\n", cudaGetErrorString(err),
        __FILE__, __LINE__);
        return -1;
    }
}

__global__ void debug_kernel() {
printf("Thread (%d, %d) in Block (%d, %d)\n",
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
}


__global__ void add_arrays(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
        printf("GPU: c[%d] = %d + %d = %d\n", i, a[i], b[i], c[i]);
    }
}

int main() {
    int n = 10;
    int a[n], b[n], c[n];
    int *d_a, *d_b, *d_c;

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = n - i;
        c[i] = 0;
    }

    for (int i = 0; i < n; i++) {
        printf("C[i] = %d\n", c[i]);
    }

    CHECK_CUDA(cudaMalloc((void **) &d_a, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **) &d_b, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **) &d_c, n * sizeof(int)));

    CHECK_CUDA(cudaMemset(d_c, -1, n * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 2;
    int numBlocks = (n + blockSize - 1) / blockSize;
    // (10 + 2 - 1) / 2 = 5
    add_arrays<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n); // <<<5, 2>>>
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}