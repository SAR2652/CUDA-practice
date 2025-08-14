#include <stdio.h>
#include <cstdlib>
#include "cuda_utils.h"
#define TILE_SIZE 32

__global__ void matmul(float* d_M, float* d_N, float* d_P, int M, int N,
    int K)
{
    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int alongRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int alongCol = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;
    Kdim = (K + TILE_SIZE - 1) / TILE_SIZE;

    for(int t = 0; t < Kdim; t++)
    {
        int Kcol = t * TILE_SIZE + threadIdx.x; 
        if(alongRow < M && Kcol < K)
        {
            tileA[threadIdx.y][threadIdx.x] = d_M[alongRow * K + Kcol];
        }
        else
        {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int Krow = t * TILE_SIZE + threadIdx.y;
        if(Krow < K && alongCol < N)
        {
            tileB[threadIdx.y][threadIdx.x] = d_N[Krow * N + alongCol];
        }
        else
        {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for(int i = 0; i < TILE_SIZE; i++)
        {
            value += (tileA[threadIdx.y][i] * tileB[i][threadIdx.x]);
        }

        __syncthreads();

        if(alongRow < M && alongCol < N)
        {
            d_P[alongRow * N + alongCol] = value;
        }
    }
}

int main()
{
    float *h_M, *h_N, *h_P, *d_M, *d_N, *d_P;

    int M = 1000;
    int K = 500;
    int N = 1000;

    std::srand(42);

    int M_size = M * K * sizeof(float);
    int N_size = K * N * sizeof(float);
    int P_size = M * N * sizeof(float);

    h_M = (float*) malloc(M_size);
    h_N = (float*) malloc(N_size);

    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < K; j++)
        {
            h_M[i * K + j] = rand() / RAND_MAX;
        }
    }

    for(int i = 0; i < K; i++)
    {
        for(int j = 0; j < N; j++)
        {
            h_N[i * K + j] = rand() / RAND_MAX;
        }
    }

    CHECK_CUDA(cudaMalloc((void **) &d_M, M_size));
    CHECK_CUDA(cudaMalloc((void **) &d_N, N_size));
    CHECK_CUDA(cudaMalloc((void **) &d_P, P_size));
    
    cudaMemcpy(d_M, h_M, M_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, N_size, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);     // 1024 threads
    gridDimX = (N + TILE_SIZE - 1) / TILE_SIZE;
    gridDimY = (M + TILE_SIZE - 1) / TILE_SIZE;
    dim3 gridDim(gridDimX, gridDimY);
    // This is because there are N columns when moving along the X-axis & N
    // rows when moving along the Y-axis. 
    // X-axis: ()

    matmul<<<gridDim, blockDim>>>(d_M, d_N, d_P, M, N, K);
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    
    free(h_M);
    free(h_N);

    h_P = (float*) malloc(P_size);
    cudaMemcpy(h_P, d_P, P_size, cudaMemcpyDeviceToHost);
    CHECK_CUDA(cudaFree(d_P));
    free(h_P);

}
