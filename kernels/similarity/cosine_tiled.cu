#include <stdio.h>
#include <math.h>

#define TILE_SIZE 32

__global__ void cosine_tiled_kernel(
    float* A,          // regulation embeddings [num_regs x embed_dim]
    float* B,          // procedure embeddings  [num_procs x embed_dim]
    float* C,          // output matrix         [num_regs x num_procs]
    int num_regs,
    int num_procs,
    int embed_dim
) {
    // Shared memory tiles - loaded once, reused by all threads in block
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int reg_idx  =  blockIdx.x * blockDim.x + threadIdx.x;
    int proc_idx =  blockIdx.y * blockDim.y + threadIdx.y;

    float dot    = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    // Loop over tiles instead of individual elements
    int num_tiles = (embed_dim + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {

        // Collaboratively laod tile into shared memory
        int k_idx = t * TILE_SIZE + threadIdx.y;
        if (reg_idx < num_regs && k_idx < embed_dim)
            tile_A[threadIdx.x][threadIdx.y] = A[reg_idx * embed_dim + k_idx];
        else 
            tile_A[threadIdx.x][threadIdx.y] = 0.0f;

        k_idx = t * TILE_SIZE + threadIdx.x;
        if (proc_idx < num_procs && k_idx < embed_dim)
            tile_B[threadIdx.y][threadIdx.x] = B[proc_idx * embed_dim + k_idx];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Wait for ALL threads in block to finish loading
        __syncthreads();

        // Compute on shared memory -- fast because it is shared in a block
        for (int k = 0; k < TILE_SIZE; k++) {
            dot    += tile_A[threadIdx.x][k] * tile_B[threadIdx.y][k];
            norm_a += tile_A[threadIdx.x][k] * tile_A[threadIdx.x][k];
            norm_b += tile_B[threadIdx.y][k] * tile_B[threadIdx.y][k];
        }

        // Wait before loading next tile
        __syncthreads();
    }

    if (reg_idx < num_regs && proc_idx < num_procs)
        C[reg_idx * num_procs + proc_idx] = dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f);

}

int main() {
    // Larger test to show speedup — 128 regs, 64 procs, 256-dim
    int num_regs  = 10000;
    int num_procs = 500;
    int embed_dim = 768;

    int size_A = num_regs  * embed_dim * sizeof(float);
    int size_B = num_procs * embed_dim * sizeof(float);
    int size_C = num_regs  * num_procs * sizeof(float);

    // Allocate host arrays
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    // Fill with random values
    for (int i = 0; i < num_regs  * embed_dim; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < num_procs * embed_dim; i++) h_B[i] = (float)rand() / RAND_MAX;

    // Device arrays
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch tiled kernel
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (num_regs  + TILE_SIZE - 1) / TILE_SIZE,
        (num_procs + TILE_SIZE - 1) / TILE_SIZE
    );

    cudaEventRecord(start);
    cosine_tiled_kernel<<<blocks, threads>>>(d_A, d_B, d_C, num_regs, num_procs, embed_dim);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("\nregmap — cosine_tiled kernel\n");
    printf("Matrix: %d regs x %d procs x %d dims\n", num_regs, num_procs, embed_dim);
    printf("Kernel time: %.4f ms\n", ms);
    printf("Sample C[0][0]: %.4f\n", h_C[0]);
    printf("Sample C[1][1]: %.4f\n", h_C[num_procs + 1]);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\ncosine_tiled OK\n");
    return 0;

}