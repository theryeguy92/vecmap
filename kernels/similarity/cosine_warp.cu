#include <stdio.h>
#include <math.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256 //8 warps per block

// Warp shuffle reduction - no shared memory needed
__device__ float warp_reduce_sum(float val) {
    // Each step passes value from thread+offset to thread
    // 5 steps covers all 32 threads in a warp
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__global__ void cosine_warp_kernel(
        float* A,        // regulation embeddings [num_regs x embed_dim]
    float* B,        // procedure embeddings  [num_procs x embed_dim]
    float* C,        // output matrix         [num_regs x num_procs]
    int num_regs,
    int num_procs,
    int embed_dim
) {
    // One warp per reg-proc pair
    int warp_id    = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id    = threadIdx.x % WARP_SIZE; // 0-31 within warp
    
    int total_pairs = num_regs * num_procs;
    if (warp_id >= total_pairs) return;

    //Map warp to reg-proc pair
    int reg_idx   = warp_id / num_procs;
    int proc_idx = warp_id % num_procs;

    float dot    = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    // Each line handles a stride of embedding dimension
    // 32 lanes split the work across embed_dim
    for (int k = lane_id; k < embed_dim; k += WARP_SIZE) {
        float a = A[reg_idx  * embed_dim + k];
        float b = B[proc_idx * embed_dim + k];
        dot    += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    // Warp shuffle reduction - all 32 lanes sum their partials
    // No shared memory, no __ syncthreads needed
    dot    = warp_reduce_sum(dot);
    norm_a = warp_reduce_sum(norm_a);
    norm_b = warp_reduce_sum(norm_b);

    // Only lane 0 writes teh final result
    if (lane_id == 0) {
        C[reg_idx * num_procs + proc_idx] = dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f);
    }

}

#ifndef REGMAP_LIBRARY
int main() {
    int num_regs  = 10000;
    int num_procs = 500;
    int embed_dim = 768;

    int size_A = num_regs  * embed_dim * sizeof(float);
    int size_B = num_procs * embed_dim * sizeof(float);
    int size_C = num_regs  * num_procs * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    for (int i = 0; i < num_regs  * embed_dim; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < num_procs * embed_dim; i++) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // One warp per pair → total warps = num_regs * num_procs
    int total_pairs  = num_regs * num_procs;
    int total_threads = total_pairs * WARP_SIZE;
    int blocks        = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEventRecord(start);
    cosine_warp_kernel<<<blocks, BLOCK_SIZE>>>(d_A, d_B, d_C, num_regs, num_procs, embed_dim);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("\nregmap — cosine_warp kernel\n");
    printf("Matrix: %d regs x %d procs x %d dims\n", num_regs, num_procs, embed_dim);
    printf("Kernel time: %.4f ms\n", ms);
    printf("Sample C[0][0]: %.4f\n", h_C[0]);
    printf("Sample C[1][1]: %.4f\n", h_C[num_procs + 1]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\ncosine_warp OK\n");
    return 0;
}
#endif  // REGMAP_LIBRARY