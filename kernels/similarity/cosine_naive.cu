// Naive kernel: one thread per reg-proc pair
#include <stdio.h>
#include <math.h>

__global__ void cosine_naive_kernel( // In CUDA, this function runs on the GPU, called from CPU
    float* A,       // regulation embeddings [num_regs x embed_dim]
    float* B,       // procedure embeddings  [num_procs x embed_dim]
    float* C,       // output matrix         [num_regs x num_procs]
    int num_regs,
    int num_procs,
    int embed_dim
) {
     int reg_idx = blockIdx.x * blockDim.x + threadIdx.x; 
     int proc_idx = blockIdx.y * blockDim.y + threadIdx.y;

     if (reg_idx >= num_regs || proc_idx >= num_procs) return;

     float dot    = 0.0f;
     float norm_a = 0.0f;
     float norm_b = 0.0f;

     for (int k = 0; k < embed_dim; k++){
        float a = A[reg_idx * embed_dim + k];
        float b = B[proc_idx * embed_dim + k];
        dot    += a * b;
        norm_a += a * a;
        norm_b += b * b;
     }

     C[reg_idx * num_procs + proc_idx] = dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f);

}

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

    dim3 threads(16, 16);
    dim3 blocks(
        (num_regs  + threads.x - 1) / threads.x,
        (num_procs + threads.y - 1) / threads.y
    );

    cudaEventRecord(start);
    cosine_naive_kernel<<<blocks, threads>>>(d_A, d_B, d_C, num_regs, num_procs, embed_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("\nregmap — cosine_naive kernel\n");
    printf("Matrix: %d regs x %d procs x %d dims\n", num_regs, num_procs, embed_dim);
    printf("Kernel time: %.4f ms\n", ms);
    printf("Sample C[0][0]: %.4f\n", h_C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\ncosine_naive OK\n");
    return 0;

}