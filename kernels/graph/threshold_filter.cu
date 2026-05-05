#include <stdio.h>
#include <stdlib.h>


// Step 1: count how many edges each regulation has above threshold
// One thread per regulation

__global__ void count_edges_kernel(
    float* similarity_matrix,   // [num_regs x num_procs]
    int*   edge_counts,         // output: how many edges per reg
    int    num_regs,
    int    num_procs,
    float  threshold    
) {
    int reg_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (reg_idx >= num_regs) return;

    int count = 0;
    for (int p =0; p < num_procs; p++) {
        if (similarity_matrix[reg_idx * num_procs + p] >= threshold)
            count++;
    }
    edge_counts[reg_idx] = count;
}

// Step 2: fill CSR values and column indices
// One thread per regulation
__global__ void fill_csr_kernel(
    float* similarity_matrix,   // [num_regs x num_procs]
    int*   row_ptr,             // CSR row pointers
    int*   col_indices,         // CSR column indices
    float* values,              // CSR edge weights
    int    num_regs,
    int    num_procs,
    float  threshold
) {
    int reg_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (reg_idx >= num_regs) return;

    int write_pos = row_ptr[reg_idx];

    for (int p = 0; p < num_procs; p++) {
        float sim = similarity_matrix[reg_idx * num_procs + p];
        if (sim >= threshold) {
            col_indices[write_pos] = p;
            values[write_pos]      = sim;
            write_pos++;
        }
    }
}

// CPU prefix sum to build row_ptr from edge_counts
void prefix_sum(int* counts, int* ptr, int n) {
    ptr[0] = 0;
    for (int i = 0; i < n; i++)
        ptr[i + 1] = ptr[i] + counts[i];
}

#ifndef REGMAP_LIBRARY
int main() {
    int   num_regs  = 10000;
    int   num_procs = 500;
    int   embed_dim = 768;  // kept for context
    float threshold = 0.75f;

    // --- Build a fake similarity matrix for testing ---
    int   size_C    = num_regs * num_procs;
    float* h_C      = (float*)malloc(size_C * sizeof(float));

    srand(42);
    for (int i = 0; i < size_C; i++)
        h_C[i] = (float)rand() / RAND_MAX;

    // Force a few known edges for verification
    h_C[0 * num_procs + 0] = 0.91f;  // REG-0 → PROC-0
    h_C[0 * num_procs + 3] = 0.87f;  // REG-0 → PROC-3
    h_C[1 * num_procs + 1] = 0.83f;  // REG-1 → PROC-1
    h_C[2 * num_procs + 2] = 0.76f;  // REG-2 → PROC-2

    // --- GPU allocations ---
    float* d_C;
    cudaMalloc(&d_C, size_C * sizeof(float));
    cudaMemcpy(d_C, h_C, size_C * sizeof(float), cudaMemcpyHostToDevice);

    int* d_edge_counts;
    cudaMalloc(&d_edge_counts, num_regs * sizeof(int));
    cudaMemset(d_edge_counts, 0, num_regs * sizeof(int));

    // --- Step 1: count edges per reg ---
    int threads = 256;
    int blocks  = (num_regs + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    count_edges_kernel<<<blocks, threads>>>(
        d_C, d_edge_counts, num_regs, num_procs, threshold
    );
    cudaDeviceSynchronize();

    // --- Copy counts back to CPU for prefix sum ---
    int* h_edge_counts = (int*)malloc(num_regs * sizeof(int));
    int* h_row_ptr     = (int*)malloc((num_regs + 1) * sizeof(int));

    cudaMemcpy(h_edge_counts, d_edge_counts, num_regs * sizeof(int), cudaMemcpyDeviceToHost);

    // --- Prefix sum on CPU → builds row_ptr ---
    prefix_sum(h_edge_counts, h_row_ptr, num_regs);

    int total_edges = h_row_ptr[num_regs];
    printf("\nregmap — threshold_filter\n");
    printf("Matrix:      %d regs x %d procs\n", num_regs, num_procs);
    printf("Threshold:   %.2f\n", threshold);
    printf("Total pairs: %d\n", size_C);
    printf("Edges kept:  %d (%.2f%% of matrix)\n",
        total_edges, 100.0f * total_edges / size_C);

    // --- GPU allocations for CSR ---
    int*   d_row_ptr;
    int*   d_col_indices;
    float* d_values;

    cudaMalloc(&d_row_ptr,     (num_regs + 1) * sizeof(int));
    cudaMalloc(&d_col_indices, total_edges    * sizeof(int));
    cudaMalloc(&d_values,      total_edges    * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr, (num_regs + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // --- Step 2: fill CSR ---
    fill_csr_kernel<<<blocks, threads>>>(
        d_C, d_row_ptr, d_col_indices, d_values,
        num_regs, num_procs, threshold
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // --- Verify known edges ---
    int*   h_col_indices = (int*)malloc(total_edges * sizeof(int));
    float* h_values      = (float*)malloc(total_edges * sizeof(float));

    cudaMemcpy(h_col_indices, d_col_indices, total_edges * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values,      d_values,      total_edges * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nVerifying known edges:\n");
    printf("REG-0 edges:\n");
    for (int e = h_row_ptr[0]; e < h_row_ptr[1]; e++)
        printf("  → PROC-%d  sim=%.4f\n", h_col_indices[e], h_values[e]);

    printf("REG-1 edges:\n");
    for (int e = h_row_ptr[1]; e < h_row_ptr[2]; e++)
        printf("  → PROC-%d  sim=%.4f\n", h_col_indices[e], h_values[e]);

    printf("REG-2 edges:\n");
    for (int e = h_row_ptr[2]; e < h_row_ptr[3]; e++)
        printf("  → PROC-%d  sim=%.4f\n", h_col_indices[e], h_values[e]);

    printf("\nKernel time: %.4f ms\n", ms);
    printf("\nCSR graph built — ready for graph algorithms\n");

    // --- Cleanup ---
    cudaFree(d_C);
    cudaFree(d_edge_counts);
    cudaFree(d_row_ptr);
    cudaFree(d_col_indices);
    cudaFree(d_values);

    free(h_C);
    free(h_edge_counts);
    free(h_row_ptr);
    free(h_col_indices);
    free(h_values);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
#endif  // REGMAP_LIBRARY