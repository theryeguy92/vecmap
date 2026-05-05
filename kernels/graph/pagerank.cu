#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE   256
#define DAMPING      0.85f
#define MAX_ITER     100
#define CONVERGENCE  1e-6f

// One iteration of PageRank
// Each thread handles one node
__global__ void pagerank_iter_kernel(
    int*   row_ptr,        // CSR row pointers
    int*   col_indices,    // CSR column indices
    float* rank,           // current ranks
    float* new_rank,       // output: updated ranks
    int*   out_degree,     // out-degree per node
    int    n,
    float  damping
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n) return;

    float sum = 0.0f;

    // Sum contributions from all neighbors pointing TO this node
    // Note: in CSR format row_ptr/col_indices store outgoing edges
    // For PageRank we need incoming edges
    // We handle this by each node scanning all edges (simplified)
    for (int src = 0; src < n; src++) {
        for (int e = row_ptr[src]; e < row_ptr[src + 1]; e++) {
            if (col_indices[e] == node) {
                int deg = row_ptr[src + 1] - row_ptr[src];
                if (deg > 0)
                    sum += rank[src] / deg;
            }
        }
    }

    new_rank[node] = (1.0f - damping) / n + damping * sum;
}

// Optimized: precompute incoming edges (transpose graph)
// Each thread handles one node using transposed CSR
__global__ void pagerank_iter_transposed_kernel(
    int*   t_row_ptr,      // transposed CSR row pointers
    int*   t_col_indices,  // transposed CSR column indices
    float* rank,           // current ranks
    float* new_rank,       // output ranks
    int*   out_degree,     // original out-degree per node
    int    n,
    float  damping
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n) return;

    float sum = 0.0f;

    // Incoming edges to this node (from transposed graph)
    for (int e = t_row_ptr[node]; e < t_row_ptr[node + 1]; e++) {
        int src = t_col_indices[e];
        int deg = out_degree[src];
        if (deg > 0)
            sum += rank[src] / (float)deg;
    }

    new_rank[node] = (1.0f - damping) / (float)n + damping * sum;
}

// Check convergence — max difference between old and new ranks
__global__ void convergence_kernel(
    float* rank,
    float* new_rank,
    float* max_diff,
    int    n
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n) return;

    float diff = fabsf(new_rank[node] - rank[node]);
    atomicMax((int*)max_diff, __float_as_int(diff));
}

#ifndef REGMAP_LIBRARY
int main() {
    int num_regs  = 500;
    int num_procs = 500;
    int n         = num_regs + num_procs;

    printf("\nregmap — GPU PageRank\n");
    printf("Nodes: %d (%d regs + %d procs)\n", n, num_regs, num_procs);

    // Build directed compliance graph
    int max_edges      = n * 10;
    int* h_row_ptr     = (int*)calloc(n + 1, sizeof(int));
    int* h_col_indices = (int*)malloc(max_edges * sizeof(int));
    int  edge_count    = 0;

    srand(42);

    // Regulations point to procedures
    for (int i = 0; i < num_regs; i++) {
        int degree = 3 + rand() % 8;
        for (int e = 0; e < degree; e++) {
            int target = num_regs + rand() % num_procs;
            h_col_indices[edge_count++] = target;
            h_row_ptr[i + 1]++;
        }
    }

    // Some regs reference other regs (hierarchy)
    for (int i = 0; i < num_regs / 3; i++) {
        int target = rand() % num_regs;
        if (target != i) {
            h_col_indices[edge_count++] = target;
            h_row_ptr[i + 1]++;
        }
    }

    // Force known high-importance regulations
    // REG-0 = "CFR master regulation" referenced by many
    for (int i = 1; i < 50; i++) {
        h_col_indices[edge_count++] = 0;  // many regs point to REG-0
        h_row_ptr[i + 1]++;
    }

    // Build prefix sum
    for (int i = 1; i <= n; i++)
        h_row_ptr[i] += h_row_ptr[i - 1];

    int total_edges = h_row_ptr[n];
    printf("Edges: %d\n", total_edges);

    // Build out-degree array
    int* h_out_degree = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        h_out_degree[i] = h_row_ptr[i + 1] - h_row_ptr[i];

    // Build transposed graph for efficient PageRank
    // Transposed: edge[j→i] becomes edge[i←j]
    int* h_t_row_ptr     = (int*)calloc(n + 1, sizeof(int));
    int* h_t_col_indices = (int*)malloc(total_edges * sizeof(int));

    // Count in-degrees
    for (int i = 0; i < total_edges; i++)
        h_t_row_ptr[h_col_indices[i] + 1]++;

    // Prefix sum
    for (int i = 1; i <= n; i++)
        h_t_row_ptr[i] += h_t_row_ptr[i - 1];

    // Fill transposed edges
    int* temp_ptr = (int*)calloc(n, sizeof(int));
    for (int src = 0; src < n; src++) {
        for (int e = h_row_ptr[src]; e < h_row_ptr[src + 1]; e++) {
            int dst = h_col_indices[e];
            int pos = h_t_row_ptr[dst] + temp_ptr[dst];
            h_t_col_indices[pos] = src;
            temp_ptr[dst]++;
        }
    }
    free(temp_ptr);

    // GPU allocations
    int*   d_t_row_ptr;
    int*   d_t_col_indices;
    int*   d_out_degree;
    float* d_rank;
    float* d_new_rank;
    float* d_max_diff;

    cudaMalloc(&d_t_row_ptr,     (n + 1)      * sizeof(int));
    cudaMalloc(&d_t_col_indices, total_edges  * sizeof(int));
    cudaMalloc(&d_out_degree,    n            * sizeof(int));
    cudaMalloc(&d_rank,          n            * sizeof(float));
    cudaMalloc(&d_new_rank,      n            * sizeof(float));
    cudaMalloc(&d_max_diff,                     sizeof(float));

    cudaMemcpy(d_t_row_ptr,     h_t_row_ptr,     (n+1)*sizeof(int),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_col_indices, h_t_col_indices, total_edges*sizeof(int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_degree,    h_out_degree,    n*sizeof(int),             cudaMemcpyHostToDevice);

    // Initialize ranks to 1/n
    float* h_rank = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h_rank[i] = 1.0f / n;
    cudaMemcpy(d_rank, h_rank, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int   blocks     = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int   iter       = 0;
    float h_max_diff = 1.0f;

    printf("\nRunning PageRank (damping=%.2f)...\n", DAMPING);

    while (iter < MAX_ITER && h_max_diff > CONVERGENCE) {
        // Reset max_diff
        cudaMemset(d_max_diff, 0, sizeof(float));

        // One PageRank iteration
        pagerank_iter_transposed_kernel<<<blocks, BLOCK_SIZE>>>(
            d_t_row_ptr, d_t_col_indices,
            d_rank, d_new_rank,
            d_out_degree, n, DAMPING
        );

        // Check convergence
        convergence_kernel<<<blocks, BLOCK_SIZE>>>(
            d_rank, d_new_rank, d_max_diff, n
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_max_diff, d_max_diff, sizeof(float), cudaMemcpyDeviceToHost);

        // Swap rank buffers
        float* temp = d_rank;
        d_rank      = d_new_rank;
        d_new_rank  = temp;

        iter++;

        if (iter % 10 == 0)
            printf("  Iteration %d: max_diff=%.8f\n", iter, h_max_diff);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy final ranks back
    cudaMemcpy(h_rank, d_rank, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nConverged after %d iterations\n", iter);
    printf("Kernel time: %.4f ms\n\n", ms);

    // Find top 10 most important regulations
    // Simple selection sort for top 10
    int   top_idx[10];
    float top_rank[10];
    bool  used[1000] = {false};

    for (int t = 0; t < 10; t++) {
        float best      = -1.0f;
        int   best_idx  = -1;
        for (int i = 0; i < n; i++) {
            if (!used[i] && h_rank[i] > best) {
                best     = h_rank[i];
                best_idx = i;
            }
        }
        top_idx[t]  = best_idx;
        top_rank[t] = best;
        used[best_idx] = true;
    }

    printf("Top 10 most critical nodes:\n");
    printf("%-6s %-15s %-12s\n", "Rank", "Node", "PageRank Score");
    printf("─────────────────────────────────\n");
    for (int t = 0; t < 10; t++) {
        int   idx  = top_idx[t];
        float rank = top_rank[t];
        char node_name[32];
        if (idx < num_regs)
            snprintf(node_name, sizeof(node_name), "REG-%d", idx);
        else
            snprintf(node_name, sizeof(node_name), "PROC-%d", idx - num_regs);
        printf("%-6d %-15s %.8f\n", t+1, node_name, rank);
    }

    printf("\nREG-0 (forced high importance): rank=%.8f\n", h_rank[0]);

    // Cleanup
    cudaFree(d_t_row_ptr);    cudaFree(d_t_col_indices);
    cudaFree(d_out_degree);   cudaFree(d_rank);
    cudaFree(d_new_rank);     cudaFree(d_max_diff);

    free(h_t_row_ptr);    free(h_t_col_indices);
    free(h_out_degree);   free(h_rank);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nPageRank OK\n");
    return 0;
}
#endif  // REGMAP_LIBRARY
