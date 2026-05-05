#include <stdio.h>
#include <stdlib.h>

#define INF      -1
#define BLOCK_SIZE 256

// One thread per node in the current frontier
// Expands all neighbors simultaneously
__global__ void bfs_frontier_kernel(
    int*  row_ptr,       // CSR row pointers
    int*  col_indices,   // CSR column indices
    int*  distances,     // distance from source (-1 = unvisited)
    int*  frontier,      // current frontier nodes
    int*  next_frontier, // next frontier nodes
    int   frontier_size,
    int*  next_size,     // output: size of next frontier
    int   current_dist
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int node = frontier[tid];

    // Expand all neighbors of this node
    for (int e = row_ptr[node]; e < row_ptr[node + 1]; e++) {
        int neighbor = col_indices[e];

        // If unvisited — add to next frontier
        if (distances[neighbor] == INF) {
            distances[neighbor] = current_dist + 1;

            // Atomic add to get unique position in next_frontier
            int pos = atomicAdd(next_size, 1);
            next_frontier[pos] = neighbor;
        }
    }
}

// Mark gap nodes — regs with no path to any proc
__global__ void mark_gaps_kernel(
    int* distances,
    int* gap_flags,     // 1 = gap, 0 = covered
    int  num_regs,
    int  num_procs,
    int  n              // total nodes
) {
    int proc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (proc_idx >= num_procs) return;

    // Procedures start at index num_regs in our node layout
    int node_idx = num_regs + proc_idx;
    gap_flags[proc_idx] = (distances[node_idx] == INF) ? 1 : 0;
}

#ifndef REGMAP_LIBRARY
int main() {
    // Graph setup
    // Nodes 0..num_regs-1         = regulations
    // Nodes num_regs..n-1         = procedures
    int num_regs  = 500;
    int num_procs = 500;
    int n         = num_regs + num_procs;

    printf("\nregmap — GPU BFS Gap Detection\n");
    printf("Nodes: %d (%d regs + %d procs)\n", n, num_regs, num_procs);

    // Build a sparse adjacency graph in CSR
    // Simulate compliance edges
    int max_edges = n * 15;
    int* h_row_ptr    = (int*)calloc(n + 1, sizeof(int));
    int* h_col_indices= (int*)malloc(max_edges * sizeof(int));
    int  edge_count   = 0;

    srand(42);

    // Count edges per node first
    for (int i = 0; i < num_regs; i++) {
        int degree = 5 + rand() % 10;  // 5-15 edges per reg
        for (int e = 0; e < degree; e++) {
            int proc = num_regs + rand() % num_procs;
            h_col_indices[edge_count++] = proc;
            h_row_ptr[i + 1]++;
        }
    }

    // Force some gaps — regs with NO edges
    // These should show up as gaps
    h_row_ptr[10 + 1] = 0;  // REG-10 has no edges → GAP
    h_row_ptr[42 + 1] = 0;  // REG-42 has no edges → GAP
    h_row_ptr[99 + 1] = 0;  // REG-99 has no edges → GAP

    // Build proper row_ptr with prefix sum
    for (int i = 1; i <= n; i++)
        h_row_ptr[i] += h_row_ptr[i - 1];

    int total_edges = h_row_ptr[num_regs];

    // Force known direct compliance edges
    // REG-0 → PROC-0 (direct coverage)
    // REG-1 → PROC-1 (direct coverage)
    h_col_indices[h_row_ptr[0]] = num_regs + 0;
    h_col_indices[h_row_ptr[1]] = num_regs + 1;

    printf("Edges: %d\n", total_edges);

    // GPU allocations
    int* d_row_ptr;
    int* d_col_indices;
    int* d_distances;
    int* d_frontier;
    int* d_next_frontier;
    int* d_next_size;
    int* d_gap_flags;

    cudaMalloc(&d_row_ptr,      (n + 1)      * sizeof(int));
    cudaMalloc(&d_col_indices,  total_edges  * sizeof(int));
    cudaMalloc(&d_distances,    n            * sizeof(int));
    cudaMalloc(&d_frontier,     n            * sizeof(int));
    cudaMalloc(&d_next_frontier,n            * sizeof(int));
    cudaMalloc(&d_next_size,                   sizeof(int));
    cudaMalloc(&d_gap_flags,    num_procs    * sizeof(int));

    cudaMemcpy(d_row_ptr,     h_row_ptr,     (n + 1)     * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, h_col_indices, total_edges * sizeof(int), cudaMemcpyHostToDevice);

    // Run BFS from every regulation
    // In real regmap this runs per regulation of interest
    // Here we demo from REG-0
    int source = 0;

    // Initialize distances to INF (-1)
    int* h_distances = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) h_distances[i] = INF;
    h_distances[source] = 0;

    cudaMemcpy(d_distances, h_distances, n * sizeof(int), cudaMemcpyHostToDevice);

    // Initial frontier = source node
    int h_frontier[1] = {source};
    cudaMemcpy(d_frontier, h_frontier, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int frontier_size = 1;
    int current_dist  = 0;
    int h_next_size   = 0;

    printf("\nBFS from REG-%d:\n", source);

    while (frontier_size > 0) {
        // Reset next frontier size
        cudaMemset(d_next_size, 0, sizeof(int));

        int blocks = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bfs_frontier_kernel<<<blocks, BLOCK_SIZE>>>(
            d_row_ptr, d_col_indices,
            d_distances,
            d_frontier, d_next_frontier,
            frontier_size, d_next_size,
            current_dist
        );
        cudaDeviceSynchronize();

        // Get next frontier size
        cudaMemcpy(&h_next_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost);

        printf("  Distance %d: %d nodes reached\n", current_dist + 1, h_next_size);

        // Swap frontiers
        int* temp    = d_frontier;
        d_frontier   = d_next_frontier;
        d_next_frontier = temp;

        frontier_size = h_next_size;
        current_dist++;

        if (current_dist > 10) break;  // safety limit
    }

    // Detect gaps — procedures unreachable from source
    int gap_blocks = (num_procs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mark_gaps_kernel<<<gap_blocks, BLOCK_SIZE>>>(
        d_distances, d_gap_flags,
        num_regs, num_procs, n
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy results back
    cudaMemcpy(h_distances, d_distances, n * sizeof(int), cudaMemcpyDeviceToHost);
    int* h_gap_flags = (int*)malloc(num_procs * sizeof(int));
    cudaMemcpy(h_gap_flags, d_gap_flags, num_procs * sizeof(int), cudaMemcpyDeviceToHost);

    // Count gaps
    int gap_count  = 0;
    int covered    = 0;
    for (int i = 0; i < num_procs; i++) {
        if (h_gap_flags[i] == 1) gap_count++;
        else covered++;
    }

    // Verify known edges
    printf("\nVerifying known direct coverage:\n");
    printf("REG-0 → PROC-0: distance=%d %s\n",
        h_distances[num_regs + 0],
        h_distances[num_regs + 0] == 1 ? "(direct)" :
        h_distances[num_regs + 0] == INF ? "(GAP)" : "(indirect)");

    printf("REG-0 → PROC-1: distance=%d %s\n",
        h_distances[num_regs + 1],
        h_distances[num_regs + 1] == 1 ? "(direct)" :
        h_distances[num_regs + 1] == INF ? "(GAP)" : "(indirect)");

    printf("\nGap Analysis from REG-%d:\n", source);
    printf("Procedures covered: %d\n", covered);
    printf("Procedures gaps:    %d\n", gap_count);
    printf("Coverage:           %.1f%%\n",
        100.0f * covered / num_procs);
    printf("Kernel time:        %.4f ms\n", ms);

    // Cleanup
    cudaFree(d_row_ptr);
    cudaFree(d_col_indices);
    cudaFree(d_distances);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_next_size);
    cudaFree(d_gap_flags);

    free(h_row_ptr);
    free(h_col_indices);
    free(h_distances);
    free(h_gap_flags);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nBFS OK\n");
    return 0;
}
#endif  // REGMAP_LIBRARY
