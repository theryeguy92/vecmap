#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// Step 1: compute in-degree for every node
__global__ void compute_indegree_kernel(
    int* col_indices,    // CSR column indices
    int* in_degree,      // output: in-degree per node
    int  total_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_edges) return;

    // Every edge col_indices[tid] increments that node's in-degree
    atomicAdd(&in_degree[col_indices[tid]], 1);
}

// Step 2: find nodes with in-degree 0 → add to frontier
__global__ void find_zero_indegree_kernel(
    int* in_degree,
    int* frontier,
    int* frontier_size,
    int* processed,
    int  n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    if (in_degree[tid] == 0 && processed[tid] == 0) {
        processed[tid] = 1;
        int pos = atomicAdd(frontier_size, 1);
        frontier[pos] = tid;
    }
}

// Step 3: process frontier — reduce neighbor in-degrees
__global__ void process_frontier_kernel(
    int* row_ptr,
    int* col_indices,
    int* in_degree,
    int* frontier,
    int* topo_order,
    int* topo_size,
    int  frontier_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int node = frontier[tid];

    // Add to topological order
    int pos = atomicAdd(topo_size, 1);
    topo_order[pos] = node;

    // Reduce in-degree of all neighbors
    for (int e = row_ptr[node]; e < row_ptr[node + 1]; e++) {
        int neighbor = col_indices[e];
        atomicSub(&in_degree[neighbor], 1);
    }
}

#ifndef REGMAP_LIBRARY
int main() {
    int num_regs  = 500;
    int num_procs = 500;
    int n         = num_regs + num_procs;

    printf("\nregmap — GPU Topological Sort\n");
    printf("Nodes: %d (%d regs + %d procs)\n", n, num_regs, num_procs);

    // Build directed graph simulating regulatory hierarchy
    // Edges go: CFR → DOE Order → Procedure → Work Instruction
    // In our graph: lower index regs → higher index regs → procs

    int max_edges  = n * 10;
    int* h_row_ptr     = (int*)calloc(n + 1, sizeof(int));
    int* h_col_indices = (int*)malloc(max_edges * sizeof(int));
    int  edge_count    = 0;

    srand(42);

    // Regulations point to procedures (directed edges)
    for (int i = 0; i < num_regs; i++) {
        int degree = 3 + rand() % 5;
        h_row_ptr[i + 1] = degree;
        for (int e = 0; e < degree; e++) {
            // Point to procedures (higher tier)
            int target = num_regs + rand() % num_procs;
            h_col_indices[edge_count++] = target;
        }
    }

    // Some regs point to other regs (CFR → DOE Order)
    for (int i = 0; i < num_regs / 4; i++) {
        int target = num_regs / 4 + rand() % (num_regs * 3 / 4);
        h_col_indices[edge_count++] = target;
        h_row_ptr[i + 1]++;
    }

    // Force known hierarchy
    // REG-0 (CFR) → REG-100 (DOE Order) → PROC-0
    h_col_indices[edge_count++] = 100;          // REG-0 → REG-100
    h_col_indices[edge_count++] = num_regs + 0; // REG-0 → PROC-0
    h_row_ptr[0 + 1] += 2;

    h_col_indices[edge_count++] = num_regs + 1; // REG-100 → PROC-1
    h_row_ptr[100 + 1]++;

    // Build prefix sum for row_ptr
    for (int i = 1; i <= n; i++)
        h_row_ptr[i] += h_row_ptr[i - 1];

    int total_edges = h_row_ptr[num_regs];
    printf("Edges: %d\n\n", total_edges);

    // GPU allocations
    int* d_row_ptr;
    int* d_col_indices;
    int* d_in_degree;
    int* d_frontier;
    int* d_topo_order;
    int* d_topo_size;
    int* d_frontier_size;
    int* d_processed;

    cudaMalloc(&d_row_ptr,      (n + 1)      * sizeof(int));
    cudaMalloc(&d_col_indices,  total_edges  * sizeof(int));
    cudaMalloc(&d_in_degree,    n            * sizeof(int));
    cudaMalloc(&d_frontier,     n            * sizeof(int));
    cudaMalloc(&d_topo_order,   n            * sizeof(int));
    cudaMalloc(&d_topo_size,                   sizeof(int));
    cudaMalloc(&d_frontier_size,               sizeof(int));
    cudaMalloc(&d_processed,    n            * sizeof(int));

    cudaMemcpy(d_row_ptr,     h_row_ptr,     (n + 1)      * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, h_col_indices, total_edges  * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_in_degree,    0, n * sizeof(int));
    cudaMemset(d_topo_order,   0, n * sizeof(int));
    cudaMemset(d_topo_size,    0, sizeof(int));
    cudaMemset(d_processed,    0, n * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Step 1: compute in-degrees
    int edge_blocks = (total_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_indegree_kernel<<<edge_blocks, BLOCK_SIZE>>>(
        d_col_indices, d_in_degree, total_edges
    );
    cudaDeviceSynchronize();

    // Step 2: find initial zero in-degree nodes
    int node_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMemset(d_frontier_size, 0, sizeof(int));

    find_zero_indegree_kernel<<<node_blocks, BLOCK_SIZE>>>(
        d_in_degree, d_frontier, d_frontier_size, d_processed, n
    );
    cudaDeviceSynchronize();

    // Step 3: process level by level
    int h_frontier_size = 0;
    int h_topo_size     = 0;
    int level           = 0;

    printf("Topological levels:\n");

    while (true) {
        cudaMemcpy(&h_frontier_size, d_frontier_size,
            sizeof(int), cudaMemcpyDeviceToHost);

        if (h_frontier_size == 0) break;

        printf("  Level %d: %d nodes\n", level, h_frontier_size);

        // Process current frontier
        int f_blocks = (h_frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        process_frontier_kernel<<<f_blocks, BLOCK_SIZE>>>(
            d_row_ptr, d_col_indices,
            d_in_degree,
            d_frontier,
            d_topo_order, d_topo_size,
            h_frontier_size
        );
        cudaDeviceSynchronize();

        // Find next frontier
        cudaMemset(d_frontier_size, 0, sizeof(int));
        find_zero_indegree_kernel<<<node_blocks, BLOCK_SIZE>>>(
            d_in_degree, d_frontier, d_frontier_size, d_processed, n
        );
        cudaDeviceSynchronize();

        level++;
        if (level > 20) break;  // safety limit
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy results back
    cudaMemcpy(&h_topo_size, d_topo_size, sizeof(int), cudaMemcpyDeviceToHost);
    int* h_topo_order = (int*)malloc(n * sizeof(int));
    cudaMemcpy(h_topo_order, d_topo_order, n * sizeof(int), cudaMemcpyDeviceToHost);

    int* h_in_degree = (int*)malloc(n * sizeof(int));
    cudaMemcpy(h_in_degree, d_in_degree, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Check for circular dependencies
    int circular = 0;
    for (int i = 0; i < n; i++)
        if (h_in_degree[i] > 0) circular++;

    printf("\nResults:\n");
    printf("Nodes processed:       %d / %d\n", h_topo_size, n);
    printf("Levels (hierarchy depth): %d\n", level);
    printf("Circular dependencies: %d\n", circular);
    printf("Kernel time:           %.4f ms\n", ms);

    // Show first 10 nodes in topological order
    printf("\nFirst 10 nodes in topological order:\n");
    for (int i = 0; i < 10 && i < h_topo_size; i++) {
        int node = h_topo_order[i];
        if (node < num_regs)
            printf("  [%d] REG-%d\n", i, node);
        else
            printf("  [%d] PROC-%d\n", i, node - num_regs);
    }

    if (circular > 0)
        printf("\n⚠️  WARNING: %d circular dependencies detected!\n", circular);
    else
        printf("\n✅ No circular dependencies — clean hierarchy\n");

    // Cleanup
    cudaFree(d_row_ptr);      cudaFree(d_col_indices);
    cudaFree(d_in_degree);    cudaFree(d_frontier);
    cudaFree(d_topo_order);   cudaFree(d_topo_size);
    cudaFree(d_frontier_size);cudaFree(d_processed);

    free(h_row_ptr);    free(h_col_indices);
    free(h_topo_order); free(h_in_degree);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nTopological Sort OK\n");
    return 0;
}
#endif  // REGMAP_LIBRARY
