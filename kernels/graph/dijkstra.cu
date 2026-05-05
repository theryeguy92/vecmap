#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE  256
#define INF_DIST    1e9f

// Parallel label-correcting Dijkstra.
// Each thread owns one node; active nodes relax all outgoing edges.
// Works for non-negative weights (edge weight = 1.0 - similarity).
//
// atomicMin on (int*)&dist[v] with __float_as_int is safe for
// non-negative floats: IEEE 754 bit patterns preserve ordering
// when sign bit is 0, and our weights are in [0, 1].
__global__ void relax_kernel(
    int*   row_ptr,
    int*   col_indices,
    float* weights,
    float* dist,
    int*   predecessor,
    int*   in_frontier,     // nodes active this round
    int*   next_frontier,   // nodes updated this round (output)
    int*   any_update,      // set to 1 if any distance improved
    int    n
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n || !in_frontier[u]) return;

    for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
        int   v        = col_indices[e];
        float new_dist = dist[u] + weights[e];

        int old_bits = atomicMin((int*)&dist[v], __float_as_int(new_dist));
        if (__int_as_float(old_bits) > new_dist) {
            next_frontier[v] = 1;
            atomicOr(any_update, 1);
            // last-writer-wins — valid predecessor, path trace uses host CSR
            predecessor[v] = u;
        }
    }
}

#ifndef REGMAP_LIBRARY
int main() {
    int num_regs  = 500;
    int num_procs = 500;
    int n         = num_regs + num_procs;

    printf("\nregmap — GPU Dijkstra (Strongest Compliance Path)\n");
    printf("Nodes: %d (%d regs + %d procs)\n", n, num_regs, num_procs);

    // Directed compliance graph: regs → procs and reg → reg hierarchy.
    // Edge weight = 1.0 - similarity (lower = stronger compliance link).
    int    max_edges   = n * 15;
    int*   h_row_ptr   = (int*)calloc(n + 1, sizeof(int));
    int*   h_col       = (int*)malloc(max_edges * sizeof(int));
    float* h_weights   = (float*)malloc(max_edges * sizeof(float));
    int    edge_count  = 0;

    srand(42);

    // All edges for each reg added together to keep CSR contiguous
    for (int i = 0; i < num_regs; i++) {
        // Reg → proc edges
        int degree = 5 + rand() % 10;
        for (int e = 0; e < degree; e++) {
            int   j   = num_regs + rand() % num_procs;
            float sim = 0.5f + 0.5f * ((float)rand() / RAND_MAX);
            h_col[edge_count]     = j;
            h_weights[edge_count] = 1.0f - sim;
            h_row_ptr[i + 1]++;
            edge_count++;
        }
        // Reg → reg hierarchy (every 5th reg)
        if (i > 0 && i % 5 == 0) {
            int   j   = rand() % i;
            float sim = 0.6f + 0.3f * ((float)rand() / RAND_MAX);
            h_col[edge_count]     = j;
            h_weights[edge_count] = 1.0f - sim;
            h_row_ptr[i + 1]++;
            edge_count++;
        }
    }

    // Prefix sum
    for (int i = 1; i <= n; i++)
        h_row_ptr[i] += h_row_ptr[i - 1];

    int total_edges = h_row_ptr[n];
    printf("Edges: %d\n", total_edges);

    // GPU allocations
    int*   d_row_ptr;
    int*   d_col;
    float* d_weights;
    float* d_dist;
    int*   d_predecessor;
    int*   d_frontier;
    int*   d_next_frontier;
    int*   d_any_update;

    cudaMalloc(&d_row_ptr,       (n + 1)      * sizeof(int));
    cudaMalloc(&d_col,           total_edges  * sizeof(int));
    cudaMalloc(&d_weights,       total_edges  * sizeof(float));
    cudaMalloc(&d_dist,          n            * sizeof(float));
    cudaMalloc(&d_predecessor,   n            * sizeof(int));
    cudaMalloc(&d_frontier,      n            * sizeof(int));
    cudaMalloc(&d_next_frontier, n            * sizeof(int));
    cudaMalloc(&d_any_update,                   sizeof(int));

    cudaMemcpy(d_row_ptr, h_row_ptr, (n + 1)     * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,     h_col,     total_edges  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, total_edges  * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize: source = REG-0, all others at infinity
    float* h_dist     = (float*)malloc(n * sizeof(float));
    int*   h_pred     = (int*)malloc(n * sizeof(int));
    int*   h_frontier = (int*)calloc(n, sizeof(int));
    int    source     = 0;

    for (int i = 0; i < n; i++) { h_dist[i] = INF_DIST; h_pred[i] = -1; }
    h_dist[source]    = 0.0f;
    h_frontier[source] = 1;

    cudaMemcpy(d_dist,        h_dist,    n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_predecessor, h_pred,    n * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier,    h_frontier, n * sizeof(int),  cudaMemcpyHostToDevice);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaDeviceSynchronize();
    cudaEventRecord(ev_start);

    int blocks     = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int iteration  = 0;
    int h_any_upd;

    printf("\nDijkstra from REG-%d:\n", source);

    do {
        cudaMemset(d_next_frontier, 0, n * sizeof(int));
        cudaMemset(d_any_update,    0, sizeof(int));

        relax_kernel<<<blocks, BLOCK_SIZE>>>(
            d_row_ptr, d_col, d_weights,
            d_dist, d_predecessor,
            d_frontier, d_next_frontier, d_any_update,
            n
        );
        cudaDeviceSynchronize();

        // Swap frontiers
        int* tmp         = d_frontier;
        d_frontier       = d_next_frontier;
        d_next_frontier  = tmp;

        cudaMemcpy(&h_any_upd, d_any_update, sizeof(int), cudaMemcpyDeviceToHost);
        iteration++;

        if (iteration % 5 == 0)
            printf("  Iteration %d\n", iteration);

    } while (h_any_upd && iteration < n);

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    cudaMemcpy(h_dist, d_dist,        n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pred, d_predecessor, n * sizeof(int),   cudaMemcpyDeviceToHost);

    printf("\nConverged after %d iterations\n", iteration);
    printf("Kernel time: %.4f ms\n\n", ms);

    // Coverage: reachable procedures
    int reachable = 0;
    for (int i = num_regs; i < n; i++)
        if (h_dist[i] < INF_DIST) reachable++;
    printf("Procedures reachable from REG-%d: %d / %d (%.1f%%)\n\n",
           source, reachable, num_procs, 100.0f * reachable / num_procs);

    // Top 10 closest procedures = strongest compliance paths
    printf("Top 10 strongest compliance paths from REG-%d:\n", source);
    printf("%-6s %-15s %-12s %-12s\n", "Rank", "Procedure", "Distance", "Avg sim");
    printf("──────────────────────────────────────────────────\n");

    int* used_proc = (int*)calloc(num_procs, sizeof(int));
    int  shown     = 0;
    int  closest_p = -1;
    float closest_d = INF_DIST;

    for (int t = 0; t < 10; t++) {
        float best_d = INF_DIST;
        int   best_i = -1;
        for (int i = 0; i < num_procs; i++) {
            int node = num_regs + i;
            if (!used_proc[i] && h_dist[node] < best_d) {
                best_d = h_dist[node];
                best_i = i;
            }
        }
        if (best_i == -1) break;
        used_proc[best_i] = 1;
        if (shown == 0) { closest_p = best_i; closest_d = best_d; }
        shown++;
        // avg sim over path = 1 - dist/hops (hops hard to know; use 1-dist as proxy)
        printf("%-6d PROC-%-10d %-12.6f %.4f\n",
               t + 1, best_i, best_d, 1.0f - best_d);
    }
    free(used_proc);

    // Path reconstruction for the single closest procedure.
    // Traces back through the host CSR scanning for the edge that
    // satisfies dist[u] + w(u,v) ≈ dist[v] at each hop.
    if (closest_p >= 0 && closest_d < INF_DIST) {
        printf("\nPath to PROC-%d (dist=%.6f):\n", closest_p, closest_d);

        int path[64];
        int path_len = 0;
        int cur      = num_regs + closest_p;

        while (cur != source && path_len < 62) {
            path[path_len++] = cur;
            int   best_pred = -1;
            float best_u_d  = INF_DIST;
            for (int u = 0; u < n; u++) {
                for (int e = h_row_ptr[u]; e < h_row_ptr[u + 1]; e++) {
                    if (h_col[e] == cur) {
                        float expected = h_dist[u] + h_weights[e];
                        if (fabsf(expected - h_dist[cur]) < 1e-4f
                            && h_dist[u] < best_u_d) {
                            best_u_d  = h_dist[u];
                            best_pred = u;
                        }
                    }
                }
            }
            if (best_pred == -1) break;
            cur = best_pred;
        }
        path[path_len++] = cur;

        for (int i = path_len - 1; i >= 0; i--) {
            int  node = path[i];
            char name[32];
            snprintf(name, sizeof(name),
                     node < num_regs ? "REG-%d"  : "PROC-%d",
                     node < num_regs ? node       : node - num_regs);
            if (i > 0) printf("%s → ", name);
            else        printf("%s\n", name);
        }
    }

    // Cleanup
    cudaFree(d_row_ptr);       cudaFree(d_col);
    cudaFree(d_weights);       cudaFree(d_dist);
    cudaFree(d_predecessor);   cudaFree(d_frontier);
    cudaFree(d_next_frontier); cudaFree(d_any_update);

    free(h_row_ptr);  free(h_col);
    free(h_weights);  free(h_dist);
    free(h_pred);     free(h_frontier);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    printf("\nDijkstra OK\n");
    return 0;
}
#endif  // REGMAP_LIBRARY
