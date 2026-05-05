#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE   256
#define INF_WEIGHT   1e30f

// Boruvka round: each node finds its cheapest outgoing edge
// that crosses a component boundary
__global__ void find_cheapest_kernel(
    int*   row_ptr,
    int*   col_indices,
    float* weights,
    int*   component,
    int*   cheapest_dst,   // best neighbor in a different component
    float* cheapest_wt,    // weight of that edge
    int    n
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;

    int   best_dst = -1;
    float best_wt  = INF_WEIGHT;

    for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
        int   v = col_indices[e];
        float w = weights[e];
        if (component[u] != component[v] && w < best_wt) {
            best_wt  = w;
            best_dst = v;
        }
    }

    cheapest_dst[u] = best_dst;
    cheapest_wt[u]  = best_wt;
}

// Union-Find with path compression and union by rank
int uf_find(int* parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

void uf_union(int* parent, int* uf_rank, int x, int y) {
    int rx = uf_find(parent, x);
    int ry = uf_find(parent, y);
    if (rx == ry) return;
    if (uf_rank[rx] < uf_rank[ry]) { int t = rx; rx = ry; ry = t; }
    parent[ry] = rx;
    if (uf_rank[rx] == uf_rank[ry]) uf_rank[rx]++;
}

#ifndef REGMAP_LIBRARY
int main() {
    int num_regs  = 500;
    int num_procs = 500;
    int n         = num_regs + num_procs;

    printf("\nregmap — GPU Boruvka MST (Minimum Coverage Set)\n");
    printf("Nodes: %d (%d regs + %d procs)\n", n, num_regs, num_procs);

    // Build undirected weighted compliance graph.
    // Edge weight = 1.0 - similarity (lower weight = stronger link).
    // Undirected: store both (u,v) and (v,u) so MST can propagate
    // from regulations into procedures.
    int  max_dir_edges = n * 15;
    int  max_edges     = max_dir_edges * 2;

    int*   h_src  = (int*)malloc(max_edges * sizeof(int));
    int*   h_dst  = (int*)malloc(max_edges * sizeof(int));
    float* h_sim  = (float*)malloc(max_edges * sizeof(float));
    int    ne     = 0;

    srand(42);

    // Regulations → procedures
    for (int i = 0; i < num_regs; i++) {
        int degree = 5 + rand() % 10;
        for (int e = 0; e < degree; e++) {
            int   j   = num_regs + rand() % num_procs;
            float sim = 0.5f + 0.5f * ((float)rand() / RAND_MAX);
            h_src[ne] = i;  h_dst[ne] = j;  h_sim[ne] = sim;  ne++;
        }
    }

    // Reg → reg hierarchy
    for (int i = 1; i < num_regs; i += 3) {
        int   j   = rand() % i;
        float sim = 0.6f + 0.3f * ((float)rand() / RAND_MAX);
        h_src[ne] = i;  h_dst[ne] = j;  h_sim[ne] = sim;  ne++;
    }

    // Add reverse edges to make undirected
    int dir_edges = ne;
    for (int e = 0; e < dir_edges; e++) {
        h_src[ne] = h_dst[e];
        h_dst[ne] = h_src[e];
        h_sim[ne] = h_sim[e];
        ne++;
    }

    printf("Edges (undirected): %d\n", ne);

    // Build CSR from edge list
    int*   h_row_ptr     = (int*)calloc(n + 1, sizeof(int));
    int*   h_col_indices = (int*)malloc(ne * sizeof(int));
    float* h_weights     = (float*)malloc(ne * sizeof(float));

    for (int e = 0; e < ne; e++)
        h_row_ptr[h_src[e] + 1]++;
    for (int i = 1; i <= n; i++)
        h_row_ptr[i] += h_row_ptr[i - 1];

    int* write_ptr = (int*)calloc(n, sizeof(int));
    for (int e = 0; e < ne; e++) {
        int u         = h_src[e];
        int pos       = h_row_ptr[u] + write_ptr[u]++;
        h_col_indices[pos] = h_dst[e];
        h_weights[pos]     = 1.0f - h_sim[e];
    }
    free(write_ptr);
    free(h_src);  free(h_dst);  free(h_sim);

    // GPU allocations
    int*   d_row_ptr;
    int*   d_col_indices;
    float* d_weights;
    int*   d_component;
    int*   d_cheapest_dst;
    float* d_cheapest_wt;

    cudaMalloc(&d_row_ptr,      (n + 1) * sizeof(int));
    cudaMalloc(&d_col_indices,  ne      * sizeof(int));
    cudaMalloc(&d_weights,      ne      * sizeof(float));
    cudaMalloc(&d_component,    n       * sizeof(int));
    cudaMalloc(&d_cheapest_dst, n       * sizeof(int));
    cudaMalloc(&d_cheapest_wt,  n       * sizeof(float));

    cudaMemcpy(d_row_ptr,     h_row_ptr,     (n + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, h_col_indices, ne      * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights,     h_weights,     ne      * sizeof(float), cudaMemcpyHostToDevice);

    // Each node starts as its own component
    int* h_component = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) h_component[i] = i;
    cudaMemcpy(d_component, h_component, n * sizeof(int), cudaMemcpyHostToDevice);

    // CPU union-find for merging components between rounds
    int* uf_parent = (int*)malloc(n * sizeof(int));
    int* uf_rank   = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) uf_parent[i] = i;

    // MST edge storage — at most n-1 edges
    int*   mst_u   = (int*)malloc(n * sizeof(int));
    int*   mst_v   = (int*)malloc(n * sizeof(int));
    float* mst_wt  = (float*)malloc(n * sizeof(float));
    int    mst_sz  = 0;

    int*   h_cheapest_dst = (int*)malloc(n * sizeof(int));
    float* h_cheapest_wt  = (float*)malloc(n * sizeof(float));

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaDeviceSynchronize();
    cudaEventRecord(ev_start);

    int blocks          = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int round           = 0;
    int merges_in_round = 1;

    printf("\nRunning Boruvka MST...\n");

    while (merges_in_round > 0 && mst_sz < n - 1) {

        // GPU: each node finds its cheapest inter-component edge
        find_cheapest_kernel<<<blocks, BLOCK_SIZE>>>(
            d_row_ptr, d_col_indices, d_weights,
            d_component,
            d_cheapest_dst, d_cheapest_wt,
            n
        );
        cudaDeviceSynchronize();

        cudaMemcpy(h_cheapest_dst, d_cheapest_dst, n * sizeof(int),   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cheapest_wt,  d_cheapest_wt,  n * sizeof(float), cudaMemcpyDeviceToHost);

        // CPU: for each component, find the node with the globally
        // cheapest outgoing edge
        int*   comp_best_src = (int*)malloc(n * sizeof(int));
        int*   comp_best_dst = (int*)malloc(n * sizeof(int));
        float* comp_best_wt  = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            comp_best_src[i] = -1;
            comp_best_dst[i] = -1;
            comp_best_wt[i]  = INF_WEIGHT;
        }

        for (int u = 0; u < n; u++) {
            if (h_cheapest_dst[u] == -1) continue;
            int root = uf_find(uf_parent, u);
            if (h_cheapest_wt[u] < comp_best_wt[root]) {
                comp_best_wt[root]  = h_cheapest_wt[u];
                comp_best_dst[root] = h_cheapest_dst[u];
                comp_best_src[root] = u;
            }
        }

        // Merge components; collect MST edges
        merges_in_round = 0;
        for (int r = 0; r < n; r++) {
            if (comp_best_dst[r] == -1) continue;
            int u  = comp_best_src[r];
            int v  = comp_best_dst[r];
            int ru = uf_find(uf_parent, u);
            int rv = uf_find(uf_parent, v);
            if (ru != rv) {
                mst_u[mst_sz]  = u;
                mst_v[mst_sz]  = v;
                mst_wt[mst_sz] = comp_best_wt[r];
                mst_sz++;
                uf_union(uf_parent, uf_rank, ru, rv);
                merges_in_round++;
            }
        }

        // Push updated component IDs back to GPU
        for (int i = 0; i < n; i++)
            h_component[i] = uf_find(uf_parent, i);
        cudaMemcpy(d_component, h_component, n * sizeof(int), cudaMemcpyHostToDevice);

        free(comp_best_src);
        free(comp_best_dst);
        free(comp_best_wt);

        round++;
        printf("  Round %d: %d merges, MST edges so far: %d\n",
               round, merges_in_round, mst_sz);
    }

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    printf("\nMST complete: %d edges in %d rounds\n", mst_sz, round);
    printf("Kernel time: %.4f ms\n\n", ms);

    // Total MST weight and average similarity
    float total_wt  = 0.0f;
    for (int e = 0; e < mst_sz; e++) total_wt += mst_wt[e];
    float avg_sim = (mst_sz > 0) ? 1.0f - (total_wt / mst_sz) : 0.0f;
    printf("Total MST weight:   %.4f\n", total_wt);
    printf("Avg edge similarity: %.4f\n\n", avg_sim);

    // Coverage: count distinct procedures appearing in any MST edge
    int* proc_in_mst = (int*)calloc(num_procs, sizeof(int));
    for (int e = 0; e < mst_sz; e++) {
        if (mst_u[e] >= num_regs) proc_in_mst[mst_u[e] - num_regs] = 1;
        if (mst_v[e] >= num_regs) proc_in_mst[mst_v[e] - num_regs] = 1;
    }
    int covered = 0;
    for (int i = 0; i < num_procs; i++) covered += proc_in_mst[i];
    printf("Procedures in MST: %d / %d (%.1f%%)\n\n",
           covered, num_procs, 100.0f * covered / num_procs);

    // Top 10 strongest (lowest weight) MST edges
    printf("Top 10 strongest MST compliance links:\n");
    printf("%-20s %-20s %s\n", "From", "To", "Similarity");
    printf("──────────────────────────────────────────────────\n");

    int* shown = (int*)calloc(mst_sz, sizeof(int));
    for (int t = 0; t < 10 && t < mst_sz; t++) {
        float best_wt = INF_WEIGHT;
        int   best_e  = -1;
        for (int e = 0; e < mst_sz; e++) {
            if (!shown[e] && mst_wt[e] < best_wt) {
                best_wt = mst_wt[e];
                best_e  = e;
            }
        }
        if (best_e == -1) break;
        shown[best_e] = 1;
        int u = mst_u[best_e], v = mst_v[best_e];
        char name_u[32], name_v[32];
        snprintf(name_u, sizeof(name_u),
                 u < num_regs ? "REG-%d"  : "PROC-%d",
                 u < num_regs ? u         : u - num_regs);
        snprintf(name_v, sizeof(name_v),
                 v < num_regs ? "REG-%d"  : "PROC-%d",
                 v < num_regs ? v         : v - num_regs);
        printf("%-20s %-20s %.4f\n", name_u, name_v, 1.0f - best_wt);
    }

    // Cleanup
    cudaFree(d_row_ptr);      cudaFree(d_col_indices);
    cudaFree(d_weights);      cudaFree(d_component);
    cudaFree(d_cheapest_dst); cudaFree(d_cheapest_wt);

    free(h_row_ptr);      free(h_col_indices);
    free(h_weights);      free(h_component);
    free(uf_parent);      free(uf_rank);
    free(mst_u);          free(mst_v);
    free(mst_wt);         free(shown);
    free(h_cheapest_dst); free(h_cheapest_wt);
    free(proc_in_mst);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    printf("\nBoruvka MST OK\n");
    return 0;
}
#endif  // REGMAP_LIBRARY
