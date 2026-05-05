/*
 * bindings/graph_bindings.cu
 *
 * pybind11 bridge — all graph algorithm kernels.
 * Registered via init_graph(), called from similarity_bindings.cu.
 *
 * Include order matters: macros (INF, BLOCK_SIZE) are redefined per file.
 * Each kernel is compiled at the point the file is included, using whatever
 * macro values are current — which matches the original standalone behaviour.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

// ---- include kernels in dependency order, clearing conflicting macros ------

#include "../kernels/graph/threshold_filter.cu"   // BLOCK_SIZE=256, prefix_sum, count/fill kernels

#undef BLOCK_SIZE
#undef INF
#include "../kernels/graph/floyd_warshall.cu"      // BLOCK_SIZE=32, INF=1e9f, fw_phase1/2/3

#undef BLOCK_SIZE
#undef INF
#include "../kernels/graph/bfs.cu"                 // BLOCK_SIZE=256, INF=-1, bfs/mark_gaps kernels

#undef BLOCK_SIZE
#include "../kernels/graph/topological_sort.cu"    // BLOCK_SIZE=256, topo kernels

#undef BLOCK_SIZE
#include "../kernels/graph/pagerank.cu"            // BLOCK_SIZE=256, DAMPING/MAX_ITER/CONVERGENCE, pr kernels

#undef BLOCK_SIZE
#undef INF_WEIGHT
#include "../kernels/graph/kruskal_mst.cu"         // BLOCK_SIZE=256, INF_WEIGHT=1e30f, find_cheapest + uf_*

#undef BLOCK_SIZE
#undef INF_DIST
#include "../kernels/graph/dijkstra.cu"            // BLOCK_SIZE=256, INF_DIST=1e9f, relax_kernel

// Block sizes baked into the kernels above; used as literals in wrappers
static const int K_FILTER   = 256;
static const int K_FW       = 32;   // floyd_warshall uses 32x32 tiles
static const int K_BFS      = 256;
static const int K_TOPO     = 256;
static const int K_PR       = 256;
static const int K_MST      = 256;
static const int K_DIJKSTRA = 256;

static const float MST_INF_WT   = 1e30f;
static const float DIJKSTRA_INF = 1e9f;

namespace py = pybind11;
using arr_f = py::array_t<float, py::array::c_style | py::array::forcecast>;
using arr_i = py::array_t<int,   py::array::c_style | py::array::forcecast>;

// ===========================================================================
// threshold_filter
// ===========================================================================

static py::dict threshold_filter_py(arr_f sim_matrix, float threshold) {
    auto buf = sim_matrix.request();
    if (buf.ndim != 2)
        throw std::runtime_error("sim_matrix must be 2-D float32");

    int num_regs  = (int)buf.shape[0];
    int num_procs = (int)buf.shape[1];
    float* h_C    = (float*)buf.ptr;

    float* d_C;
    cudaMalloc(&d_C, (size_t)num_regs * num_procs * sizeof(float));
    cudaMemcpy(d_C, h_C, (size_t)num_regs * num_procs * sizeof(float), cudaMemcpyHostToDevice);

    // Step 1 — count edges per reg
    int* d_edge_counts;
    cudaMalloc(&d_edge_counts, num_regs * sizeof(int));
    cudaMemset(d_edge_counts, 0, num_regs * sizeof(int));

    int blocks = (num_regs + K_FILTER - 1) / K_FILTER;
    count_edges_kernel<<<blocks, K_FILTER>>>(d_C, d_edge_counts, num_regs, num_procs, threshold);
    cudaDeviceSynchronize();

    std::vector<int> h_edge_counts(num_regs);
    std::vector<int> h_row_ptr(num_regs + 1);
    cudaMemcpy(h_edge_counts.data(), d_edge_counts, num_regs * sizeof(int), cudaMemcpyDeviceToHost);
    prefix_sum(h_edge_counts.data(), h_row_ptr.data(), num_regs);
    int total_edges = h_row_ptr[num_regs];
    cudaFree(d_edge_counts);

    // Step 2 — fill CSR
    int*   d_row_ptr, *d_col_indices;
    float* d_values;
    cudaMalloc(&d_row_ptr,  (num_regs + 1) * sizeof(int));
    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (num_regs + 1) * sizeof(int), cudaMemcpyHostToDevice);

    py::array_t<int>   row_ptr_out(num_regs + 1);
    py::array_t<int>   col_indices_out(std::max(total_edges, 1));
    py::array_t<float> values_out(std::max(total_edges, 1));

    if (total_edges > 0) {
        cudaMalloc(&d_col_indices, total_edges * sizeof(int));
        cudaMalloc(&d_values,      total_edges * sizeof(float));
        fill_csr_kernel<<<blocks, K_FILTER>>>(
            d_C, d_row_ptr, d_col_indices, d_values, num_regs, num_procs, threshold);
        cudaDeviceSynchronize();
        cudaMemcpy(col_indices_out.mutable_data(), d_col_indices, total_edges * sizeof(int),   cudaMemcpyDeviceToHost);
        cudaMemcpy(values_out.mutable_data(),      d_values,      total_edges * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_col_indices);
        cudaFree(d_values);
    }

    cudaMemcpy(row_ptr_out.mutable_data(), d_row_ptr, (num_regs + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_C);
    cudaFree(d_row_ptr);

    // Offset col_indices from proc-local [0, num_procs) to global [num_regs, n)
    // so that all graph algorithms receive a consistent n-node CSR.
    {
        int* ci = col_indices_out.mutable_data();
        for (int i = 0; i < total_edges; i++) ci[i] += num_regs;
    }

    // Extend row_ptr to cover all n = num_regs + num_procs nodes.
    // Procedure nodes have no outgoing edges, so their row_ptr entries = total_edges.
    int n_nodes = num_regs + num_procs;
    py::array_t<int> full_row_ptr(n_nodes + 1);
    {
        int* full = full_row_ptr.mutable_data();
        std::memcpy(full, row_ptr_out.data(), (num_regs + 1) * sizeof(int));
        for (int i = num_regs + 1; i <= n_nodes; i++) full[i] = total_edges;
    }

    py::dict result;
    result["row_ptr"]     = full_row_ptr;
    result["col_indices"] = col_indices_out;
    result["values"]      = values_out;
    result["num_regs"]    = num_regs;
    result["num_procs"]   = num_procs;
    result["num_nodes"]   = n_nodes;
    result["num_edges"]   = total_edges;
    return result;
}

// ===========================================================================
// floyd_warshall
// ===========================================================================

static py::array_t<float> floyd_warshall_py(arr_f dist_matrix) {
    auto buf = dist_matrix.request();
    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1])
        throw std::runtime_error("dist_matrix must be square 2-D float32");

    int    n      = (int)buf.shape[0];
    float* h_dist = (float*)buf.ptr;

    float* d_dist;
    cudaMalloc(&d_dist, (size_t)n * n * sizeof(float));
    cudaMemcpy(d_dist, h_dist, (size_t)n * n * sizeof(float), cudaMemcpyHostToDevice);

    // fw_phase1/2/3 kernels were compiled with BLOCK_SIZE=32
    int blocks_n = (n + K_FW - 1) / K_FW;
    for (int k = 0; k < blocks_n; k++) {
        fw_phase1<<<1, dim3(K_FW, K_FW)>>>(d_dist, n, k * K_FW);
        cudaDeviceSynchronize();
        fw_phase2<<<blocks_n, dim3(K_FW, K_FW)>>>(d_dist, n, k * K_FW);
        cudaDeviceSynchronize();
        fw_phase3<<<dim3(blocks_n, blocks_n), dim3(K_FW, K_FW)>>>(d_dist, n, k * K_FW);
        cudaDeviceSynchronize();
    }

    py::array_t<float> result({n, n});
    cudaMemcpy(result.mutable_data(), d_dist, (size_t)n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_dist);
    return result;
}

// ===========================================================================
// bfs
// ===========================================================================

static py::dict bfs_py(arr_i row_ptr, arr_i col_indices,
                       int n, int num_regs, int num_procs, int source) {
    auto rp_buf = row_ptr.request();
    auto ci_buf = col_indices.request();
    int* h_rp = (int*)rp_buf.ptr;
    int* h_ci = (int*)ci_buf.ptr;
    int  E    = (int)ci_buf.shape[0];

    int *d_rp, *d_ci, *d_dist;
    cudaMalloc(&d_rp,   (n + 1) * sizeof(int));
    cudaMalloc(&d_ci,   E       * sizeof(int));
    cudaMalloc(&d_dist, n       * sizeof(int));
    cudaMemcpy(d_rp, h_rp, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci, h_ci, E       * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_dist, -1, n * sizeof(int));   // -1 == INF (unvisited)

    int zero = 0;
    cudaMemcpy(d_dist + source, &zero, sizeof(int), cudaMemcpyHostToDevice);

    // Frontier: compact array of node IDs (like bfs.cu main())
    int *d_frontier, *d_next_frontier, *d_next_size;
    cudaMalloc(&d_frontier,      n * sizeof(int));
    cudaMalloc(&d_next_frontier, n * sizeof(int));
    cudaMalloc(&d_next_size,     sizeof(int));

    cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice);
    int frontier_size = 1;
    int current_dist  = 0;

    while (frontier_size > 0) {
        cudaMemset(d_next_frontier, 0, frontier_size * sizeof(int));
        cudaMemset(d_next_size, 0, sizeof(int));

        int blk = (frontier_size + K_BFS - 1) / K_BFS;
        bfs_frontier_kernel<<<blk, K_BFS>>>(
            d_rp, d_ci, d_dist, d_frontier, d_next_frontier,
            frontier_size, d_next_size, current_dist);
        cudaDeviceSynchronize();

        cudaMemcpy(&frontier_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost);
        if (frontier_size > 0)
            cudaMemcpy(d_frontier, d_next_frontier, frontier_size * sizeof(int), cudaMemcpyDeviceToDevice);
        current_dist++;
    }

    // Mark gap procedures (unreachable from source)
    int* d_gaps;
    cudaMalloc(&d_gaps, num_procs * sizeof(int));
    cudaMemset(d_gaps, 0, num_procs * sizeof(int));
    {
        int blk = (num_procs + K_BFS - 1) / K_BFS;
        mark_gaps_kernel<<<blk, K_BFS>>>(d_dist, d_gaps, num_regs, num_procs, n);
        cudaDeviceSynchronize();
    }

    py::array_t<int> dist_out(n);
    py::array_t<int> gaps_out(num_procs);
    cudaMemcpy(dist_out.mutable_data(), d_dist, n         * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(gaps_out.mutable_data(), d_gaps, num_procs * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_dist);
    cudaFree(d_frontier); cudaFree(d_next_frontier); cudaFree(d_next_size);
    cudaFree(d_gaps);

    // Count gaps
    int* gap_ptr = gaps_out.mutable_data();
    int  n_gaps  = 0;
    for (int i = 0; i < num_procs; i++) n_gaps += gap_ptr[i];

    py::dict result;
    result["distances"] = dist_out;
    result["gaps"]      = gaps_out;
    result["num_gaps"]  = n_gaps;
    return result;
}

// ===========================================================================
// topological_sort
// ===========================================================================

static py::dict topological_sort_py(arr_i row_ptr, arr_i col_indices, int n) {
    auto rp_buf = row_ptr.request();
    auto ci_buf = col_indices.request();
    int* h_rp = (int*)rp_buf.ptr;
    int* h_ci = (int*)ci_buf.ptr;
    int  E    = (int)ci_buf.shape[0];

    int *d_rp, *d_ci, *d_indegree, *d_processed;
    int *d_frontier, *d_frontier_sz, *d_topo_order, *d_topo_sz;

    cudaMalloc(&d_rp,          (n + 1) * sizeof(int));
    cudaMalloc(&d_ci,          E       * sizeof(int));
    cudaMalloc(&d_indegree,    n       * sizeof(int));
    cudaMalloc(&d_processed,   n       * sizeof(int));
    cudaMalloc(&d_frontier,    n       * sizeof(int));
    cudaMalloc(&d_frontier_sz, sizeof(int));
    cudaMalloc(&d_topo_order,  n       * sizeof(int));
    cudaMalloc(&d_topo_sz,     sizeof(int));

    cudaMemcpy(d_rp, h_rp, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci, h_ci, E       * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_indegree,  0, n * sizeof(int));
    cudaMemset(d_processed, 0, n * sizeof(int));
    cudaMemset(d_topo_sz,   0, sizeof(int));

    // Compute in-degrees
    {
        int blk = (E + K_TOPO - 1) / K_TOPO;
        compute_indegree_kernel<<<blk, K_TOPO>>>(d_ci, d_indegree, E);
        cudaDeviceSynchronize();
    }

    int total_sorted = 0;

    while (true) {
        cudaMemset(d_frontier_sz, 0, sizeof(int));

        int blk = (n + K_TOPO - 1) / K_TOPO;
        find_zero_indegree_kernel<<<blk, K_TOPO>>>(
            d_indegree, d_frontier, d_frontier_sz, d_processed, n);
        cudaDeviceSynchronize();

        int fs = 0;
        cudaMemcpy(&fs, d_frontier_sz, sizeof(int), cudaMemcpyDeviceToHost);
        if (fs == 0) break;

        process_frontier_kernel<<<(fs + K_TOPO - 1) / K_TOPO, K_TOPO>>>(
            d_rp, d_ci, d_indegree, d_frontier, d_topo_order, d_topo_sz, fs);
        cudaDeviceSynchronize();

        total_sorted += fs;
    }

    py::array_t<int> order_out(n);
    cudaMemcpy(order_out.mutable_data(), d_topo_order, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_indegree); cudaFree(d_processed);
    cudaFree(d_frontier); cudaFree(d_frontier_sz); cudaFree(d_topo_order); cudaFree(d_topo_sz);

    py::dict result;
    result["order"]     = order_out;
    result["has_cycle"] = (total_sorted < n);
    return result;
}

// ===========================================================================
// pagerank — uses transposed CSR for efficient incoming-edge iteration
// ===========================================================================

static py::array_t<float> pagerank_py(
    arr_i row_ptr, arr_i col_indices, int n,
    float damping = 0.85f, int max_iter = 100
) {
    auto rp_buf = row_ptr.request();
    auto ci_buf = col_indices.request();
    int* h_rp = (int*)rp_buf.ptr;
    int* h_ci = (int*)ci_buf.ptr;
    int  E    = (int)ci_buf.shape[0];

    // Build transposed CSR on CPU
    std::vector<int> out_degree(n, 0);
    for (int u = 0; u < n; u++)
        out_degree[u] = h_rp[u + 1] - h_rp[u];

    // Count in-degrees for transposed row_ptr
    std::vector<int> in_degree(n, 0);
    for (int e = 0; e < E; e++)
        in_degree[h_ci[e]]++;

    std::vector<int> t_row_ptr(n + 1, 0);
    for (int i = 0; i < n; i++)
        t_row_ptr[i + 1] = t_row_ptr[i] + in_degree[i];

    std::vector<int> t_col_indices(E);
    std::vector<int> fill_pos(t_row_ptr.begin(), t_row_ptr.end());
    for (int u = 0; u < n; u++) {
        for (int e = h_rp[u]; e < h_rp[u + 1]; e++) {
            int v = h_ci[e];
            t_col_indices[fill_pos[v]++] = u;
        }
    }

    // Upload to GPU
    int *d_rp, *d_ci, *d_t_rp, *d_t_ci, *d_out_deg;
    float *d_rank, *d_new_rank, *d_max_diff;

    cudaMalloc(&d_rp,       (n + 1) * sizeof(int));
    cudaMalloc(&d_ci,       E       * sizeof(int));
    cudaMalloc(&d_t_rp,     (n + 1) * sizeof(int));
    cudaMalloc(&d_t_ci,     E       * sizeof(int));
    cudaMalloc(&d_out_deg,  n       * sizeof(int));
    cudaMalloc(&d_rank,     n       * sizeof(float));
    cudaMalloc(&d_new_rank, n       * sizeof(float));
    cudaMalloc(&d_max_diff, sizeof(float));

    cudaMemcpy(d_rp,     h_rp,              (n + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci,     h_ci,              E       * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_rp,   t_row_ptr.data(),  (n + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_ci,   t_col_indices.data(), E    * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_deg, out_degree.data(), n      * sizeof(int),   cudaMemcpyHostToDevice);

    // Initialize ranks to 1/n
    float init_rank = 1.0f / n;
    std::vector<float> h_rank(n, init_rank);
    cudaMemcpy(d_rank, h_rank.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blk = (n + K_PR - 1) / K_PR;
    float convergence_thr = 1e-6f;

    for (int iter = 0; iter < max_iter; iter++) {
        pagerank_iter_transposed_kernel<<<blk, K_PR>>>(
            d_t_rp, d_t_ci, d_rank, d_new_rank, d_out_deg, n, damping);
        cudaDeviceSynchronize();

        // Check convergence
        float zero_f = 0.0f;
        cudaMemcpy(d_max_diff, &zero_f, sizeof(float), cudaMemcpyHostToDevice);
        convergence_kernel<<<blk, K_PR>>>(d_rank, d_new_rank, d_max_diff, n);
        cudaDeviceSynchronize();

        float max_diff = 0.0f;
        cudaMemcpy(&max_diff, d_max_diff, sizeof(float), cudaMemcpyDeviceToHost);

        // Swap rank and new_rank
        std::swap(d_rank, d_new_rank);

        if (max_diff < convergence_thr) break;
    }

    py::array_t<float> result(n);
    cudaMemcpy(result.mutable_data(), d_rank, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_t_rp); cudaFree(d_t_ci);
    cudaFree(d_out_deg); cudaFree(d_rank); cudaFree(d_new_rank); cudaFree(d_max_diff);

    return result;
}

// ===========================================================================
// kruskal_mst (Borůvka) — returns MST edge list
// ===========================================================================

static py::dict kruskal_mst_py(arr_i row_ptr, arr_i col_indices, arr_f values, int n) {
    auto rp_buf = row_ptr.request();
    auto ci_buf = col_indices.request();
    auto vl_buf = values.request();
    int* h_rp  = (int*)rp_buf.ptr;
    int* h_ci  = (int*)ci_buf.ptr;
    float* h_w = (float*)vl_buf.ptr;
    int E      = (int)ci_buf.shape[0];

    int   *d_rp, *d_ci, *d_component, *d_cheapest_dst;
    float *d_w, *d_cheapest_wt;

    cudaMalloc(&d_rp,          (n + 1) * sizeof(int));
    cudaMalloc(&d_ci,          E       * sizeof(int));
    cudaMalloc(&d_w,           E       * sizeof(float));
    cudaMalloc(&d_component,   n       * sizeof(int));
    cudaMalloc(&d_cheapest_dst,n       * sizeof(int));
    cudaMalloc(&d_cheapest_wt, n       * sizeof(float));

    cudaMemcpy(d_rp, h_rp, (n + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci, h_ci, E       * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_w,  h_w,  E       * sizeof(float), cudaMemcpyHostToDevice);

    // CPU union-find
    std::vector<int> parent(n), uf_rank(n, 0);
    for (int i = 0; i < n; i++) parent[i] = i;

    // Upload initial component (each node its own component)
    cudaMemcpy(d_component, parent.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int>   mst_src, mst_dst;
    std::vector<float> mst_wt;
    float total_weight = 0.0f;

    int blk = (n + K_MST - 1) / K_MST;

    for (int round = 0; round < 30; round++) {
        // Reset cheapest arrays
        std::vector<int>   h_cdst(n, -1);
        std::vector<float> h_cwt(n, MST_INF_WT);
        cudaMemcpy(d_cheapest_dst, h_cdst.data(), n * sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_cheapest_wt,  h_cwt.data(),  n * sizeof(float), cudaMemcpyHostToDevice);

        find_cheapest_kernel<<<blk, K_MST>>>(
            d_rp, d_ci, d_w, d_component, d_cheapest_dst, d_cheapest_wt, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_cdst.data(), d_cheapest_dst, n * sizeof(int),   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cwt.data(),  d_cheapest_wt,  n * sizeof(float), cudaMemcpyDeviceToHost);

        bool any_merge = false;
        for (int u = 0; u < n; u++) {
            int v = h_cdst[u];
            if (v < 0 || h_cwt[u] >= MST_INF_WT) continue;
            int ru = uf_find(parent.data(), u);
            int rv = uf_find(parent.data(), v);
            if (ru == rv) continue;
            uf_union(parent.data(), uf_rank.data(), u, v);
            mst_src.push_back(u);
            mst_dst.push_back(v);
            mst_wt.push_back(h_cwt[u]);
            total_weight += h_cwt[u];
            any_merge = true;
        }
        if (!any_merge) break;

        // Update component labels (flatten union-find)
        for (int i = 0; i < n; i++) parent[i] = uf_find(parent.data(), i);
        cudaMemcpy(d_component, parent.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_w);
    cudaFree(d_component); cudaFree(d_cheapest_dst); cudaFree(d_cheapest_wt);

    int mst_n = (int)mst_src.size();
    py::array_t<int>   src_out(mst_n), dst_out(mst_n);
    py::array_t<float> wt_out(mst_n);
    if (mst_n > 0) {
        std::memcpy(src_out.mutable_data(), mst_src.data(), mst_n * sizeof(int));
        std::memcpy(dst_out.mutable_data(), mst_dst.data(), mst_n * sizeof(int));
        std::memcpy(wt_out.mutable_data(),  mst_wt.data(),  mst_n * sizeof(float));
    }

    py::dict result;
    result["src"]          = src_out;
    result["dst"]          = dst_out;
    result["weights"]      = wt_out;
    result["total_weight"] = total_weight;
    result["num_edges"]    = mst_n;
    return result;
}

// ===========================================================================
// dijkstra — parallel label-correcting SSSP
// ===========================================================================

static py::dict dijkstra_py(arr_i row_ptr, arr_i col_indices, arr_f values,
                            int n, int source) {
    auto rp_buf = row_ptr.request();
    auto ci_buf = col_indices.request();
    auto vl_buf = values.request();
    int* h_rp  = (int*)rp_buf.ptr;
    int* h_ci  = (int*)ci_buf.ptr;
    float* h_w = (float*)vl_buf.ptr;
    int E      = (int)ci_buf.shape[0];

    int   *d_rp, *d_ci, *d_predecessor, *d_in_frontier, *d_next_frontier, *d_any_update;
    float *d_w, *d_dist;

    cudaMalloc(&d_rp,           (n + 1) * sizeof(int));
    cudaMalloc(&d_ci,           E       * sizeof(int));
    cudaMalloc(&d_w,            E       * sizeof(float));
    cudaMalloc(&d_dist,         n       * sizeof(float));
    cudaMalloc(&d_predecessor,  n       * sizeof(int));
    cudaMalloc(&d_in_frontier,  n       * sizeof(int));
    cudaMalloc(&d_next_frontier,n       * sizeof(int));
    cudaMalloc(&d_any_update,   sizeof(int));

    cudaMemcpy(d_rp, h_rp, (n + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci, h_ci, E       * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_w,  h_w,  E       * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> h_dist(n, DIJKSTRA_INF);
    h_dist[source] = 0.0f;
    cudaMemcpy(d_dist, h_dist.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<int> h_pred(n, -1), h_frontier(n, 0);
    h_frontier[source] = 1;
    cudaMemcpy(d_predecessor,  h_pred.data(),    n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_frontier,  h_frontier.data(),n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_next_frontier, 0, n * sizeof(int));

    int blk = (n + K_DIJKSTRA - 1) / K_DIJKSTRA;

    for (int iter = 0; iter < n; iter++) {
        cudaMemset(d_next_frontier, 0, n * sizeof(int));
        cudaMemset(d_any_update,    0, sizeof(int));

        relax_kernel<<<blk, K_DIJKSTRA>>>(
            d_rp, d_ci, d_w, d_dist, d_predecessor,
            d_in_frontier, d_next_frontier, d_any_update, n);
        cudaDeviceSynchronize();

        int any = 0;
        cudaMemcpy(&any, d_any_update, sizeof(int), cudaMemcpyDeviceToHost);
        if (!any) break;

        // Swap frontier and next_frontier (copy next → current)
        cudaMemcpy(d_in_frontier, d_next_frontier, n * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    py::array_t<float> dist_out(n);
    py::array_t<int>   pred_out(n);
    cudaMemcpy(dist_out.mutable_data(), d_dist,        n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pred_out.mutable_data(), d_predecessor, n * sizeof(int),   cudaMemcpyDeviceToHost);

    cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_w); cudaFree(d_dist);
    cudaFree(d_predecessor); cudaFree(d_in_frontier); cudaFree(d_next_frontier);
    cudaFree(d_any_update);

    py::dict result;
    result["distances"]    = dist_out;
    result["predecessors"] = pred_out;
    return result;
}

// ===========================================================================
// Registration
// ===========================================================================

void init_graph(py::module_& m) {
    m.def("threshold_filter", &threshold_filter_py,
        py::arg("sim_matrix"), py::arg("threshold") = 0.75f,
        R"(Convert a similarity matrix to a sparse CSR graph.

Args:
    sim_matrix: float32 [num_regs, num_procs] — cosine similarity output
    threshold:  float — minimum similarity to create an edge (default 0.75)

Returns:
    dict with keys: row_ptr [n+1], col_indices [E], values [E],
    num_regs, num_procs, num_nodes (=num_regs+num_procs), num_edges.
    col_indices are global node IDs in [num_regs, num_nodes).
    row_ptr covers all n nodes; proc rows have 0 out-edges.)");

    m.def("floyd_warshall", &floyd_warshall_py,
        py::arg("dist_matrix"),
        R"(All-pairs shortest paths on a dense distance matrix.

Args:
    dist_matrix: float32 [N, N] — edge weights (use 1e9 for no edge)

Returns:
    float32 [N, N] — shortest-path distances)");

    m.def("bfs", &bfs_py,
        py::arg("row_ptr"), py::arg("col_indices"),
        py::arg("n"), py::arg("num_regs"), py::arg("num_procs"),
        py::arg("source") = 0,
        R"(BFS gap detection from a source regulation node.

Args:
    row_ptr, col_indices: CSR graph arrays (int32)
    n:         total nodes (num_regs + num_procs)
    num_regs:  number of regulation nodes (indices 0..num_regs-1)
    num_procs: number of procedure nodes (indices num_regs..n-1)
    source:    source regulation index (default 0)

Returns:
    dict: distances [n] int32, gaps [num_procs] int32, num_gaps int)");

    m.def("topological_sort", &topological_sort_py,
        py::arg("row_ptr"), py::arg("col_indices"), py::arg("n"),
        R"(Parallel topological sort for regulatory hierarchy.

Args:
    row_ptr, col_indices: CSR graph arrays (int32)
    n: total nodes

Returns:
    dict: order [n] int32 (topological order), has_cycle bool)");

    m.def("pagerank", &pagerank_py,
        py::arg("row_ptr"), py::arg("col_indices"), py::arg("n"),
        py::arg("damping") = 0.85f, py::arg("max_iter") = 100,
        R"(PageRank — identify the most critical regulation nodes.

Args:
    row_ptr, col_indices: CSR graph arrays (int32)
    n:        total nodes
    damping:  damping factor (default 0.85)
    max_iter: iteration cap (default 100)

Returns:
    float32 [n] — PageRank score per node)");

    m.def("kruskal_mst", &kruskal_mst_py,
        py::arg("row_ptr"), py::arg("col_indices"), py::arg("values"), py::arg("n"),
        R"(Borůvka MST — minimum set of procedures covering all regulations.

Args:
    row_ptr, col_indices: CSR graph arrays (int32)
    values: edge weights float32 (use 1-similarity so lower = stronger)
    n: total nodes

Returns:
    dict: src/dst/weights arrays, total_weight float, num_edges int)");

    m.def("dijkstra", &dijkstra_py,
        py::arg("row_ptr"), py::arg("col_indices"), py::arg("values"),
        py::arg("n"), py::arg("source") = 0,
        R"(Parallel Dijkstra SSSP — strongest compliance path from a source.

Args:
    row_ptr, col_indices: CSR graph arrays (int32)
    values: edge weights float32 (use 1-similarity so lower = stronger)
    n:      total nodes
    source: source node index (default 0 = first regulation)

Returns:
    dict: distances [n] float32, predecessors [n] int32)");
}
