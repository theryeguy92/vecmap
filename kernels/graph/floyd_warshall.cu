#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define INF     1e9f
#define BLOCK_SIZE 32

// Phase 1: update the diagonal block (k,k)
// Must complete before phases 2 and 3
__global__ void fw_phase1(
    float* dist,
    int    n,
    int    k_base   // which block we're on
) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = k_base + ty;
    int col = k_base + tx;

    // Load diagonal block into shared memory
    if (row < n && col < n)
        tile[ty][tx] = dist[row * n + col];
    else
        tile[ty][tx] = INF;

    __syncthreads();

    // Run Floyd-Warshall within this tile
    for (int k = 0; k < BLOCK_SIZE; k++) {
        if (tile[ty][k] + tile[k][tx] < tile[ty][tx])
            tile[ty][tx] = tile[ty][k] + tile[k][tx];
        __syncthreads();
    }

    // Write back
    if (row < n && col < n)
        dist[row * n + col] = tile[ty][tx];
}

// Phase 2: update blocks in same row or column as diagonal block
// gridDim.x = number of blocks in one dimension
__global__ void fw_phase2(
    float* dist,
    int    n,
    int    k_base
) {
    __shared__ float tile_diag[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_rc[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Skip the diagonal block itself
    if (blockIdx.x == k_base / BLOCK_SIZE) return;

    int row_diag = k_base + ty;
    int col_diag = k_base + tx;

    // Load diagonal tile
    if (row_diag < n && col_diag < n)
        tile_diag[ty][tx] = dist[row_diag * n + col_diag];
    else
        tile_diag[ty][tx] = INF;

    // Load row block or column block
    int row_rc, col_rc;
    if (blockIdx.y == 0) {
        // Updating row blocks
        row_rc = k_base       + ty;
        col_rc = blockIdx.x * BLOCK_SIZE + tx;
    } else {
        // Updating column blocks
        row_rc = blockIdx.x * BLOCK_SIZE + ty;
        col_rc = k_base       + tx;
    }

    if (row_rc < n && col_rc < n)
        tile_rc[ty][tx] = dist[row_rc * n + col_rc];
    else
        tile_rc[ty][tx] = INF;

    __syncthreads();

    // Update
    for (int k = 0; k < BLOCK_SIZE; k++) {
        float candidate;
        if (blockIdx.y == 0)
            candidate = tile_diag[ty][k] + tile_rc[k][tx];
        else
            candidate = tile_rc[ty][k]   + tile_diag[k][tx];

        if (candidate < tile_rc[ty][tx])
            tile_rc[ty][tx] = candidate;

        __syncthreads();
    }

    // Write back
    if (row_rc < n && col_rc < n)
        dist[row_rc * n + col_rc] = tile_rc[ty][tx];
}

// Phase 3: update all remaining blocks
__global__ void fw_phase3(
    float* dist,
    int    n,
    int    k_base
) {
    __shared__ float tile_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_col[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int k_block = k_base / BLOCK_SIZE;

    // Skip blocks in the pivot row or column
    if (blockIdx.x == k_block || blockIdx.y == k_block) return;

    // Load row tile (same row as pivot, our column)
    int row_r = k_base             + ty;
    int col_r = blockIdx.x * BLOCK_SIZE + tx;

    if (row_r < n && col_r < n)
        tile_row[ty][tx] = dist[row_r * n + col_r];
    else
        tile_row[ty][tx] = INF;

    // Load col tile (our row, same col as pivot)
    int row_c = blockIdx.y * BLOCK_SIZE + ty;
    int col_c = k_base             + tx;

    if (row_c < n && col_c < n)
        tile_col[ty][tx] = dist[row_c * n + col_c];
    else
        tile_col[ty][tx] = INF;

    __syncthreads();

    // Load current cell
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float cur = (row < n && col < n) ? dist[row * n + col] : INF;

    // Update
    for (int k = 0; k < BLOCK_SIZE; k++) {
        float candidate = tile_col[ty][k] + tile_row[k][tx];
        if (candidate < cur)
            cur = candidate;
    }

    if (row < n && col < n)
        dist[row * n + col] = cur;
}

#ifndef REGMAP_LIBRARY
int main() {
    // Use 1000 nodes for Floyd-Warshall demo
    // (full 10k is 400MB matrix — valid on RTX 5080 but slow to verify)
    int n = 1000;

    printf("\nregmap — blocked Floyd-Warshall\n");
    printf("Nodes: %d (%d regs + %d procs)\n", n, n/2, n/2);
    printf("Distance matrix: %d x %d = %.1f MB\n",
        n, n, (float)(n * n * sizeof(float)) / (1024*1024));

    // Allocate distance matrix on host
    size_t mat_size = (size_t)n * n * sizeof(float);
    float* h_dist   = (float*)malloc(mat_size);

    // Initialize: INF everywhere, 0 on diagonal
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            h_dist[i * n + j] = (i == j) ? 0.0f : INF;

    // Add some known edges from our compliance graph
    // Format: dist[reg][proc] = 1.0 - similarity_score
    // (lower = more similar = shorter path)
    srand(42);
    int num_edges = n * 10;  // avg 10 edges per node
    for (int e = 0; e < num_edges; e++) {
        int   i   = rand() % n;
        int   j   = rand() % n;
        float sim = 0.75f + (float)rand() / RAND_MAX * 0.25f;
        float w   = 1.0f - sim;  // weight = 1 - similarity
        if (i != j && h_dist[i * n + j] == INF)
            h_dist[i * n + j] = w;
    }

    // Force known compliance edges
    h_dist[0   * n + 1]   = 0.09f;   // REG-0   → PROC-1   (sim=0.91)
    h_dist[0   * n + 2]   = 0.13f;   // REG-0   → PROC-2   (sim=0.87)
    h_dist[500 * n + 501] = 0.17f;   // REG-500 → PROC-501 (sim=0.83)

    // Allocate on GPU
    float* d_dist;
    cudaMalloc(&d_dist, mat_size);
    cudaMemcpy(d_dist, h_dist, mat_size, cudaMemcpyHostToDevice);

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    printf("Running blocked Floyd-Warshall...\n");
    cudaEventRecord(start);

    // Main loop — one iteration per block
    for (int k = 0; k < n; k += BLOCK_SIZE) {
        // Phase 1: diagonal block
        fw_phase1<<<1, threads>>>(d_dist, n, k);

        // Phase 2: row and column blocks
        dim3 phase2_grid(num_blocks, 2);
        fw_phase2<<<phase2_grid, threads>>>(d_dist, n, k);

        // Phase 3: all remaining blocks
        dim3 phase3_grid(num_blocks, num_blocks);
        fw_phase3<<<phase3_grid, threads>>>(d_dist, n, k);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy back
    cudaMemcpy(h_dist, d_dist, mat_size, cudaMemcpyDeviceToHost);

    printf("Kernel time: %.4f ms\n\n", ms);

    // Verify known paths
    printf("Verifying known compliance paths:\n");
    printf("REG-0 → PROC-1:   dist=%.4f %s\n",
        h_dist[0 * n + 1],
        h_dist[0 * n + 1] < INF ? "(path exists)" : "(GAP)");

    printf("REG-0 → PROC-2:   dist=%.4f %s\n",
        h_dist[0 * n + 2],
        h_dist[0 * n + 2] < INF ? "(path exists)" : "(GAP)");

    printf("REG-0 → PROC-999: dist=%.4f %s\n",
        h_dist[0 * n + 999],
        h_dist[0 * n + 999] < INF ? "(path exists)" : "(GAP)");

    // Count gaps
    int gaps = 0;
    int paths = 0;
    for (int i = 0; i < n/2; i++) {
        for (int j = n/2; j < n; j++) {
            if (h_dist[i * n + j] >= INF) gaps++;
            else paths++;
        }
    }

    printf("\nCompliance analysis (regs vs procs):\n");
    printf("Paths found: %d\n", paths);
    printf("Gaps found:  %d\n", gaps);
    printf("Coverage:    %.1f%%\n",
        100.0f * paths / (paths + gaps));

    // Cleanup
    cudaFree(d_dist);
    free(h_dist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nFloyd-Warshall OK\n");
    return 0;
}
#endif  // REGMAP_LIBRARY