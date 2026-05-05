/*
 * bindings/similarity_bindings.cu
 *
 * pybind11 bridge — cosine similarity kernel + module entry point.
 * graph kernels are registered by init_graph() in graph_bindings.cu.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "../kernels/similarity/cosine_warp.cu"

namespace py = pybind11;

// Defined in graph_bindings.cu — called from PYBIND11_MODULE below
void init_graph(py::module_& m);

// ---------------------------------------------------------------------------
// cosine_similarity(A [M,D], B [N,D]) -> C [M,N]
// ---------------------------------------------------------------------------

static py::array_t<float> cosine_similarity(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B
) {
    auto buf_A = A.request();
    auto buf_B = B.request();

    if (buf_A.ndim != 2 || buf_B.ndim != 2)
        throw std::runtime_error("A and B must be 2-D float32 arrays");
    if (buf_A.shape[1] != buf_B.shape[1])
        throw std::runtime_error("Embedding dim mismatch: A.shape[1] != B.shape[1]");

    int M = (int)buf_A.shape[0];
    int N = (int)buf_B.shape[0];
    int D = (int)buf_A.shape[1];

    float* h_A = (float*)buf_A.ptr;
    float* h_B = (float*)buf_B.ptr;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, (size_t)M * D * sizeof(float));
    cudaMalloc(&d_B, (size_t)N * D * sizeof(float));
    cudaMalloc(&d_C, (size_t)M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, (size_t)M * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice);

    // One warp per (reg, proc) pair; BLOCK_SIZE=256 threads = 8 warps/block
    long long total_warps = (long long)M * N;
    int threads = BLOCK_SIZE;   // 256 from cosine_warp.cu
    int blocks  = (int)((total_warps * WARP_SIZE + threads - 1) / threads);

    cosine_warp_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, D);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        throw std::runtime_error(std::string("cosine_warp_kernel: ") + cudaGetErrorString(err));
    }

    py::array_t<float> result({M, N});
    cudaMemcpy(result.mutable_data(), d_C, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(regmap_cuda, m) {
    m.doc() = "regmap CUDA kernels: cosine similarity + graph algorithms";

    m.def("cosine_similarity", &cosine_similarity,
        py::arg("A"), py::arg("B"),
        R"(Compute cosine similarity between two embedding matrices on GPU.

Args:
    A: float32 ndarray [M, embed_dim]  (regulation embeddings)
    B: float32 ndarray [N, embed_dim]  (procedure embeddings)

Returns:
    float32 ndarray [M, N] — pairwise cosine similarity scores)");

    init_graph(m);
}
