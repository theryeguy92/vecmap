cmake_minimum_required(VERSION 3.18)
project(regmap CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)

# RTX 5080 = Blackwell = sm_120
#set(CMAKE_CUDA_ARCHITECTURES 120)

add_executable(cosine_naive
    kernels/similarity/cosine_naive.cu
)
add_executable(cosine_tiled
    kernels/similarity/cosine_tiled.cu
)
add_executable(cosine_warp
    kernels/similarity/cosine_warp.cu
)
add_executable(threshold_filter
    kernels/graph/threshold_filter.cu
)
add_executable(floyd_warshall
    kernels/graph/floyd_warshall.cu
)
add_executable(bfs
    kernels/graph/bfs.cu
)
add_executable(topological_sort
    kernels/graph/topological_sort.cu
)

add_executable(pagerank
    kernels/graph/pagerank.cu
)
add_executable(kruskal_mst
    kernels/graph/kruskal_mst.cu
)
add_executable(dijkstra
    kernels/graph/dijkstra.cu
)

# ---------------------------------------------------------------------------
# Python extension module: regmap_cuda
# ---------------------------------------------------------------------------

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(regmap_cuda MODULE
    bindings/similarity_bindings.cu
    bindings/graph_bindings.cu
)

set_target_properties(regmap_cuda PROPERTIES
    PREFIX          ""
    SUFFIX          "${PYTHON_EXT_SUFFIX}"
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(regmap_cuda PRIVATE
    "${PYBIND11_INCLUDE_DIR}"
    "${Python3_INCLUDE_DIRS}"
)

target_link_libraries(regmap_cuda PRIVATE
    "${Python3_LIBRARIES}"
)

target_compile_definitions(regmap_cuda PRIVATE REGMAP_LIBRARY=1)
