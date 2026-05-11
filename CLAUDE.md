# regmap — CLAUDE.md

This file gives Claude Code persistent context about this project.
Read this before making any changes to any file.

---

## What This Project Is

regmap is a GPU-accelerated command line tool that solves a real compliance problem in the US government contractor space.

Organizations that work under DOE (Department of Energy) oversight must map their internal procedures against federal regulations:

- DOE Orders (e.g. DOE-O-414.1D — Quality Assurance)
- Code of Federal Regulations (e.g. 10 CFR 830)
- Internal Procedures (SOPs, Work Instructions)

This mapping is currently done manually — it takes weeks and costs significant consulting dollars. regmap automates it using GPU-accelerated graph algorithms and NLP embeddings.

---

## What It Produces

### Compliance Requirements Matrix

Shows which regulations are covered, partially covered, or gaps against internal procedures.

```
DOE-O-414.1D  ->  SOP-QA-001   COVERED   (sim=0.91)
DOE-O-414.1D  ->  SOP-QA-002   PARTIAL   (sim=0.76)
10-CFR-830    ->  SOP-ENV-003  GAP       (no path)
```

### Knowledge Graph

Nodes = regulations and procedures
Edges = similarity scores above threshold
Algorithms reveal hierarchy, gaps, critical nodes, and coverage

---

## Commercial Context

- Target buyers: DOE contractors, nuclear facilities, defense organizations
- Current tools: manual spreadsheets, expensive GRC software
- regmap advantage: GPU speed + graph intelligence + terminal workflow
- Comparable products: Palantir compliance tools, ServiceNow GRC

---

## Tech Stack

```
Layer           Technology              Purpose
-----------------------------------------------------
GPU Kernels     CUDA C++ (sm_120)       core computation
Pipeline        Python 3.12             doc parsing + embeddings
Embeddings      HuggingFace             text to float vectors
Bridge          pybind11                CUDA to Python
Output          Graphviz / JSON         knowledge graph export
CLI             Python / argparse       user interface
```

---

## Hardware and Environment

```
GPU:            NVIDIA RTX 5080 (Blackwell)
Architecture:   sm_120 (compute capability 12.0)
CUDA:           12.8
OS:             WSL2 Ubuntu 24.04 on Windows 11
Driver:         Windows NVIDIA driver only
```

### CRITICAL WSL2 Rule

NEVER install Linux NVIDIA GPU drivers inside WSL2.
The Windows driver is stubbed as libcuda.so inside WSL2.
Installing a Linux driver will break CUDA entirely.
Only ever install cuda-toolkit-12-x inside WSL2.

---

## Project Structure

```
regmap/
|
|-- CLAUDE.md
|-- CMakeLists.txt
|-- requirements.txt
|
|-- kernels/
|   |-- similarity/
|   |   |-- cosine_naive.cu         baseline, one thread per pair
|   |   |-- cosine_tiled.cu         shared memory optimization
|   |   |-- cosine_warp.cu          warp shuffle, fastest, use this
|   |
|   |-- graph/
|       |-- threshold_filter.cu     similarity matrix to sparse CSR graph
|       |-- floyd_warshall.cu       all-pairs shortest paths
|       |-- bfs.cu                  gap detection from source node
|       |-- topological_sort.cu     regulatory hierarchy + cycle detection
|       |-- pagerank.cu             critical regulation ranking
|       |-- kruskal_mst.cu          minimum coverage set
|       |-- dijkstra.cu             strongest compliance path
|
|-- pipeline/
|   |-- parser.py                   PDF/docx ingestion
|   |-- embedder.py                 HuggingFace embeddings
|   |-- scraper.py                  PDF scraping utility
|
|-- bindings/
|   |-- similarity_bindings.cu      exposes cosine kernels to Python
|   |-- graph_bindings.cu           exposes graph kernels to Python
|
|-- graph/
|   |-- builder.py                  constructs EmbeddingSet from .npy + _meta.json
|
|-- output/
|   |-- matrix.py                   compliance matrix renderer
|   |-- knowledge_graph.py          Graphviz DOT export
|   |-- gap_report.py               BFS gap analysis report
|   |-- rankings.py                 PageRank output formatter
|
|-- cli/
|   |-- main.py                     entry point
|   |-- commands/
|       |-- analyze.py              full pipeline run
|       |-- report.py               generate output reports
|       |-- benchmark.py            benchmarking mode
|
|-- benchmarks/
|   |-- benchmark_chart.py          matplotlib kernel timing chart
|   |-- benchmark_chart.png         generated chart (git-ignored)
|
|-- examples/
|   |-- generate_examples.py        generates sample_output.txt + .dot
|   |-- sample_output.txt           sample analysis run (git-ignored)
|   |-- sample_knowledge_graph.dot  sample Graphviz export (git-ignored)
|
|-- tests/
|   |-- sample_docs/                parsed DOE order PDFs (.npy + _meta.json)
|
|-- build/
    |-- regmap_cuda...so            compiled pybind11 module (~1.4 MB)
```

---

## Build System

```bash
# Full rebuild (single .so target)
cd build
cmake ..
make
```

### CMake Architecture Flag

```cmake
set(CMAKE_CUDA_ARCHITECTURES 120)
```

Never change this to native or auto on this machine.
Never use sm_89 or lower — the RTX 5080 requires sm_120.

---

## CUDA Rules for This Project

### Memory Hierarchy — always prefer in this order

```
Registers        fastest, private per thread
Shared Memory    fast, shared within block via __shared__
L2 Cache         automatic
Global Memory    slowest, avoid repeated reads
```

### Kernel Conventions

```cpp
// Thread identity — always use this pattern
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= n) return;

// Shared memory pattern
__shared__ float tile[TILE_SIZE][TILE_SIZE];
__syncthreads();

// Warp reduction pattern
val += __shfl_down_sync(0xffffffff, val, 16);
val += __shfl_down_sync(0xffffffff, val, 8);
val += __shfl_down_sync(0xffffffff, val, 4);
val += __shfl_down_sync(0xffffffff, val, 2);
val += __shfl_down_sync(0xffffffff, val, 1);
```

### Common Pitfalls to Avoid

- Never use std::to_string in .cu files — use snprintf instead
- Never use std::string in device code
- Always call cudaDeviceSynchronize() before timing measurements
- Always free GPU memory with cudaFree() not free()
- Never mix h_ (host) and d_ (device) pointers

### Naming Conventions

```
h_*          host CPU memory          e.g. h_distances
d_*          device GPU memory        e.g. d_distances
*_kernel     GPU kernel function
*_cu         CUDA source file
```

---

## Graph Data Model

```
Node types:
  0 to num_regs-1              regulations (CFR, DOE Orders)
  num_regs to n-1              procedures (SOPs, Work Instructions)

Edge types:
  similarity score > threshold  compliance relationship
  edge weight = 1.0 - similarity (lower = stronger)

CSR Format used by all graph kernels:
  row_ptr[i]       start index of node i's edges
  row_ptr[i+1]     end index of node i's edges
  col_indices[e]   destination node of edge e
  values[e]        edge weight of edge e
```

---

## Algorithm Map

```
Question                         Algorithm          Kernel
------------------------------------------------------------------
What is related to what?         Floyd-Warshall     floyd_warshall.cu
What are the gaps?               BFS                bfs.cu
What depends on what?            Topological Sort   topological_sort.cu
What matters most?               PageRank           pagerank.cu
Minimum set to cover all regs?   Kruskal MST        kruskal_mst.cu
Strongest compliance path?       Dijkstra           dijkstra.cu
```

---

## Benchmark Results

RTX 5080, 10,000 regs x 500 procs x 768 dims

```
Kernel              Time        Notes
--------------------------------------------------
cosine_naive        9.51 ms     baseline
cosine_tiled        6.88 ms     1.38x faster
cosine_warp         3.99 ms     2.38x faster
threshold_filter    1.39 ms     5M pairs to 1.2M edges
floyd_warshall      1.81 ms     1000 node demo
bfs                 0.70 ms     single source
topological_sort    1.58 ms     4 hierarchy levels
pagerank            1.21 ms     1000 nodes, 6 iterations to converge
kruskal_mst         1.10 ms     1000 nodes, 5 Boruvka rounds, 999 MST edges
dijkstra            0.53 ms     single source, 2 iterations to converge
--------------------------------------------------
Total pipeline      ~12.31 ms   CPU equivalent: hours
```

---

## Current Build Status

```
CUDA Kernels        COMPLETE   (10/10 kernels implemented)
Python Pipeline     COMPLETE   (parser, embedder, scraper)
pybind11 Bridge     COMPLETE   (similarity + graph bindings)
CLI                 COMPLETE   (analyze, report, benchmark)
Tests               MISSING    (no test suite exists)
```

---

## Intended CLI Usage

```bash
# Full compliance analysis
regmap analyze --regs ./doe_orders/ --procs ./procedures/

# With knowledge graph output
regmap analyze --regs ./doe_orders/ --procs ./procedures/ --graph

# Set similarity threshold
regmap analyze --regs ./doe_orders/ --procs ./procedures/ --threshold 0.75

# Query specific regulation
regmap query --source "DOE-O-414.1D" --algorithm bfs

# Generate gap report
regmap report --type gaps --output gap_report.pdf

# Benchmark kernels
regmap benchmark --kernel cosine
```

---

## Important Rules for AI Assistance

- .cu files NEVER read PDFs or text files — they only receive float arrays
- PDF parsing happens entirely in pipeline/parser.py
- Text to embeddings happens entirely in pipeline/embedder.py
- CUDA kernels receive float arrays and return float arrays only
- The pybind11 bridge in bindings/ connects Python to CUDA
- All graph algorithms operate on CSR format sparse graphs
- Do not suggest installing Linux NVIDIA drivers in WSL2
- Do not use std::to_string or std::string in .cu files
- Always target sm_120 for this machine
