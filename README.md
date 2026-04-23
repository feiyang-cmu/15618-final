# Accelerating MoE Token Routing and Dispatch on GPU


## Overview

This project targets the **token packing bottleneck** in Mixture-of-Experts (MoE) inference.
In Expert Parallelism (EP) multi-GPU settings, as computation spreads across more cards,
the local packing step (scatter/permutation) becomes a growing fraction of total latency.
We implement and compare 3 parallel strategies for the packing and dispatch pipeline, 
analyze the scaling behavior from one to several GPUs, and characterize the tradeoffs 
between synchronization cost, memory bandwidth utilization, and communication overhead.

**Key insight:** As GPU count increases, Step 3 FFN time shrinks linearly,
but Step 1 packing time stays constant, **packing fraction grows and becomes the bottleneck.**

| Step | Operation                                    | Scope        | Scaling Behavior       |
|------|----------------------------------------------|--------------|------------------------|
| 1    | Scatter / Token Permutation / Packing        | Single GPU   | **Constant**           |
| 2    | AllToAll dispatch                            | Cross-GPU    | Grows with N           |
| 3    | Local Expert FFN                             | Single GPU   | **Shrinks as N grows** |
| 4    | AllToAll combine                             | Cross-GPU    | Grows with N           |
| 5    | Local Unpermute + Weighted Sum               | Single GPU   | Constant               |


## Project Structure
 
```
.
├── include/moe/
│   ├── scatter.cuh         # IScatter interface + ScatterTimings + factory declaration
│   ├── types.h             # MoEParams, MoEBuffers (plain device-pointer structs)
│   ├── cpu_reference.hpp   # cpu_scatter + verify_scatter_outputs declarations
│   ├── npy_io.hpp          # NpyArray: 2-D .npy loader (<f2 / <f4 / <i4)
│   ├── fp16_host.hpp       # Host-side fp16 helpers + deterministic random fill
│   └── utils.cuh           # MOE_CUDA_CHECK, CudaTimer, device_alloc, device_zero_async
│
├── src/
│   ├── common/
│   │   ├── cpu_reference.cpp   # cpu_scatter (serial scatter) + verify_scatter_outputs
│   │   └── npy_io.cpp          # load_npy implementation
│   └── scatter/
│       ├── factory.cu          # make_scatter(): dispatches "atomic" / "sort" / "warp"
│       ├── scatter_atomic.cu   # Atomic strategy: shared-mem histogram + CUB prefix sum + atomicAdd scatter
│       ├── scatter_sort.cu     # Sort strategy: CUB radix sort + binary-search bounds + block-per-row gather
│       └── scatter_warp.cu     # Warp strategy: ballot-based warp-cooperative scatter (no sort)
│
├── apps/
│   ├── verify.cu           # Correctness harness: runs CPU reference + any IScatter, checks invariants
│   ├── bench_scatter.cu    # Isolated scatter benchmark: auto-scales iters, reports min/p50/p90/p99/mean
│   ├── bench_e2e.cu        # End-to-end single-GPU MoE layer: routing → IScatter → cuBLAS GEMM → unpack
│   ├── bench_ep.cu         # 2-GPU Expert Parallelism benchmark: scatter + NCCL AllToAll dispatch/combine
│   └── grouped_gemm_test.cu # CUTLASS grouped GEMM vs sequential cuBLAS hgemm microbench (requires CUTLASS)
│
├── data/
│   ├── gen_synthetic_routing.py    # Generate Uniform / Zipf routing .npy files
│   └── extract_qwen_routing.py     # Extract real routing from Qwen1.5-MoE-A2.7B Layer 0 gate
│
├── analysis/
│   ├── scaling_projection.py   # EP scaling projection (not yet implemented)
│   └── plot_results.py         # Result figures (not yet implemented)
│
├── profiling/
│   └── run_ncu.sh          # Nsight Compute profiling script
│
├── results/                # Generated .npy routing data and benchmark outputs
├── Makefile
└── requirements.txt
```


### 1. Data Generation (`data/`) — *[Python]*
- **`extract_qwen_routing.py`**
  - **Core Functions:** `load_qwen_model()`, `extract_layer_0_gate()`, `compute_real_routing()`
  - Loads Qwen1.5-MoE-A2.7B via HuggingFace `transformers`. Feeds WikiText-103 text through
    the Layer 0 gate network to obtain real token-to-expert assignments and gating weights.
    Serializes `embeddings[T, d]`, `assignments[T, K]`, `weights[T, K]` to `.npy`.
- **`gen_synthetic_routing.py`**
  - **Core Functions:** `gen_uniform_routing()`, `gen_zipf_routing()`
  - Generates Uniform (imbalance ~1.15×) and Zipf (imbalance ~12×) routing distributions
    using the Gumbel-max trick for exact without-replacement sampling. Used as controlled
    baselines against the real Qwen routing data.


### 2. CPU Reference & I/O (`src/common/`) — *[C++]*
- **`cpu_reference.cpp`**
  - **Core Functions:** `cpu_scatter()`, `verify_scatter_outputs()`
  - Serial single-core scatter: count → exclusive prefix sum → memcpy-based scatter.
    Produces ground-truth `packed`, `expert_start`, `expert_count`, `perm` arrays.
    `verify_scatter_outputs()` checks six invariants order-independently so it works
    against non-deterministic GPU strategies (atomic/warp permute rows within each expert).
- **`npy_io.cpp`**
  - **Core Functions:** `load_npy()`
  - Minimal 2-D `.npy` loader supporting `<f2`, `<f4`, `<i4` dtypes. Used by all
    benchmark binaries to read routing data from `results/`.


### 3. Scatter Strategies (`src/scatter/`) — *[CUDA]*
- **`factory.cu`**
  - **Core Functions:** `make_scatter()`
  - Single dispatch point: maps strategy name string to the corresponding
    `std::unique_ptr<IScatter>` constructor.
- **`scatter_atomic.cu`**
  - **Core Functions:** `count_kernel()`, `scatter_kernel()`
  - Shared-memory histogram reduces global atomic pressure: each block accumulates
    per-expert counts in `__shared__` memory then issues one `atomicAdd` per (block, expert).
    CUB `ExclusiveSum` converts counts to offsets. A scatter kernel uses one `atomicAdd`
    per (token, k) slot to claim a write position, then copies the token row and weight.
- **`scatter_sort.cu`**
  - **Core Functions:** `init_values_kernel()`, `expert_bounds_kernel()`, `gather_rows_kernel()`
  - CUB `DeviceRadixSort::SortPairs` on (expert\_id, slot\_index) using only log₂(E) bits.
    E+1 parallel binary searches compute `expert_start`. A block-per-output-row gather
    kernel copies each token row to its sorted position with full memory coalescing.
    Eliminates all write conflicts at the cost of two full HBM passes.
- **`scatter_warp.cu`**
  - **Core Functions:** `count_kernel()`, `ballot_scatter_kernel()`
  - Ballot-based warp-cooperative scatter (no sort). Each warp uses `__ballot_sync` to
    identify which lanes target the same expert, issues one `atomicAdd` per warp per expert
    to claim slots, then each matching lane copies its embedding row independently.
    Reduces atomic ops from O(T×K) to O(T×K/32) and eliminates the sort's two HBM passes.


### 4. Applications (`apps/`) — *[CUDA]*
- **`verify.cu`**
  - **Core Functions:** `run_mini_tests()`, `run_full()`
  - Correctness harness. Runs mini synthetic inputs through CPU reference and any
    `IScatter` strategy, checks all six `verify_scatter_outputs()` invariants.
    Accepts any strategy name and any `.npy` prefix via CLI.
- **`bench_scatter.cu`**
  - **Core Functions:** `parse()`, `time_iters()`, `compute_stats()`
  - Isolated scatter kernel benchmark. Uploads inputs once, runs warmup iterations,
    then auto-scales timed iterations until total wall time exceeds 50 ms. Reports
    min / p50 / p90 / p99 / mean / stdev and effective HBM bandwidth.
- **`bench_e2e.cu`**
  - **Core Functions:** `run_e2e()`, `run_mini_test()`, `run_full()`
  - End-to-end single-GPU MoE layer benchmark. Supports two routing modes (naive:
    3 separate kernel launches; fused: gate + softmax + topK in one kernel) and
    any `IScatter` strategy. Expert FFNs use sequential `cublasHgemm` calls (up-proj
    then down-proj per expert). Measures per-stage latency breakdown.
- **`bench_ep.cu`**
  - **Core Functions:** `rank_worker()`, `Barrier2`
  - 2-GPU Expert Parallelism benchmark using NCCL. Two CPU threads each control one
    GPU. Each GPU runs local scatter via `IScatter`, then NCCL `AllToAll` dispatches
    packed tokens to the GPU owning each expert, and a second `AllToAll` combines
    results back. Measures scatter, dispatch, and combine latency independently to
    quantify how packing fraction grows as FFN is distributed across GPUs.
    Requires NCCL; build with `NCCL=<path>`.
- **`grouped_gemm_test.cu`**
  - **Core Functions:** `run_sequential_cublas()`, `run_grouped_cutlass()`
  - Microbench comparing sequential cuBLAS hgemm (one call per expert) against a
    single CUTLASS `GemmGrouped` kernel launch across all E experts. Requires CUTLASS
    headers; not built by default on GHC machines.


### 5. Profiling (`profiling/`) — *[Shell]*
- **`run_ncu.sh`**
  - Nsight Compute script. Captures HBM bandwidth, L2 sector traffic, shared memory
    bank conflicts, and atomic op counts per strategy. Outputs `.ncu-rep` to `results/`.


### 6. Analysis & Visualization (`analysis/`) — *[Python]*
- **`scaling_projection.py`**
  - Models packing as a fixed cost and FFN as linearly scaling with GPU count.
    Projects packing fraction of total EP latency as a function of N GPUs under
    Uniform and Zipf routing. (Implementation in progress.)
- **`plot_results.py`**
  - Generates strategy latency comparison bar charts, HBM bandwidth utilization plots,
    and EP scaling projection curves from benchmark CSV outputs. (Implementation in progress.)


### 7. Results (`results/`)
Generated `.npy` routing data, benchmark outputs, Nsight Compute `.ncu-rep` reports,
and figures. Not checked into version control (except `.gitkeep`).


## Key Data Interfaces
 
| Symbol           | Shape     | Dtype   | Description                              |
|------------------|-----------|---------|------------------------------------------|
| `embeddings`     | [T, d]    | float16 | Input token embeddings                   |
| `assignments`    | [T, K]    | int32   | Top-K expert IDs per token               |
| `router_weights` | [T, K]    | float32 | Gating weights per (token, expert)       |
| `packed`         | [T×K, d]  | float16 | Expert-grouped output buffer             |
| `expert_start`   | [E+1]     | int32   | Exclusive prefix sum of expert counts    |
| `expert_count`   | [E]       | int32   | Token count per expert                   |
| `perm`           | [T×K]     | int32   | perm[pos] = source token index           |
 

## Scatter Strategies
 
All three strategies implement the `IScatter` interface in `include/moe/scatter.cuh`
and are selected at runtime via `make_scatter("atomic" | "sort" | "warp")`.
 
**Atomic** (`scatter_atomic.cu`)
Three stages: a shared-memory histogram kernel reduces global atomic pressure by
accumulating per-block counts before a single `atomicAdd` per (block, expert); CUB
`ExclusiveSum` converts counts to offsets; a scatter kernel then uses one `atomicAdd`
per (token, k) slot to claim a write position. Under skewed routing, the scatter
stage serializes on hot experts.
 
**Sort** (`scatter_sort.cu`)
Four stages: initialize slot indices 0..TK-1; CUB `DeviceRadixSort::SortPairs` on
(expert\_id, slot\_index) using only log₂(E) bits; E+1 parallel binary searches to
compute `expert_start`; a block-per-output-row gather kernel copies each source token
row to its sorted position with full memory coalescing. Eliminates all write conflicts.

**Warp** (`scatter_warp.cu`)
Three stages: shared-memory histogram count; CUB `ExclusiveSum`; ballot-based scatter.
Each warp uses `__ballot_sync` and `__popc` to identify lanes targeting the same expert,
issues one `atomicAdd` per warp per expert to claim consecutive slots, then each
matching lane copies its own embedding row independently. Reduces atomic operations
from O(T×K) to O(T×K/32) and eliminates the two full HBM passes required by the
sort-based approach.


## Build
 
The project requires CUDA 11.7 and gcc-11 on GHC machines. System CUDA 11.7 ships
cuBLAS 11 which is incompatible with the code; use the cuBLAS 12 from the system
pytorch installation instead.
 
```bash
# GHC machines (sm_75, RTX 2080)
make ARCH=sm_75 \
  NVCCFLAGS="-O3 -std=c++17 -arch=sm_75 --expt-relaxed-constexpr -Iinclude -ccbin gcc-11 -lstdc++ -lm" \
  CUBLAS12=/opt/pytorch/lib/python3.12/site-packages/nvidia/cublas

# PSC machines (sm_70, V100, with NCCL for bench_ep)
make ARCH=sm_70 \
  NVCCFLAGS="-O3 -std=c++17 -arch=sm_70 --expt-relaxed-constexpr -Iinclude" \
  CUBLAS_LINK="" \
  NCCL=/jet/home/<user>/.local/lib/python3.6/site-packages/nvidia/nccl
 
# Other machines (default in Makefile, sm_86, conda cuBLAS)
make
```
 
All binaries are placed in `build/bin/`.

 
## Quick Start
 
### 1. Set up the environment
 
```bash
python3 -m venv /tmp/moe_env
source /tmp/moe_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
 
### 2. Generate routing data
 
```bash
# Synthetic Uniform and Zipf distributions
python data/gen_synthetic_routing.py --tokens 2048 --experts 64 --topk 4 --dim 2048
 
# Real Qwen1.5-MoE-A2.7B routing (requires ~5 GB download)
export HF_HOME=/tmp/hf_cache
python data/extract_qwen_routing.py --topk 4
 
# Dry-run (no model download, mock Qwen-style data)
python data/extract_qwen_routing.py --dry-run --topk 4
```
 
### 3. Build
 
```bash
make ARCH=sm_75 \
  NVCCFLAGS="-O3 -std=c++17 -arch=sm_75 --expt-relaxed-constexpr -Iinclude -ccbin gcc-11 -lstdc++ -lm" \
  CUBLAS12=/opt/pytorch/lib/python3.12/site-packages/nvidia/cublas
```
 
### 4. Verify correctness
 
```bash
# Mini-tests only
./build/bin/verify
 
# Full correctness check: verify <prefix> <E> <data_dir> <strategy>
./build/bin/verify syn_uniform_T2048 64 results atomic
./build/bin/verify syn_zipf_T2048   64 results sort
./build/bin/verify syn_zipf_T2048   64 results warp
```
 
### 5. Benchmark scatter strategies
 
```bash
# Auto-scaled iterations, atomic strategy
./build/bin/bench_scatter --prefix=syn_zipf_T8192 --strategy=atomic
 
# Fixed iterations, sort strategy
./build/bin/bench_scatter --prefix=syn_uniform_T2048 --strategy=sort --iters=200 --warmup=20
 
# Warp strategy
./build/bin/bench_scatter --prefix=syn_zipf_T8192 --strategy=warp
```
 
### 6. End-to-end MoE layer benchmark
 
```bash
./build/bin/bench_e2e
./build/bin/bench_e2e syn_uniform_T2048
./build/bin/bench_e2e syn_zipf_T2048
```

### 7. 2-GPU Expert Parallelism benchmark

Requires 2 GPUs with P2P access and NCCL. Build with `NCCL=<path>` as shown above.

```bash
./build/bin/bench_ep --prefix=syn_uniform_T2048 --strategy=sort
./build/bin/bench_ep --prefix=syn_zipf_T2048   --strategy=sort
./build/bin/bench_ep --prefix=syn_uniform_T8192 --strategy=warp
./build/bin/bench_ep --prefix=syn_zipf_T8192   --strategy=warp
```
 
### 8. Nsight Compute profiling
 
```bash
bash profiling/run_ncu.sh atomic
bash profiling/run_ncu.sh sort
bash profiling/run_ncu.sh warp
```
 
### 9. Analysis and plotting
 
```bash
python analysis/scaling_projection.py --max-gpus 64
python analysis/plot_results.py
```