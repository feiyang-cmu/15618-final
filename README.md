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

### 1. Data Generation (`data/`) — *[Python]*
- **`gen_routing.py`**
  - **Core Functions:** `gen_uniform_routing()`, `gen_zipf_routing()`, `save_routing()`
  - Generates synthetic `assignments[T][K]` and `router_weights[T][K]` under uniform or Zipf
    distributions. Serializes to `.npy` for use across all kernels and benchmarks.

### 2. Baseline (`baseline/`) — *[C++ / CUDA]*
- **`sequential_cpu.cpp`**
  - **Core Functions:** `cpu_scatter()`
  - Serial reference implementation. Produces ground-truth `packed`, `expert_start`,
    `expert_count` for correctness verification.
- **`naive_gpu.cu`**
  - **Core Functions:** `naive_scatter_kernel()`
  - Per-expert block assignment baseline. Each CUDA block handles one expert;
    no load balancing. Serves as the performance floor.


### 3. Packing Strategies (`strategies/`) — *[CUDA]*
- **`strategy_a_sort.cu`**
  - **Core Functions:** `radix_sort_pairs()`, `coalesced_gather_kernel()`
  - Sort-based packing. Radix sorts `(expert_id, token_k_index)` pairs then performs a
    coalesced gather into `packed`. Eliminates write conflicts at the cost of two full HBM passes.
- **`strategy_b_atomic.cu`**
  - **Core Functions:** `atomic_scatter_kernel()`
  - One thread per `(token, k)` pair. Each thread calls `atomicAdd` on a per-expert counter
    to claim its write slot. Simple but prone to atomic contention under Zipf routing.
- **`strategy_c_warp.cu`**
  - **Core Functions:** `warp_scan_scatter_kernel()`
  - Warp-cooperative packing. Uses `__ballot_sync` + `__popc` to census expert assignments
    within each warp, issues one `atomicAdd` per warp per expert, then broadcasts the base
    offset so all lanes write independently.
    Reduces atomic ops from O(T·K) → O(T·K/32).


### 4. Fused Kernel (`fused/`) — *[CUDA]*
- **`fused_routing_pack.cu`**
  - **Core Functions:** `fused_gate_topk_scatter_kernel()`
  - Fuses the routing decision (Gate projection + Top-K selection) directly with the scatter operation.
    Eliminates intermediate HBM writes for routing metadata.
    Outputs `packed[T*K][d]` and `expert_count[E]`.


### 5. End-to-End Benchmark (`e2e/`) — *[CUDA]*
- **`bench_e2e.cu`**
  - **Core Functions:** `run_moe_layer()`
  - Fused Kernel 1 → grouped `cublasGemmEx` (one per expert) → Unpermute + weighted sum.
    Measures full MoE layer latency and throughput (tokens/s).


### 6. Correctness Verification (`verify/`) — *[CUDA]*
- **`verify.cu`**
  - **Core Functions:** `check_packed()`, `check_expert_offsets()`
  - Compares any GPU strategy output against the CPU reference.
    Checks value correctness and that expert regions are non-overlapping and contiguous.


### 7. Benchmarking (`benchmark/`) — *[CUDA]*
- **`bench_kernel.cu`**
  - **Core Functions:** `run_sweep()`
  - Isolated per-kernel timing harness. Sweeps `(T, K, E, distribution)` and reports
    latency and effective HBM bandwidth for all three strategies.
- **`overlap_simulate.cu`**
  - **Core Functions:** `simulate_alltoall_overlap()`
  - Simulates AllToAll latency via CUDA streams + `cudaMemcpyAsync`, overlapping with
    local packing compute to estimate EP pipeline hiding.


### 8. Profiling (`profiling/`) — *[Shell]*
- **`run_ncu.sh`**
  - Nsight Compute script. Captures HBM bandwidth, L2 sector traffic, shared memory bank
    conflicts, and atomic op counts per strategy. Outputs `.ncu-rep` to `results/`.


### 9. Analysis & Visualization (`analysis/`) — *[Python]*
- **`scaling_projection.py`**
  - **Core Functions:** `project_packing_fraction()`, `plot_ep_scaling()`
  - Models packing as fixed cost and FFN as linearly scaling. Projects packing fraction
    of total EP latency as a function of N GPUs under uniform and Zipf routing.
- **`plot_results.py`**
  - **Core Functions:** `plot_strategy_comparison()`, `plot_roofline()`
  - Generates all final figures: strategy latency comparison, atomic op counts,
    HBM bandwidth utilization, and EP scaling projection curves.


### 10. Results (`results/`)
Raw benchmark CSVs, Nsight Compute `.ncu-rep` reports, and generated figures.
Not checked into version control (except `.gitkeep`).


## Key Data Interfaces

| Symbol | Shape | Dtype | Description |
|---|---|---|---|
| `embeddings` | `[T, d]` | float16 | Input token embeddings |
| `assignments` | `[T, K]` | int32 | Top-K expert IDs per token |
| `router_weights` | `[T, K]` | float16 | Gating weights per (token, expert) |
| `packed` | `[T×K, d]` | float16 | Expert-grouped output buffer |
| `expert_start` | `[E]` | int32 | Start offset per expert in `packed` |
| `expert_count` | `[E]` | int32 | Token count per expert |


## TODO — By Phase

### Phase 1 — Infrastructure & Baseline (~6h)
- [ ] `data/gen_routing.py`: uniform + Zipf routing generation, serialize to `.npy`
- [ ] `baseline/sequential_cpu.cpp`: correct serial scatter for ground-truth output
- [ ] `baseline/naive_gpu.cu`: per-expert block assignment, verify against CPU reference
- [ ] `verify/verify.cu`: automated correctness harness
- [ ] **Signal**: all strategy outputs match CPU reference on both distributions

### Phase 2 — Three Packing Strategies (~10h)
- [ ] `strategies/strategy_a_sort.cu`: Radix Sort → coalesced gather
- [ ] `strategies/strategy_b_atomic.cu`: per-thread atomicAdd scatter
- [ ] `strategies/strategy_c_warp.cu`: warp-cooperative ballot + batched atomicAdd
- [ ] `benchmark/bench_kernel.cu`: unified timing harness, parameter sweep
- [ ] **Signal**: Strategy C shows fewer atomic ops (NCU) and lower HBM traffic than A+B under Zipf

### Phase 3 — Fused Routing Kernel (~8h)
- [ ] `fused/fused_routing_pack.cu`: Gate + Softmax + Top-K + Scatter in one kernel
- [ ] `e2e/bench_e2e.cu`: fused kernel → cuBLAS grouped GEMM → unpermute
- [ ] **Signal**: fused kernel reduces HBM round-trips vs. chained kernels (NCU)

### Phase 4 — Profiling & Analysis (~6h)
- [ ] `profiling/run_ncu.sh`: collect bandwidth, bank conflicts, atomic counts
- [ ] `analysis/scaling_projection.py`: project packing fraction across GPU counts
- [ ] `benchmark/overlap_simulate.cu`: simulate AllToAll + compute overlap
- [ ] `analysis/plot_results.py`: all final figures
- [ ] **Signal**: projection shows packing fraction exceeds 40% at 16+ GPUs under Zipf

### Phase 5 — Write-up & Polish (~4h)
- [ ] Results tables: throughput, latency breakdown, HBM utilization per strategy
- [ ] Final report: motivation → methodology → results → EP scaling conclusion
- [ ] Clean repo, confirm `verify.cu` passes all strategies, tag final commit


## Quick Start

```bash
# Generate routing data
python data/gen_routing.py --tokens 4096 --experts 64 --topk 2 --dist zipf

# Correctness check
nvcc -O3 -arch=sm_80 verify/verify.cu -o verify && ./verify

# Run strategy benchmark sweep
nvcc -O3 -arch=sm_80 benchmark/bench_kernel.cu -o bench && ./bench

# NCU profiling
bash profiling/run_ncu.sh strategy_c_warp

# EP scaling projection
python analysis/scaling_projection.py --max-gpus 64
```