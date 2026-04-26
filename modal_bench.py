"""Run MoE packing benchmarks on Modal cloud GPUs (A100).

Mounts the local working tree — no git push required to iterate.

Usage:
    modal run modal_bench.py                  # default: 2 GPUs, all strategies
    modal run modal_bench.py --n-gpus 1       # single-GPU baseline
    modal run modal_bench.py --n-gpus 4       # 4 GPUs
    modal run modal_bench.py --n-gpus 8       # 8 GPUs
    modal run modal_bench.py --strategy warp  # pick strategy
    modal run modal_bench.py --mode prefill   # multi-round prefill benchmark
    modal run modal_bench.py --mode real      # extract real Qwen traces + bench

GPU mapping (ep/prefill mode):
    2 GPU  -> T=2048
    4 GPU  -> T=4096
    8 GPU  -> T=8192

T scales with GPU count so all-to-all chunks stay ~32 MB per rank pair.
"""

import sys

import modal

app = modal.App("moe-scatter-bench")

# ── Image: CUDA 12.4 devel + NCCL + CUTLASS ──────────────────────────────────

_base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "cuda-nsight-compute-12-4", "cuda-nsight-systems-12-4")
    .pip_install("numpy")  # for on-demand decode-shape data generation
    .run_commands(
        "apt-get install -y --allow-change-held-packages libnccl2 libnccl-dev"
    )
    .run_commands(
        "ln -sf /usr/local/cuda/lib64 /usr/local/cuda/lib",
        "mkdir -p /opt/nccl/lib /opt/nccl/include",
        "ln -sf /usr/lib/x86_64-linux-gnu/libnccl.so.2 /opt/nccl/lib/libnccl.so.2",
        "ln -sf /usr/include/nccl.h /opt/nccl/include/nccl.h",
    )
    .run_commands(
        "git clone --depth 1 -b v3.5.1 "
        "https://github.com/NVIDIA/cutlass.git /opt/cutlass"
    )
)

# add_local_dir must be last (Modal requirement)
image = _base_image.add_local_dir(
    ".",
    remote_path="/workspace",
    ignore=[".git", "build/", "third_party/", "analysis/", "**/__pycache__"],
)

# Image with PyTorch + transformers for real Qwen trace extraction
# pip_install before add_local_dir
image_torch = (
    _base_image
    .pip_install("torch", "transformers", "datasets", "numpy", "accelerate")
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=[".git", "build/", "third_party/", "analysis/", "**/__pycache__"],
    )
)

# ── Shared build + bench logic ────────────────────────────────────────────────

MAKE_CMD = [
    "make", "-j",
    "ARCH=sm_80",
    "CUBLAS12=/usr/local/cuda",
    "CUTLASS=/opt/cutlass",
    "NCCL=/opt/nccl",
]

# T scales with GPU count for constant all-to-all chunk size
GPU_TO_T = {1: 2048, 2: 2048, 4: 4096, 8: 8192}

STRATEGIES = ["atomic", "sort", "warp", "vec", "csort"]


def _run_cmd(cmd, label=""):
    import subprocess
    try:
        r = subprocess.run(
            cmd, cwd="/workspace", capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        return f"=== {label} ===\nTIMEOUT after 600s\n"
    out = f"=== {label} ===\n{r.stdout}"
    if r.stderr.strip():
        out += f"STDERR: {r.stderr}\n"
    if r.returncode != 0:
        out += f"[exit code {r.returncode}]\n"
    return out + "\n"


def _build():
    import subprocess
    r = subprocess.run(MAKE_CMD, cwd="/workspace", capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"BUILD FAILED:\n{r.stdout}\n{r.stderr}\n"
    return True, "Build OK\n\n"


def _gpu_info(n_gpus):
    import subprocess
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return f"GPUs ({n_gpus} requested):\n{r.stdout}\n"


def _run_ep(n_gpus: int, strategy: str):
    """Single-iteration EP benchmark (bench_ep)."""
    output = _gpu_info(n_gpus)
    ok, msg = _build()
    output += msg
    if not ok:
        return output

    strategies = [strategy] if strategy != "all" else STRATEGIES
    T = GPU_TO_T.get(n_gpus, 2048)

    if n_gpus == 1:
        prefixes = [f"syn_uniform_T{T}", f"syn_zipf_T{T}"]
        for s in strategies:
            output += _run_cmd(
                ["./build/bin/verify", f"syn_uniform_T{T}", "64", "results", s],
                f"verify {s}",
            )
        for prefix in prefixes:
            for s in strategies:
                output += _run_cmd(
                    ["./build/bin/bench_scatter",
                     f"--prefix={prefix}", "--E=64", f"--strategy={s}"],
                    f"bench_scatter {prefix} {s}",
                )
            for s in strategies:
                output += _run_cmd(
                    ["./build/bin/bench_e2e", prefix, "64", "results", s],
                    f"bench_e2e {prefix} {s}",
                )
    else:
        prefixes = [f"syn_uniform_T{T}", f"syn_zipf_T{T}"]
        for prefix in prefixes:
            for s in strategies:
                output += _run_cmd(
                    ["./build/bin/bench_ep",
                     f"--prefix={prefix}", f"--strategy={s}",
                     f"--n-gpus={n_gpus}"],
                    f"bench_ep {prefix} {s} ({n_gpus} GPUs)",
                )

    return output


def _run_prefill(n_gpus: int, strategy: str):
    """Multi-round prefill benchmark (bench_prefill).

    Default behaviour (`--strategy=all`) sweeps the additive optimization
    chain so we get a single table showing cumulative improvement:

        baseline   = warp  + serial
        +vec       = vec   + serial
        +csort     = csort + serial
        +overlap   = csort + overlap  (csort wins from prior pass + 2-stream)

    A specific --strategy=foo runs only that strategy in both serial and
    overlap modes for direct A/B comparison.
    """
    output = _gpu_info(n_gpus)
    ok, msg = _build()
    output += msg
    if not ok:
        return output

    T = GPU_TO_T.get(n_gpus, 2048)

    if strategy == "all":
        # Cumulative chain: each row adds one optimization on the previous.
        # `atomic` is the true naive baseline (per-thread atomic + per-thread
        # d-element copy loop) — what a from-scratch implementation looks like.
        # All speedups in the report are claimed against this baseline.
        chain = [
            ("atomic", 0, 1, "naive baseline (atomic + serial)"),
            ("warp",   0, 1, "+ warp gather (radix sort + warp-per-row)"),
            ("vec",    0, 1, "+ uint4 (vectorized gather)"),
            ("csort",  0, 1, "+ csort (counting sort, replace radix + bounds)"),
            ("csort",  1, 1, "+ overlap (2-stream comm/compute)"),
            ("csort",  1, 2, "+ chunked dispatch K=2 (1F1B intra-round)"),
            ("csort",  1, 4, "+ chunked dispatch K=4"),
        ]
        for dist in ["uniform", "zipf"]:
            prefix = f"syn_{dist}_T{T}_N32"
            for s, ov, ch, label in chain:
                output += _run_cmd(
                    ["./build/bin/bench_prefill",
                     f"--prefix={prefix}", f"--strategy={s}",
                     f"--n-gpus={n_gpus}", f"--overlap={ov}",
                     f"--chunks={ch}"],
                    f"prefill {dist}  {label}",
                )
        return output

    # Single strategy A/B (serial vs overlap)
    for dist in ["uniform", "zipf"]:
        prefix = f"syn_{dist}_T{T}_N32"
        for ov in (0, 1):
            tag = "overlap" if ov else "serial"
            output += _run_cmd(
                ["./build/bin/bench_prefill",
                 f"--prefix={prefix}", f"--strategy={strategy}",
                 f"--n-gpus={n_gpus}", f"--overlap={ov}"],
                f"bench_prefill {prefix} {strategy} ({n_gpus} GPUs, {tag})",
            )

    return output


def _run_real(n_gpus: int, strategy: str):
    """Extract real Qwen traces on GPU, then run bench_prefill with them."""
    import subprocess
    output = _gpu_info(n_gpus)

    # Extract real routing traces at multiple T sizes
    T = GPU_TO_T.get(n_gpus, 2048)
    output += f"Extracting real Qwen routing traces (T={T})...\n"
    r = subprocess.run(
        ["python", "data/extract_qwen_routing.py",
         "--topk", "4", f"--max-tokens={T}",
         "--out", "results", "--n-sentences", "2000"],
        cwd="/workspace", capture_output=True, text=True, timeout=600,
    )
    output += r.stdout
    if r.returncode != 0:
        output += f"STDERR: {r.stderr}\n[exit code {r.returncode}]\n"
        return output

    # Build
    ok, msg = _build()
    output += msg
    if not ok:
        return output

    strategies = [strategy] if strategy != "all" else STRATEGIES

    if n_gpus == 1:
        for s in strategies:
            output += _run_cmd(
                ["./build/bin/bench_scatter",
                 "--prefix=real", "--E=64", f"--strategy={s}"],
                f"bench_scatter real {s}",
            )
            output += _run_cmd(
                ["./build/bin/bench_e2e", "real", "64", "results", s],
                f"bench_e2e real {s}",
            )
    else:
        for s in strategies:
            output += _run_cmd(
                ["./build/bin/bench_ep",
                 "--prefix=real", f"--strategy={s}",
                 f"--n-gpus={n_gpus}"],
                f"bench_ep real {s} ({n_gpus} GPUs)",
            )

    return output


def _run_decode(n_gpus: int, strategy: str):
    """Decode-shaped sweep: T ∈ {16, 32, 64, 128, 256} per round.

    At small T, comm/compute ratio flips relative to prefill — EP overhead
    grows from ~10% (prefill T=2048) to >50% (decode T=16). Our overlap
    optimization should show much larger relative wins in this regime.

    For each T, runs serial vs csort+overlap and reports the speedup.
    Skips T values where T % n_gpus != 0 (split-input requires T_local ≥ 1).
    """
    import subprocess

    output = _gpu_info(n_gpus)

    # Generate decode-shaped routing data inside the container (numpy is in
    # the base image now). Multi-round (N_rounds=32) so the bench loops.
    output += "Generating decode-shaped routing data...\n"
    r = subprocess.run(
        ["python", "data/gen_synthetic_routing.py",
         "--decode-sweep", "--rounds", "32"],
        cwd="/workspace", capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        output += f"data gen FAILED:\n{r.stdout}\n{r.stderr}\n"
        return output
    output += "data gen OK\n\n"

    ok, msg = _build()
    output += msg
    if not ok:
        return output

    # Sweep T sizes that divide n_gpus evenly. Compare serial vs csort+overlap
    # at each T to highlight that overlap matters more as T shrinks.
    decode_T = [16, 32, 64, 128, 256]
    s = strategy if strategy != "all" else "csort"
    for dist in ("uniform", "zipf"):
        for T in decode_T:
            if T % n_gpus != 0:
                continue
            prefix = f"syn_{dist}_T{T}_N32"
            for ov in (0, 1):
                tag = "overlap" if ov else "serial"
                output += _run_cmd(
                    ["./build/bin/bench_prefill",
                     f"--prefix={prefix}", f"--strategy={s}",
                     f"--n-gpus={n_gpus}", f"--overlap={ov}"],
                    f"decode T={T} {dist} {tag}",
                )
    return output


def _run_grouped_gemm(strategy: str):
    """Re-test CUTLASS grouped GEMM vs sequential cuBLAS on A100.

    The original negative result was on RTX 4070 Super (56 SMs, 78 TFLOPS).
    A100 has 108 SMs and 312 TFLOPS — single per-expert GEMM (M=128) may
    leave SMs idle, which grouped GEMM can fill.
    """
    import subprocess
    output = _gpu_info(1)
    ok, msg = _build()
    output += msg
    if not ok:
        return output

    tile_sweep = [
        (64,  64,  32, 32, 32),
        (64,  128, 32, 32, 64),
        (128, 64,  32, 64, 32),
        (128, 128, 32, 64, 64),
        (256, 128, 32, 64, 64),
        (128, 256, 32, 64, 64),
    ]
    for tm, tn, tk, wm, wn in tile_sweep:
        defs = (f"-DTILE_M={tm} -DTILE_N={tn} -DTILE_K={tk} "
                f"-DWARP_M={wm} -DWARP_N={wn}")
        output += _run_cmd(
            ["bash", "-lc",
             "rm -f build/obj/grouped_gemm_test.o build/bin/grouped_gemm_test && "
             f"NVCC_FLAGS_EXTRA='{defs}' make build/bin/grouped_gemm_test "
             "ARCH=sm_80 CUBLAS12=/usr/local/cuda CUTLASS=/opt/cutlass NCCL=/opt/nccl"],
            f"build tile {tm}x{tn}x{tk}",
        )
        for dist in ("uniform", "zipf"):
            output += _run_cmd(
                ["./build/bin/grouped_gemm_test", dist],
                f"tile {tm}x{tn}x{tk}/{wm}x{wn} {dist}",
            )
    return output


def _run_profile(strategy: str):
    """Per-stage breakdown via bench_scatter's built-in ScatterTimings.

    NCU is blocked on cloud A100 (binary SIGSEGVs under instrumentation).
    nsys traces but consistently fails at report-gen with "No GPU associated
    to the given UUID" (cloud GPU virtualization quirk; both 2023.4.4 and
    2024.1.1 nsys versions fail). Locally NCU is blocked by WSL2 and nsys
    produces no CUDA trace data.

    So we rely on cudaEvent-based per-stage timers + effective-bandwidth
    metric. Cross-validate on local 4080 (sm_89) and Modal A100 (sm_80) for
    direction-of-effect agreement.
    """
    import subprocess
    output = _gpu_info(1)
    ok, msg = _build()
    output += msg
    if not ok:
        return output

    strategies = [strategy] if strategy != "all" else STRATEGIES
    for dist in ["uniform", "zipf"]:
        for s in strategies:
            output += _run_cmd(
                ["./build/bin/bench_scatter",
                 f"--prefix=syn_{dist}_T2048", "--E=64", f"--strategy={s}",
                 "--iters=200", "--warmup=20"],
                f"breakdown {dist} {s}",
            )
    return output


def _run_blocksparse(strategy: str):
    """Block-sparse / Megablocks-style microbench: pre-packed cuBLAS vs
    fused gather+W1 kernel that skips materializing bufs.packed.

    Tests prefill T=2048 and decode T=16/64/256 shapes. The fused kernel
    here is naive (no tensor cores) — it proves correctness and quantifies
    the memory-pass saving; production would use CUTLASS for tensor-core
    throughput.
    """
    output = _gpu_info(1)
    ok, msg = _build()
    output += msg
    if not ok:
        return output

    for dist in ("uniform", "zipf"):
        for T in (16, 64, 256, 2048):
            output += _run_cmd(
                ["./build/bin/bench_blocksparse", dist, str(T)],
                f"blocksparse T={T} {dist}",
            )
    return output


# ── Per-GPU-count functions ───────────────────────────────────────────────────

@app.function(image=image, gpu="A100", timeout=900)
def bench_1gpu(strategy: str = "all", mode: str = "ep"):
    if mode == "prefill":
        return _run_prefill(1, strategy)
    if mode == "profile":
        return _run_profile(strategy)
    if mode == "grouped_gemm":
        return _run_grouped_gemm(strategy)
    if mode == "decode":
        return _run_decode(1, strategy)
    if mode == "blocksparse":
        return _run_blocksparse(strategy)
    return _run_ep(1, strategy)


@app.function(image=image, gpu="A100:2", timeout=900)
def bench_2gpu(strategy: str = "all", mode: str = "ep"):
    if mode == "prefill":
        return _run_prefill(2, strategy)
    if mode == "decode":
        return _run_decode(2, strategy)
    return _run_ep(2, strategy)


@app.function(image=image, gpu="A100:4", timeout=1200)
def bench_4gpu(strategy: str = "all", mode: str = "ep"):
    if mode == "prefill":
        return _run_prefill(4, strategy)
    if mode == "decode":
        return _run_decode(4, strategy)
    return _run_ep(4, strategy)


@app.function(image=image, gpu="A100:8", timeout=1800)
def bench_8gpu(strategy: str = "all", mode: str = "ep"):
    if mode == "prefill":
        return _run_prefill(8, strategy)
    if mode == "decode":
        return _run_decode(8, strategy)
    return _run_ep(8, strategy)


@app.function(image=image_torch, gpu="A100", timeout=1200)
def bench_real_1gpu(strategy: str = "all"):
    return _run_real(1, strategy)


@app.function(image=image_torch, gpu="A100:2", timeout=1200)
def bench_real_2gpu(strategy: str = "all"):
    return _run_real(2, strategy)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(n_gpus: int = 2, strategy: str = "all", mode: str = "ep"):
    if mode == "real":
        fn_map = {1: bench_real_1gpu, 2: bench_real_2gpu}
        if n_gpus not in fn_map:
            print(f"real mode supports 1 or 2 GPUs (got {n_gpus})")
            sys.exit(1)
        print(f"Launching real trace extraction + bench on {n_gpus}x A100 ...")
        output = fn_map[n_gpus].remote(strategy)
    else:
        fn_map = {1: bench_1gpu, 2: bench_2gpu, 4: bench_4gpu, 8: bench_8gpu}
        if n_gpus not in fn_map:
            print(f"n_gpus must be 1, 2, 4, or 8 (got {n_gpus})")
            sys.exit(1)
        print(f"Launching {mode} on {n_gpus}x A100 (strategy={strategy}) ...")
        output = fn_map[n_gpus].remote(strategy, mode)

    print(output)
