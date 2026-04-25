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

STRATEGIES = ["atomic", "sort", "warp", "vec"]


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

    Runs each (dist, strategy) twice: once serial (--overlap=0) and once
    with comm/compute overlap (--overlap=1) so we can A/B compare wall
    time and confirm the L2 checksum still matches.
    """
    output = _gpu_info(n_gpus)
    ok, msg = _build()
    output += msg
    if not ok:
        return output

    strategies = [strategy] if strategy != "all" else STRATEGIES
    T = GPU_TO_T.get(n_gpus, 2048)

    for dist in ["uniform", "zipf"]:
        prefix = f"syn_{dist}_T{T}_N32"
        for s in strategies:
            for ov in (0, 1):
                tag = "overlap" if ov else "serial"
                output += _run_cmd(
                    ["./build/bin/bench_prefill",
                     f"--prefix={prefix}", f"--strategy={s}",
                     f"--n-gpus={n_gpus}", f"--overlap={ov}"],
                    f"bench_prefill {prefix} {s} ({n_gpus} GPUs, {tag})",
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


def _run_profile(strategy: str):
    """NCU profile of bench_scatter for the chosen strategy on uniform + zipf.

    Captures key metrics: SM/memory throughput, occupancy, stall reasons.
    Prints a per-kernel summary so we can see where the time goes.
    """
    import subprocess
    output = _gpu_info(1)
    ok, msg = _build()
    output += msg
    if not ok:
        return output

    # Locate ncu (apt installs to /opt/nvidia/nsight-compute/<ver>/ncu)
    ncu_paths = [
        "/usr/local/cuda/bin/ncu",
        "/opt/nvidia/nsight-compute/2024.1.1/ncu",
        "/usr/bin/ncu",
        "ncu",
    ]
    ncu = None
    for path in ncu_paths:
        try:
            r = subprocess.run([path, "--version"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                ncu = path
                output += f"using ncu at: {path}\n{r.stdout.splitlines()[0]}\n\n"
                break
        except FileNotFoundError:
            continue
    if ncu is None:
        # Try shell-search
        r = subprocess.run(
            ["bash", "-lc", "find /opt /usr -name ncu -type f 2>/dev/null | head -5"],
            capture_output=True, text=True,
        )
        output += f"ncu not found. find result:\n{r.stdout}\n"
        return output

    # Diagnostics
    output += "\n--- diagnostics ---\n"
    for cmd in [
        "id",
        "cat /proc/driver/nvidia/params 2>&1 | grep -iE 'profil|admin' | head -5 || true",
        "nvidia-smi --query-gpu=persistence_mode,compute_mode --format=csv",
        "find /opt /usr -name 'ncu' -o -name 'nsys' 2>/dev/null | head -10",
    ]:
        r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
        output += f"$ {cmd}\n{r.stdout}{r.stderr}\n"

    # Locate nsys
    r = subprocess.run(["bash", "-lc", "which nsys || find /opt /usr -name nsys -type f 2>/dev/null | head -1"],
                       capture_output=True, text=True)
    nsys = r.stdout.strip() or None
    output += f"nsys found at: {nsys}\n\n"

    # Per-stage breakdown via bench_scatter's built-in ScatterTimings reporting
    output += "\n--- per-stage breakdown (no profiler) ---\n"
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


# ── Per-GPU-count functions ───────────────────────────────────────────────────

@app.function(image=image, gpu="A100", timeout=900)
def bench_1gpu(strategy: str = "all", mode: str = "ep"):
    if mode == "prefill":
        return _run_prefill(1, strategy)
    if mode == "profile":
        return _run_profile(strategy)
    return _run_ep(1, strategy)


@app.function(image=image, gpu="A100:2", timeout=900)
def bench_2gpu(strategy: str = "all", mode: str = "ep"):
    if mode == "prefill":
        return _run_prefill(2, strategy)
    return _run_ep(2, strategy)


@app.function(image=image, gpu="A100:4", timeout=1200)
def bench_4gpu(strategy: str = "all", mode: str = "ep"):
    if mode == "prefill":
        return _run_prefill(4, strategy)
    return _run_ep(4, strategy)


@app.function(image=image, gpu="A100:8", timeout=1800)
def bench_8gpu(strategy: str = "all", mode: str = "ep"):
    if mode == "prefill":
        return _run_prefill(8, strategy)
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
