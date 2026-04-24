"""Run MoE packing benchmarks on Modal cloud GPUs (A100).

Mounts the local working tree — no git push required to iterate.

Usage:
    modal run modal_bench.py                  # default: 2 GPUs, all strategies
    modal run modal_bench.py --n-gpus 1       # single-GPU baseline
    modal run modal_bench.py --n-gpus 4       # 4 GPUs
    modal run modal_bench.py --n-gpus 8       # 8 GPUs
    modal run modal_bench.py --strategy warp  # pick strategy

GPU mapping:
    1 GPU  -> verify + bench_scatter + bench_e2e (single-GPU baseline)
    2 GPU  -> bench_ep (T=2048, all N ranks)
    4 GPU  -> bench_ep (T=4096, all N ranks)
    8 GPU  -> bench_ep (T=8192, all N ranks)

T scales with GPU count so all-to-all chunks stay ~32 MB per rank pair,
keeping the benchmark bandwidth-bound (not latency-bound).
"""

import sys

import modal

app = modal.App("moe-scatter-bench")

# ── Image: CUDA 12.4 devel + NCCL + CUTLASS ──────────────────────────────────

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git")
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
GPU_TO_PREFIX = {
    1: ["syn_uniform_T2048", "syn_zipf_T2048"],
    2: ["syn_uniform_T2048", "syn_zipf_T2048",
        "real_T2048"],
    4: ["syn_uniform_T4096", "syn_zipf_T4096",
        "real_T4096"],
    8: ["syn_uniform_T8192", "syn_zipf_T8192",
        "real_T8192"],
}

STRATEGIES = ["atomic", "sort", "warp"]


def _run_cmd(cmd, label=""):
    import subprocess
    try:
        r = subprocess.run(
            cmd, cwd="/workspace", capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        return f"=== {label} ===\nTIMEOUT after 300s\n"
    out = f"=== {label} ===\n{r.stdout}"
    if r.stderr.strip():
        out += f"STDERR: {r.stderr}\n"
    if r.returncode != 0:
        out += f"[exit code {r.returncode}]\n"
    return out + "\n"


def _run(n_gpus: int, strategy: str):
    import subprocess

    # ── GPU info ──────────────────────────────────────────────────────────
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    output = f"GPUs ({n_gpus} requested):\n{r.stdout}\n"

    # ── Build ─────────────────────────────────────────────────────────────
    r = subprocess.run(MAKE_CMD, cwd="/workspace", capture_output=True, text=True)
    if r.returncode != 0:
        return output + f"BUILD FAILED:\n{r.stdout}\n{r.stderr}\n"
    output += "Build OK\n\n"

    strategies = [strategy] if strategy != "all" else STRATEGIES
    prefixes = GPU_TO_PREFIX.get(n_gpus, GPU_TO_PREFIX[2])

    if n_gpus == 1:
        # ── Single GPU: verify + bench_scatter + bench_e2e ────────────────
        for s in strategies:
            output += _run_cmd(
                ["./build/bin/verify", "syn_uniform_T2048", "64", "results", s],
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
                    ["./build/bin/bench_e2e",
                     prefix, "64", "results", s],
                    f"bench_e2e {prefix} {s}",
                )
    else:
        # ── Multi-GPU: bench_ep with N ranks ──────────────────────────────
        for prefix in prefixes:
            for s in strategies:
                output += _run_cmd(
                    ["./build/bin/bench_ep",
                     f"--prefix={prefix}", f"--strategy={s}",
                     f"--n-gpus={n_gpus}"],
                    f"bench_ep {prefix} {s} ({n_gpus} GPUs)",
                )

    return output


# ── Per-GPU-count functions (Modal needs static gpu= at decoration time) ─────

@app.function(image=image, gpu="A100", timeout=900)
def bench_1gpu(strategy: str = "all"):
    return _run(1, strategy)


@app.function(image=image, gpu="A100:2", timeout=900)
def bench_2gpu(strategy: str = "all"):
    return _run(2, strategy)


@app.function(image=image, gpu="A100:4", timeout=900)
def bench_4gpu(strategy: str = "all"):
    return _run(4, strategy)


@app.function(image=image, gpu="A100:8", timeout=1200)
def bench_8gpu(strategy: str = "all"):
    return _run(8, strategy)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(n_gpus: int = 2, strategy: str = "all"):
    fn_map = {1: bench_1gpu, 2: bench_2gpu, 4: bench_4gpu, 8: bench_8gpu}
    if n_gpus not in fn_map:
        print(f"n_gpus must be 1, 2, 4, or 8 (got {n_gpus})")
        sys.exit(1)

    print(f"Launching on {n_gpus}x A100 (strategy={strategy}) ...")
    output = fn_map[n_gpus].remote(strategy)
    print(output)
