"""
Generate synthetic Top-K expert routing data for MoE packing benchmarks.

Supports two distributions:
  - uniform  : every expert receives roughly equal traffic
  - zipf     : hot expert receives alpha more traffic,
               simulate models' real-world long-tail linguistic routing imbalance.

Outputs:
  results/syn_{dist}_embeddings.npy   float16  [T, d]
  results/syn_{dist}_assignments.npy  int32    [T, K]
  results/syn_{dist}_weights.npy      float16  [T, K]
"""

import argparse
import os
import numpy as np
from typing import Literal

def gen_uniform_routing(
    T: int,
    K: int,
    E: int,
    d: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Each token independently samples K distinct experts from [0, E) with
    equal probability.  Router weights are softmax-normalized uniform logits
    so they sum to 1 per token. Used for measuring theoretical upper bound.

    Parameters
    ----------
    T : int   Number of tokens in the batch
    K : int   Top-K experts per token
    E : int   Total number of experts
    d : int   Token embedding dimension
    seed : int  RNG seed for reproducibility

    Returns
    -------
    embeddings   float16  [T, d]   Random unit-norm token embeddings
    assignments  int32    [T, K]   Expert IDs for ith token, no duplicates
                                   (will be packed to [T*K, d] for expert FFN)
    weights      float16  [T, K]   Gating weights
    """
    assert K <= E, f"K={K} must be <= E={E}"
    rng = np.random.default_rng(seed)

    # Token embeddings: random unit-norm vectors
    embeddings = rng.standard_normal((T, d)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    embeddings = (embeddings / norms).astype(np.float16)

    # Expert assignments: sample K unique experts per token
    noise = rng.random((T, E))           # uniform [0, 1)
    assignments = np.argsort(noise, axis=1)[:, :K].astype(np.int32)

    # Router weights: softmax over K uniform logits
    logits = rng.standard_normal((T, K)).astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)   # numerical stability
    exp_logits = np.exp(logits)
    weights = (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float16)

    return embeddings, assignments, weights


def gen_zipf_routing(
    T: int,
    K: int,
    E: int,
    d: int,
    alpha: float = 1.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expert e has unnormalized selection probability ∝ 1/(e+1)^alpha according to zipf's law.
    Popular experts handles a disproportionate fraction of tokens.
    
    Zipf-Mandelbrot law (from wiki):
    frequency ∝ 1/(rank+b)^alpha. We set b=0 for simplicity, rank=e+1.

    K experts are drawn without replacement per token.
    Use Gumbel-max trick: expert_id = argsort(log(p_e) + Gumbel(0,1))[-K:]

    Parameters
    ----------
    T     : int    Number of tokens
    K     : int    Top-K experts per token, number of experts for each token
    E     : int    Total number of experts
    d     : int    Token embedding dimension
    alpha : float  Zipf exponent
    seed  : int    RNG seed

    Returns
    -------
    embeddings   float16  [T, d]
    assignments  int32    [T, K]   Zipf-biased expert IDs
    weights      float16  [T, K]
    """
    assert K <= E, f"K={K} must be <= E={E}"
    rng = np.random.default_rng(seed)

    # Token embeddings (same as uniform)
    embeddings = rng.standard_normal((T, d)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    embeddings = (embeddings / norms).astype(np.float16)

    # Zipf base probabilities (unnormalized log-probs for Gumbel-max)
    ranks = np.arange(1, E + 1, dtype=np.float64)
    log_probs = -alpha * np.log(ranks)        # log(1/rank^alpha), shape [E]

    # Gumbel-max trick: add i.i.d. Gumbel noise then argsort
    gumbel_noise = -np.log(-np.log(rng.random((T, E)).clip(1e-15)))  # [T, E]
    scores = log_probs[None, :] + gumbel_noise                        # [T, E]
    assignments = np.argsort(scores, axis=1)[:, -K:][:, ::-1]        # top-K, descending score
    assignments = assignments.astype(np.int32)

    # Router weights: softmax, same structure as uniform
    logits = rng.standard_normal((T, K)).astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    weights = (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float16)

    return embeddings, assignments, weights


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def save_routing(
    out_dir: str,
    prefix: str,
    embeddings: np.ndarray,
    assignments: np.ndarray,
    weights: np.ndarray,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{prefix}_embeddings.npy"), embeddings)
    np.save(os.path.join(out_dir, f"{prefix}_assignments.npy"), assignments)
    np.save(os.path.join(out_dir, f"{prefix}_weights.npy"), weights)
    print(f"[save] {out_dir}/{prefix}_*.npy  "
          f"(embeddings {embeddings.shape}, assignments {assignments.shape})")


# ─────────────────────────────────────────────────────────────────────────────
# Local tests and verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_routing(
    label: str,
    embeddings: np.ndarray,
    assignments: np.ndarray,
    weights: np.ndarray,
    T: int,
    K: int,
    E: int,
    d: int,
) -> bool:
    """
    Run a suite of assertions on the generated arrays and print a summary.
    Returns True if all checks pass.
    """
    errors: list[str] = []

    # ── Shape checks ──────────────────────────────────────────────────────────
    if embeddings.shape != (T, d):
        errors.append(f"embeddings shape {embeddings.shape} != ({T},{d})")
    if assignments.shape != (T, K):
        errors.append(f"assignments shape {assignments.shape} != ({T},{K})")
    if weights.shape != (T, K):
        errors.append(f"weights shape {weights.shape} != ({T},{K})")

    # ── Dtype checks ─────────────────────────────────────────────────────────
    if embeddings.dtype != np.float16:
        errors.append(f"embeddings dtype {embeddings.dtype} != float16")
    if assignments.dtype != np.int32:
        errors.append(f"assignments dtype {assignments.dtype} != int32")
    if weights.dtype != np.float16:
        errors.append(f"weights dtype {weights.dtype} != float16")

    # ── Range checks ─────────────────────────────────────────────────────────
    if assignments.min() < 0 or assignments.max() >= E:
        errors.append(f"assignments out of [0,{E}): min={assignments.min()} max={assignments.max()}")

    # ── No duplicate experts per token ────────────────────────────────────────
    for i in range(min(T, 1000)):   # check first 1000 rows (fast)
        row = assignments[i]
        if len(set(row)) != K:
            errors.append(f"token {i} has duplicate experts: {row}")
            break

    # ── Weights sum ≈ 1 ───────────────────────────────────────────────────────
    w_f32 = weights.astype(np.float32)
    row_sums = w_f32.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-2):
        errors.append(f"weights don't sum to 1: min={row_sums.min():.4f} max={row_sums.max():.4f}")

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Verification: [{label}]")
    print(f"{'='*60}")
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        print(f"  → {len(errors)} error(s) found")
        return False

    # Summary statistics
    expert_counts = np.bincount(assignments.ravel(), minlength=E)
    print(f"  ✓ shapes   embeddings={embeddings.shape}  assignments={assignments.shape}  weights={weights.shape}")
    print(f"  ✓ dtypes   float16 / int32 / float16")
    print(f"  ✓ assignments in [0, {E})  (no duplicates per token)")
    print(f"  ✓ weights row-sum: mean={row_sums.mean():.4f}  std={row_sums.std():.6f}")
    print(f"  ✓ expert load distribution:")
    print(f"      min_tokens={expert_counts.min()}  max_tokens={expert_counts.max()}"
          f"  mean={expert_counts.mean():.1f}  std={expert_counts.std():.1f}")
    print(f"      top-3 experts by load: {np.argsort(expert_counts)[-3:][::-1].tolist()}"
          f"  counts: {np.sort(expert_counts)[-3:][::-1].tolist()}")
    imbalance = expert_counts.max() / expert_counts.mean()
    print(f"      imbalance ratio (max/mean): {imbalance:.2f}x")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Sweep: three input sizes for benchmarking across workload scales
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_SIZES = {
    "small":  512,     # short prompt, single-request prefill
    "medium": 2048,    # typical context window
    "large":  8192,    # long-context prefill, where packing fraction dominates
}


if __name__ == "__main__":
    # usage: python data/gen_synthetic_routing.py --tokens 2048 --experts 64 --topk 4 --dim 2048
    # usage: python data/gen_synthetic_routing.py --dist zipf
    # usage: python data/gen_synthetic_routing.py --sweep          # generates all 3 sizes x 2 dists
    parser = argparse.ArgumentParser(description="Generate synthetic MoE routing data")
    parser.add_argument("--tokens",  type=int,   default=2048,  help="Number of tokens T")
    parser.add_argument("--experts", type=int,   default=64,    help="Number of experts E")
    parser.add_argument("--topk",    type=int,   default=4,     help="Top-K per token")
    parser.add_argument("--dim",     type=int,   default=2048,  help="Embedding dimension d")
    parser.add_argument("--alpha",   type=float, default=1.2,   help="Zipf exponent")
    parser.add_argument("--seed",    type=int,   default=42,    help="RNG seed")
    parser.add_argument("--dist",    type=str,   default="both",
                        choices=["uniform", "zipf", "both"],
                        help="Which distribution(s) to generate")
    parser.add_argument("--sweep",   action="store_true",
                        help="Generate small/medium/large sizes (ignores --tokens)")
    parser.add_argument("--out",     type=str,   default="results",
                        help="Output directory")
    args = parser.parse_args()

    K, E, d = args.topk, args.experts, args.dim

    # Build list of (T, label) pairs to generate
    if args.sweep:
        runs = [(t, label) for label, t in SWEEP_SIZES.items()]
    else:
        runs = [(args.tokens, None)]

    ok = True

    for T, size_label in runs:
        # File prefix: "syn_uniform" for single-run, "syn_uniform_T512" for sweep
        tag = f"_T{T}" if size_label else ""

        print(f"\n{'─'*60}")
        print(f"Config: T={T}  K={K}  E={E}  d={d}  alpha={args.alpha}  seed={args.seed}"
              + (f"  [{size_label}]" if size_label else ""))
        print(f"{'─'*60}")

        if args.dist in ("uniform", "both"):
            emb, asgn, w = gen_uniform_routing(T, K, E, d, seed=args.seed)
            ok &= verify_routing(f"Uniform T={T}", emb, asgn, w, T, K, E, d)
            save_routing(args.out, f"syn_uniform{tag}", emb, asgn, w)

        if args.dist in ("zipf", "both"):
            emb, asgn, w = gen_zipf_routing(T, K, E, d, alpha=args.alpha, seed=args.seed)
            ok &= verify_routing(f"Zipf T={T}", emb, asgn, w, T, K, E, d)
            save_routing(args.out, f"syn_zipf{tag}", emb, asgn, w)

    print(f"\n{'='*60}")
    print("  All checks passed ✓" if ok else "  SOME CHECKS FAILED ✗")
    print(f"{'='*60}\n")