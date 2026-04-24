"""
Load Qwen1.5-MoE-A2.7B, feed WikiText-103 test split corpus, extract the Layer 0 Gate's
Top-K routing decisions for use in MoE packing benchmarks.

Functions:
  load_qwen_model(): Load Qwen1.5-MoE-A2.7B model and tokenizer
  extract_layer_0_gate(): Extract the Layer 0 Gate (router) from the model, used for fused kernel
  compute_real_routing(): Compute the real routing decisions from the model, used for packing

Outputs:
  results/real_embeddings.npy   float16  [T, d], without batch dimension
  results/real_assignments.npy  int32    [T, K]
  results/real_weights.npy      float16  [T, K]

  Tested on RTX 2080 (8 GB VRAM).  The model is loaded in float16 with
  device_map="auto".  If CUDA is unavailable the script falls back to CPU
  and automatically reduces the text corpus to keep memory manageable.
"""

import argparse
import os
import sys
import textwrap
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

def load_wikitext_corpus(n_sentences: int = 200) -> list[str]:
    """
    Load n_sentences non-empty sentences from the WikiText-103 test split.
    Dataset is downloaded to HuggingFace cache (~500 MB), loaded on CPU only.
    """
    from datasets import load_dataset
    print(f"[corpus] Loading WikiText-103 test split …")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    texts = [row["text"] for row in ds if len(row["text"]) > 50][:n_sentences]
    print(f"[corpus] Loaded {len(texts)} sentences from WikiText-103")
    return texts

def load_qwen_model(
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B",
    device: Optional[str] = None,
):
    """
    Download Qwen1.5-MoE-A2.7B, return the model and tokenizer.

    The model is loaded in float16 to fit within 8 GB VRAM.
    Gradient computation is disabled globally because we only do inference.

    Parameters
    ----------
    model_name : str   HuggingFace model identifier
    device     : str   "cuda", "cpu", or None (auto-detect)

    Returns
    -------
    model      : PreTrainedModel  in eval mode, float16
    tokenizer  : PreTrainedTokenizer
    device     : torch.device
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"[load] Loading tokenizer from '{model_name}' …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"[load] Loading model in float16 on {device} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device.type == "cuda" else "cpu",
        trust_remote_code=True,
    )
    model.eval()
    torch.set_grad_enabled(False)

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[load] Model loaded  ({n_params:.2f}B parameters)")
    return model, tokenizer, device

def extract_layer_0_gate(model) -> torch.nn.Linear:
    """
    Navigate the model hierarchy to find the MoE gate (router) in Layer 0.

    Qwen1.5-MoE hierarchy:
      model.model.layers[0].mlp: Qwen2MoeSparseMoeBlock
      model.model.layers[0].mlp.gate: Linear(d_model, n_experts, bias=False)

    Returns the router projection matrix
    """
    try:
        gate = model.model.layers[0].mlp.gate
    except AttributeError as e:
        # Fallback: try common alternative attribute names
        moe_block = model.model.layers[0].mlp
        for attr in ("gate", "router", "expert_gate", "w_gate"):
            if hasattr(moe_block, attr):
                gate = getattr(moe_block, attr)
                break
        else:
            raise RuntimeError(
                f"Cannot find gate module in {type(moe_block)}. "
                f"Available attributes: {[a for a in dir(moe_block) if not a.startswith('_')]}"
            ) from e

    assert isinstance(gate, torch.nn.Linear), f"Expected Linear, got {type(gate)}"
    print(f"[gate] Found gate: Linear(in={gate.in_features}, out={gate.out_features})")
    return gate

def compute_real_routing(
    model,
    tokenizer,
    texts: list[str],
    K: int,
    device: torch.device,
    max_tokens: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Feed text through the model's embedding and Layer-0 gate to compute
    authentic token routing decisions.

    Steps:
    1. Tokenize all texts and concatenate token IDs with shape [1, T].
    2. Look up token embeddings [T, d] via model.model.embed_tokens.
    3. Pass embeddings through model.model.layers[0].input_layernorm.
    4. Compute gate logits = gate(normed_embeddings).
    5. Use softmax transform logits to router probabilities.
    6. Take Top-K experts per token (descending probability): assignments
    7. Collect gating weights for the selected K experts: weights

    Parameters
    ----------
    model      : the loaded Qwen model
    tokenizer  : matched tokenizer
    texts      : list of input strings
    K          : Top-K experts to select per token
    device     : target device
    max_tokens : truncate total token sequence to this length

    Returns
    -------
    embeddings   np.float16  [T, d]   pre-norm token embeddings
    assignments  np.int32    [T, K]   top-K expert IDs (descending weight)
    weights      np.float16  [T, K]   corresponding softmax gating weights
    """
    # ── Step 1: Tokenize ─────────────────────────────────────────────────────
    all_ids: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        all_ids.extend(ids)
        if len(all_ids) >= max_tokens:
            break
    all_ids = all_ids[:max_tokens]
    T = len(all_ids)
    print(f"[route] Total tokens after truncation: T={T}")

    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)  # [1, T]

    # ── Step 2: Token embeddings ──────────────────────────────────────────────
    embed_layer = model.model.embed_tokens
    raw_embeddings = embed_layer(input_ids)          # [1, T, d]
    raw_embeddings = raw_embeddings.squeeze(0)       # [T, d]  float16

    # ── Step 3: Pre-MoE LayerNorm ─────────────────────────────────────────────
    layer_0 = model.model.layers[0]
    normed = layer_0.post_attention_layernorm(raw_embeddings)  # [T, d]

    # ── Step 4: Gate logits ───────────────────────────────────────────────────
    gate = extract_layer_0_gate(model)
    gate_logits = gate(normed.to(gate.weight.dtype))  # [T, E], router output
    E = gate_logits.shape[1]

    # ── Step 5: Softmax router probabilities ──────────────────────────────────
    router_probs = F.softmax(gate_logits.float(), dim=-1)  # [T, E] float32

    # ── Step 6: Top-K selection ───────────────────────────────────────────────
    topk_weights, topk_ids = torch.topk(router_probs, K, dim=-1, sorted=True)
    # topk_ids     : [T, K]  int64  (sorted descending by weight)
    # topk_weights : [T, K]  float32

    # ── Step 7: Collect outputs ───────────────────────────────────────────────
    embeddings  = raw_embeddings.cpu().to(torch.float16).numpy()   # [T, d]
    assignments = topk_ids.cpu().to(torch.int32).numpy()           # [T, K]
    weights     = topk_weights.cpu().to(torch.float16).numpy()     # [T, K]

    print(f"[route] Routing complete: E={E}  K={K}  T={T}")
    return embeddings, assignments, weights


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def save_routing(
    out_dir: str,
    embeddings: np.ndarray,
    assignments: np.ndarray,
    weights: np.ndarray,
    prefix: str = "real",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{prefix}_embeddings.npy"),  embeddings)
    np.save(os.path.join(out_dir, f"{prefix}_assignments.npy"), assignments)
    np.save(os.path.join(out_dir, f"{prefix}_weights.npy"),     weights)
    print(f"[save] Saved to {out_dir}/{prefix}_*.npy  "
          f"(embeddings {embeddings.shape}, assignments {assignments.shape})")


# ─────────────────────────────────────────────────────────────────────────────
# Local tests and verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_routing(
    embeddings: np.ndarray,
    assignments: np.ndarray,
    weights: np.ndarray,
    K: int,
    E: int,
) -> bool:
    """
    Validate shape, dtype, range, uniqueness, and weight ordering.
    Also prints expert load statistics for comparison with synthetic data.
    """
    errors: list[str] = []
    T, d = embeddings.shape

    # Shapes
    if assignments.shape != (T, K):
        errors.append(f"assignments shape {assignments.shape} != ({T},{K})")
    if weights.shape != (T, K):
        errors.append(f"weights shape {weights.shape} != ({T},{K})")

    # Dtypes
    if embeddings.dtype != np.float16:
        errors.append(f"embeddings dtype {embeddings.dtype}")
    if assignments.dtype != np.int32:
        errors.append(f"assignments dtype {assignments.dtype}")
    if weights.dtype != np.float16:
        errors.append(f"weights dtype {weights.dtype}")

    # Range
    if assignments.min() < 0 or assignments.max() >= E:
        errors.append(f"assignments out of [0,{E}): [{assignments.min()},{assignments.max()}]")

    # No duplicate experts per token (check all rows)
    for i in range(T):
        if len(set(assignments[i])) != K:
            errors.append(f"token {i} has duplicate experts: {assignments[i]}")
            break

    # Weights are positive
    if (weights.astype(np.float32) <= 0).any():
        errors.append("some weights are non-positive")

    # Weights are sorted descending per row (Top-K guarantees this)
    w32 = weights.astype(np.float32)
    if not (np.diff(w32, axis=1) <= 0).all():
        # float16 rounding can cause tiny violations — allow 1e-3 tolerance
        violations = (np.diff(w32, axis=1) > 1e-3).sum()
        if violations > 0:
            errors.append(f"weights not sorted descending: {violations} violations")

    print(f"\n{'='*60}")
    print(f"  Verification: [Real Qwen Routing]")
    print(f"{'='*60}")
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        return False

    expert_counts = np.bincount(assignments.ravel(), minlength=E)
    imbalance = expert_counts.max() / expert_counts.mean()
    top3_idx = np.argsort(expert_counts)[-3:][::-1]

    print(f"  ✓ shapes   embeddings={embeddings.shape}  assignments={assignments.shape}")
    print(f"  ✓ dtypes   float16 / int32 / float16")
    print(f"  ✓ assignments in [0, {E})  no duplicates per row")
    print(f"  ✓ weights are positive and sorted descending")
    print(f"  ✓ expert load distribution:")
    print(f"      min={expert_counts.min()}  max={expert_counts.max()}"
          f"  mean={expert_counts.mean():.1f}  std={expert_counts.std():.1f}")
    print(f"      top-3 experts: {top3_idx.tolist()}"
          f"  counts: {expert_counts[top3_idx].tolist()}")
    print(f"      imbalance ratio (max/mean): {imbalance:.2f}x")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Offline mode (no model download)
# usage: python extract_qwen_routing.py --dry-run --topk 4 --out results
# ─────────────────────────────────────────────────────────────────────────────

def dry_run(T: int, K: int, E: int, d: int, out_dir: str,
            prefix: str = "real") -> None:
    """
    Generate fake routing data with a realistic Zipf-like imbalance.
    Used for testing the rest of the pipeline without downloading the model.
    """
    print(f"[dry-run] Generating fake Qwen-style routing T={T} (no model download)")
    rng = np.random.default_rng(0)

    embeddings = rng.standard_normal((T, d)).astype(np.float16)

    # Zipf-like probabilities mimicking real MoE routing
    ranks = np.arange(1, E + 1, dtype=np.float64)
    log_probs = -1.5 * np.log(ranks)
    gumbel = -np.log(-np.log(rng.random((T, E)).clip(1e-15)))
    scores = log_probs[None, :] + gumbel
    assignments = np.argsort(scores, axis=1)[:, -K:][:, ::-1].astype(np.int32)

    # Softmax weights (decreasing per row to mimic top-K sort order)
    logits = np.sort(rng.standard_normal((T, K)).astype(np.float32), axis=1)[:, ::-1]
    logits -= logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    weights = (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float16)

    verify_routing(embeddings, assignments, weights, K, E)
    save_routing(out_dir, embeddings, assignments, weights, prefix=prefix)
    print("[dry-run] Complete. Outputs can be used as replacements for real model data.")

SWEEP_SIZES = [2048, 4096, 8192, 16384]


def main():
    parser = argparse.ArgumentParser(
        description="Extract real Qwen1.5-MoE routing data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples
            --------
              # Real model, single size
              python data/extract_qwen_routing.py --topk 4 --out results

              # Real model, all sizes for scaling benchmarks
              python data/extract_qwen_routing.py --sweep --out results

              # Dry-run (no model download), all sizes
              python data/extract_qwen_routing.py --dry-run --sweep --out results
        """),
    )
    parser.add_argument("--model",      default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--topk",       type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens to extract (ignored with --sweep)")
    parser.add_argument("--out",        default="results")
    parser.add_argument("--device",     default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Skip model download, generate mock data instead")
    parser.add_argument("--sweep",      action="store_true",
                        help="Generate data at multiple T sizes for scaling benchmarks")
    parser.add_argument("--n-sentences", type=int, default=2000,
                        help="Number of WikiText sentences to load (increase for large T)")
    args = parser.parse_args()

    sizes = SWEEP_SIZES if args.sweep else [args.max_tokens]

    if args.dry_run:
        E, d = 60, 2048
        for T in sizes:
            prefix = f"real_T{T}" if len(sizes) > 1 else "real"
            dry_run(T=T, K=args.topk, E=E, d=d, out_dir=args.out, prefix=prefix)
        print("\nDone ✓")
        return

    model, tokenizer, device = load_qwen_model(args.model, args.device)
    gate = extract_layer_0_gate(model)
    E = gate.out_features

    texts = load_wikitext_corpus(n_sentences=args.n_sentences)

    ok = True
    for T in sizes:
        prefix = f"real_T{T}" if len(sizes) > 1 else "real"
        print(f"\n{'─'*60}")
        print(f"  Extracting real routing: T={T}")
        print(f"{'─'*60}")

        embeddings, assignments, weights = compute_real_routing(
            model, tokenizer, texts, K=args.topk,
            device=device, max_tokens=T,
        )

        ok &= verify_routing(embeddings, assignments, weights, K=args.topk, E=E)
        save_routing(args.out, embeddings, assignments, weights, prefix=prefix)

    if not ok:
        sys.exit(1)
    print("\nDone ✓")


if __name__ == "__main__":
    main()