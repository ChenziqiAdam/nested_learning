#!/usr/bin/env python
"""
RQ1 Analysis: Do different CMS layers affect different tokens differently?

Trains a CMS baseline (if no checkpoint) then runs three experiments:
  A) Per-layer representation delta
  B) Layer ablation per-token loss impact
  C) Gradient-based analysis

Usage:
  # Full run (train + analyze):
  python scripts/rq1_analysis.py --config configs/rq1_pilot.yaml --output-dir results/rq1

  # Analysis only (existing checkpoint):
  python scripts/rq1_analysis.py --config configs/rq1_pilot.yaml \
      --checkpoint artifacts/rq1_checkpoint/step_005000.pt \
      --output-dir results/rq1 --skip-training

  # Quick smoke test (CPU, random init):
  python scripts/rq1_analysis.py --config configs/rq1_pilot.yaml \
      --output-dir results/rq1_smoke --skip-training --num-batches 2 \
      --seq-len 64 --batch-size 2 --device cpu
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.nested_learning.training import unwrap_config, build_model_from_cfg, run_training_loop
from src.nested_learning.data import SyntheticTextConfig, SyntheticTextDataset, collate_batch
from src.nested_learning.model import HOPEModel


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output-dir", default="results/rq1")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--num-batches", type=int, default=50)
    p.add_argument("--num-grad-batches", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(cfg_path: str, device: str) -> Path:
    cfg = OmegaConf.load(cfg_path)
    cfg = unwrap_config(cfg)
    OmegaConf.update(cfg, "train.device", device, merge=True)
    dev = torch.device(device)
    print(f"[train] Starting {cfg.train.steps} steps on {device}...")
    run_training_loop(cfg, device=dev)
    ckpt_dir = Path(cfg.train.checkpoint.dir)
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")
    latest = checkpoints[-1]
    print(f"[train] Done. Latest checkpoint: {latest}")
    return latest


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg_path: str, checkpoint_path: Path | None, device: torch.device) -> HOPEModel:
    cfg = OmegaConf.load(cfg_path)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    if checkpoint_path is not None and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = state.get("model", state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[warn] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[warn] Unexpected keys: {len(unexpected)}")
        print(f"[info] Loaded checkpoint: {checkpoint_path}")
    else:
        print("[warn] No checkpoint — using random init")
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Data iterator
# ---------------------------------------------------------------------------

def make_data_iter(
    vocab_size: int,
    num_batches: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    seed: int = 42,
):
    cfg_d = SyntheticTextConfig(
        vocab_size=vocab_size,
        seq_len=seq_len,
        dataset_size=num_batches * batch_size + batch_size,
    )
    dataset = SyntheticTextDataset(cfg_d)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_batch, generator=g,
    )
    for i, tokens in enumerate(loader):
        if i >= num_batches:
            break
        yield tokens.to(device)


# ---------------------------------------------------------------------------
# CMSInterceptor — Task 4
# ---------------------------------------------------------------------------

class CMSInterceptor:
    """Forward hooks on every CMSBlock to capture (inp, out) per level per layer."""

    def __init__(self, model: HOPEModel):
        self.model = model
        self._hooks: list = []
        self.intercepted: dict[int, dict[str, list[tuple[torch.Tensor, torch.Tensor]]]] = {}

    def __enter__(self):
        for layer_idx, block in enumerate(self.model.blocks):
            cms = getattr(block, "cms", None)
            if cms is None:
                continue
            self.intercepted[layer_idx] = {}
            for level_name, cms_block in cms.blocks.items():
                self.intercepted[layer_idx][level_name] = []

                def _make_hook(li: int, ln: str):
                    def _hook(module, inp, out):
                        self.intercepted[li][ln].append(
                            (inp[0].detach().cpu(), out.detach().cpu())
                        )
                    return _hook

                h = cms_block.register_forward_hook(_make_hook(layer_idx, level_name))
                self._hooks.append(h)
        return self

    def __exit__(self, *_):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        for d in self.intercepted.values():
            for v in d.values():
                v.clear()

    def get_level_names(self) -> list[str]:
        for d in self.intercepted.values():
            return list(d.keys())
        return []


# ---------------------------------------------------------------------------
# Experiment A — Task 5
# ---------------------------------------------------------------------------

def run_experiment_a(
    model: HOPEModel,
    vocab_size: int,
    num_batches: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Per-token relative contribution ρ_t^(s) = Δ_t^(s) / Σ_j Δ_t^(j)
    where Δ_t^(s) = ||h_out - h_in||_2 for each CMS level s.
    """
    accum: dict[str, list[np.ndarray]] = {}

    with CMSInterceptor(model) as ic:
        for tokens in make_data_iter(vocab_size, num_batches, seq_len, batch_size, device, seed):
            ic.clear()
            with torch.no_grad():
                model(tokens)

            for level_dict in ic.intercepted.values():
                deltas: dict[str, torch.Tensor] = {}
                for level_name, calls in level_dict.items():
                    if not calls:
                        continue
                    inp_t, out_t = calls[-1]
                    delta_norm = (out_t - inp_t).norm(dim=-1).reshape(-1)  # [B*T]
                    deltas[level_name] = delta_norm

                if len(deltas) < 2:
                    continue

                total = sum(deltas.values()).clamp(min=1e-8)
                for name, d in deltas.items():
                    rho = (d / total).numpy()
                    accum.setdefault(name, []).append(rho)

    return {name: np.concatenate(vs) for name, vs in accum.items()}


def summarize_experiment_a(rho_per_level: dict[str, np.ndarray]) -> dict:
    return {
        name: {"mean": float(arr.mean()), "variance": float(arr.var()), "n": int(len(arr))}
        for name, arr in rho_per_level.items()
    }


def plot_experiment_a(rho_per_level: dict[str, np.ndarray], output_dir: Path) -> None:
    levels = list(rho_per_level.keys())

    fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 4), sharey=False)
    if len(levels) == 1:
        axes = [axes]
    for ax, name in zip(axes, levels):
        arr = rho_per_level[name]
        ax.hist(arr, bins=50, density=True, alpha=0.7, color="steelblue")
        ax.set_title(f"{name}\nvar={arr.var():.4f}")
        ax.set_xlabel("Relative contribution ρ")
        ax.set_ylabel("Density")
    fig.suptitle("Exp A: CMS Level Relative Contribution Distribution")
    plt.tight_layout()
    fig.savefig(output_dir / "rq1_a_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    means = [rho_per_level[n].mean() for n in levels]
    stds = [rho_per_level[n].std() for n in levels]
    x = np.arange(len(levels))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=20)
    ax.set_ylabel("Mean relative contribution ρ")
    ax.set_title("Exp A: Mean CMS Level Contribution ± Std")
    plt.tight_layout()
    fig.savefig(output_dir / "rq1_a_mean_contribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment B — Task 6
# ---------------------------------------------------------------------------

@contextmanager
def ablate_cms_level(model: HOPEModel, level_name: str):
    """Temporarily replace one CMS level with identity across all blocks."""
    patched: list[tuple] = []
    for block in model.blocks:
        cms = getattr(block, "cms", None)
        if cms is None or level_name not in cms.blocks:
            continue
        cms_block = cms.blocks[level_name]
        original_forward = cms_block.forward
        cms_block.forward = lambda x, _orig=original_forward: x
        patched.append((cms_block, original_forward))
    try:
        yield
    finally:
        for cms_block, orig in patched:
            cms_block.forward = orig


_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "and", "or", "but", "not", "if", "it", "he", "she", "they", "we",
    "i", "you", "this", "that", "these", "those", "its", "his", "her",
})
_PUNCT_RE = re.compile(r'^[^\w\s]+$')


def categorize_token(piece: str, freq_rank: int | None) -> str:
    clean = piece.lstrip("▁ ").strip().lower()
    if not clean:
        return "punctuation"
    if _PUNCT_RE.match(clean):
        return "punctuation"
    if clean in _FUNCTION_WORDS:
        return "function"
    if freq_rank is not None and freq_rank > 20000:
        return "rare"
    return "content"


def run_experiment_b(
    model: HOPEModel,
    cms_level_names: list[str],
    vocab_size: int,
    num_batches: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> list[dict]:
    """
    I_t^(s) = L_t^(\\s) - L_t^full  (positive = level was important for this token)
    """
    freq: dict[int, int] = {}
    for tokens in make_data_iter(vocab_size, num_batches, seq_len, batch_size, device, seed):
        for tid in tokens.cpu().numpy().reshape(-1).tolist():
            freq[tid] = freq.get(tid, 0) + 1
    freq_rank = {
        tid: rank
        for rank, (tid, _) in enumerate(sorted(freq.items(), key=lambda x: -x[1]))
    }

    records: list[dict] = []

    for tokens in make_data_iter(vocab_size, num_batches, seq_len, batch_size, device, seed):
        with torch.no_grad():
            logits_full = model(tokens)
        lp_full = F.log_softmax(logits_full[:, :-1, :], dim=-1)
        targets = tokens[:, 1:]
        nll_full = -lp_full.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        for level_name in cms_level_names:
            with ablate_cms_level(model, level_name):
                with torch.no_grad():
                    logits_abl = model(tokens)
            lp_abl = F.log_softmax(logits_abl[:, :-1, :], dim=-1)
            nll_abl = -lp_abl.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            importance = nll_abl - nll_full  # [B, T-1]

            token_ids = targets.cpu().numpy().reshape(-1).tolist()
            imp_vals = importance.cpu().numpy().reshape(-1).tolist()

            for tid, imp in zip(token_ids, imp_vals):
                rank = freq_rank.get(int(tid), None)
                cat = categorize_token(f"id_{tid}", rank)
                records.append({
                    "token_id": int(tid),
                    "category": cat,
                    "level": level_name,
                    "importance": float(imp),
                })

    return records


def plot_experiment_b(records: list[dict], output_dir: Path) -> None:
    data: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for r in records:
        data[r["category"]][r["level"]].append(r["importance"])

    categories = sorted(data.keys())
    levels = sorted({r["level"] for r in records})

    fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 5), sharey=True)
    if len(categories) == 1:
        axes = [axes]
    for ax, cat in zip(axes, categories):
        vals_per_level = [data[cat][lv] for lv in levels]
        ax.boxplot(vals_per_level, tick_labels=levels, showfliers=False)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"{cat}")
        ax.set_xlabel("CMS Level")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Importance ΔL (higher = more important)")
    fig.suptitle("Exp B: CMS Layer Ablation Importance by Token Category")
    plt.tight_layout()
    fig.savefig(output_dir / "rq1_b_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment C — Task 7
# ---------------------------------------------------------------------------

def run_experiment_c(
    model: HOPEModel,
    cms_level_names: list[str],
    vocab_size: int,
    num_batches: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    n_tokens_sample: int = 32,
) -> dict[str, dict[str, list[float]]]:
    """
    Spearman correlation between per-token perplexity and per-token CMS gradient norm.
    """
    results: dict[str, dict[str, list]] = {
        name: {"perplexities": [], "grad_norms": []} for name in cms_level_names
    }

    for tokens in make_data_iter(vocab_size, num_batches, seq_len, batch_size, device, seed):
        B, T = tokens.shape

        with torch.no_grad():
            logits = model(tokens)
        lp = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = tokens[:, 1:]
        per_tok_nll = -lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        cms_inputs: dict[tuple[int, str], torch.Tensor] = {}
        tmp_hooks = []
        for layer_idx, block in enumerate(model.blocks):
            cms = getattr(block, "cms", None)
            if cms is None:
                continue
            for level_name, cms_block in cms.blocks.items():
                def _hook(module, inp, out, _li=layer_idx, _ln=level_name):
                    cms_inputs[(_li, _ln)] = inp[0].detach()
                tmp_hooks.append(cms_block.register_forward_hook(_hook))
        with torch.no_grad():
            model(tokens)
        for h in tmp_hooks:
            h.remove()

        t_indices = torch.randperm(T - 1)[:n_tokens_sample].tolist()

        level_grad_norms: dict[str, list[tuple[int, float]]] = {
            name: [] for name in cms_level_names
        }

        for (layer_idx, level_name), h_in in cms_inputs.items():
            block = model.blocks[layer_idx]
            cms_block = block.cms.blocks[level_name]
            params = list(cms_block.parameters())
            if not params:
                continue

            for t_idx in t_indices:
                h_t = h_in[:, t_idx : t_idx + 1, :].detach().clone().requires_grad_(False)
                with torch.enable_grad():
                    h_t_d = h_t.detach().clone()
                    out_t = cms_block(h_t_d)
                    loss_t = (out_t - h_t_d).pow(2).mean()
                    grads = torch.autograd.grad(loss_t, params, allow_unused=True)
                gn = sum(g.norm().item() ** 2 for g in grads if g is not None) ** 0.5
                level_grad_norms[level_name].append((t_idx, gn))

        for level_name in cms_level_names:
            for t_idx, gn in level_grad_norms[level_name]:
                ppl = per_tok_nll[:, t_idx].mean().item()
                results[level_name]["perplexities"].append(ppl)
                results[level_name]["grad_norms"].append(gn)

    return results


def plot_experiment_c(correlations: dict[str, dict], output_dir: Path) -> None:
    from scipy.stats import spearmanr

    levels = list(correlations.keys())
    fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 4))
    if len(levels) == 1:
        axes = [axes]
    for ax, name in zip(axes, levels):
        ppls = np.array(correlations[name]["perplexities"])
        gns = np.array(correlations[name]["grad_norms"])
        ax.scatter(ppls, gns, alpha=0.3, s=10, rasterized=True, color="steelblue")
        if len(ppls) > 2:
            rho, pval = spearmanr(ppls, gns)
        else:
            rho, pval = float("nan"), float("nan")
        ax.set_title(f"{name}\nSpearman r={rho:.3f}")
        ax.set_xlabel("Per-token NLL")
        ax.set_ylabel("Gradient norm")
    fig.suptitle("Exp C: Token Perplexity vs CMS Gradient Norm")
    plt.tight_layout()
    fig.savefig(output_dir / "rq1_c_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main — Task 8/9
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    checkpoint_path: Path | None = None
    if not args.skip_training:
        checkpoint_path = run_training(args.config, args.device)
    elif args.checkpoint:
        checkpoint_path = Path(args.checkpoint)

    model = load_model(args.config, checkpoint_path, device)

    cms_level_names: list[str] = []
    for block in model.blocks:
        cms = getattr(block, "cms", None)
        if cms is not None:
            cms_level_names = list(cms.blocks.keys())
            break
    if not cms_level_names:
        raise RuntimeError("No CMS levels found — is this a HOPEAttentionBlock model?")
    print(f"[info] CMS levels: {cms_level_names}")

    cfg = OmegaConf.load(args.config)
    cfg = unwrap_config(cfg)
    vocab_size = int(cfg.model.vocab_size)

    # Experiment A
    print("\n[Exp A] Per-layer representation delta...")
    rho_per_level = run_experiment_a(
        model, vocab_size, args.num_batches, args.seq_len, args.batch_size, device, args.seed
    )
    a_summary = summarize_experiment_a(rho_per_level)
    save_json(a_summary, output_dir / "rq1_a_summary.json")
    plot_experiment_a(rho_per_level, output_dir)
    print(f"[Exp A] Variance: { {k: round(v['variance'], 5) for k, v in a_summary.items()} }")

    # Experiment B
    print("\n[Exp B] Layer ablation per-token loss impact...")
    records = run_experiment_b(
        model, cms_level_names, vocab_size, args.num_batches, args.seq_len, args.batch_size, device, args.seed
    )
    b_agg: dict[str, list] = collections.defaultdict(list)
    for r in records:
        b_agg[f"{r['category']}__{r['level']}"].append(r["importance"])
    b_summary = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        for k, v in b_agg.items()
    }
    save_json(b_summary, output_dir / "rq1_b_importance.json")
    plot_experiment_b(records, output_dir)
    print("[Exp B] Done.")

    # Experiment C
    print(f"\n[Exp C] Gradient-based analysis ({args.num_grad_batches} batches)...")
    correlations = run_experiment_c(
        model, cms_level_names, vocab_size, args.num_grad_batches,
        args.seq_len, args.batch_size, device, args.seed,
    )
    from scipy.stats import spearmanr
    c_summary = {}
    for name, data in correlations.items():
        ppls = np.array(data["perplexities"])
        gns = np.array(data["grad_norms"])
        rho, pval = spearmanr(ppls, gns) if len(ppls) > 2 else (float("nan"), float("nan"))
        c_summary[name] = {
            "spearman_r": float(rho),
            "spearman_p": float(pval),
            "grad_norm_variance": float(np.var(gns)),
            "n_samples": len(ppls),
        }
    save_json(c_summary, output_dir / "rq1_c_gradient.json")
    plot_experiment_c(correlations, output_dir)
    print(f"[Exp C] Correlations: { {k: round(v['spearman_r'], 3) for k, v in c_summary.items()} }")

    print(f"\n[DONE] Results in {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
