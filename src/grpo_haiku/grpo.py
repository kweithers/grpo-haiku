"""Memory-safe GRPO (Group Relative Policy Optimization) trainer in Apple MLX.

Engineered for 16GB Unified Memory (Apple Silicon M5):
- 4-bit quantized base model + LoRA adapters (rank 8).
- Zero-copy reference model KL (`beta > 0`): temporarily sets LoRA `scale = 0.0`
  to compute frozen reference log-probs without loading a second model copy.
- Micro-batch=1 gradient accumulation over the G completions in each group so
  large `(G, seq_len, vocab_size)` logit tensors are never materialized at once.
- Scoped gradient checkpointing active ONLY during backward passes so KV-cached
  batched rollout generation runs at full native speed.
- Explicit Metal cache limit + `mx.clear_cache()` + live peak-memory telemetry.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

from grpo_haiku.config import GRPOConfig, ModelConfig, RewardConfig
from grpo_haiku.data import HaikuPromptExample, format_chat_prompt, load_jsonl
from grpo_haiku.relevance import MLXEmbeddingRelevanceScorer, RelevanceScorer
from grpo_haiku.rewards import ClicheTracker, RewardBreakdown, score_group
from grpo_haiku.syllables import tokenize_words

_LORA_TYPES = (LoRALinear, LoRASwitchLinear, LoRAEmbedding)


# ---------------------------------------------------------------------------
# Pure GRPO math helpers (independently unit-tested)
# ---------------------------------------------------------------------------


def cosine_lr(step: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine decay from `max_lr` at step 1 down to `min_lr` at `total_steps`."""
    if total_steps <= 1 or max_lr <= min_lr:
        return max_lr
    progress = min(1.0, max(0.0, (step - 1) / (total_steps - 1)))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def compute_group_advantages(
    rewards: list[float], eps: float = 1e-6
) -> tuple[list[float], bool]:
    """Normalize rewards within a GRPO group; return (advantages, has_variance)."""
    if not rewards:
        return [], False
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = var_r**0.5
    if std_r < eps:
        return [0.0 for _ in rewards], False
    return [(r - mean_r) / (std_r + eps) for r in rewards], True


_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_THINK_LINE_PREFIX_RE = re.compile(r"^\s*([123])\s*:")


def map_tokens_to_lines(
    completion_ids: list[int],
    tokenizer,
    thinking_mode: bool = False,
) -> list[int]:
    """Map each completion token to its 0-based haiku line index (`0..2`, or `-1` for tags/meta).

    In `thinking_mode=True`, both numbered scratchpad drafting lines (`1:`, `2:`, `3:`)
    inside `<think>...</think>` AND the 3 final haiku lines after `</think>` map to
    `0, 1, 2` so per-line syllable credit reaches the drafting/revision tokens directly,
    while `<think>` / `</think>` tags and unprefixed lines map to `-1` (pure `A_global`).
    """
    if not thinking_mode:
        line_indices: list[int] = []
        line_idx = 0
        seen_content = False
        for tid in completion_ids:
            piece = tokenizer.decode([tid])
            if "\n" not in piece:
                line_indices.append(line_idx)
                if piece.strip():
                    seen_content = True
            else:
                before_nl, after_nl = piece.split("\n", 1)
                if not before_nl.strip() and after_nl.strip() and seen_content:
                    line_idx += 1
                    line_indices.append(line_idx)
                    seen_content = True
                else:
                    line_indices.append(line_idx)
                    if before_nl.strip():
                        seen_content = True
                    if seen_content:
                        line_idx += 1
                        seen_content = bool(after_nl.strip())
        return line_indices

    pieces = [tokenizer.decode([tid]) for tid in completion_ids]
    spans: list[tuple[int, int]] = []
    pos = 0
    for p in pieces:
        spans.append((pos, pos + len(p)))
        pos += len(p)
    full_text = "".join(pieces)

    close_m = _THINK_CLOSE_RE.search(full_text)
    if close_m is None:
        return [-1] * len(completion_ids)

    char_label = [-1] * len(full_text)

    # 1. Inside <think>...</think>: label numbered drafting lines 1:, 2:, 3: -> 0, 1, 2
    open_m = _THINK_OPEN_RE.search(full_text, 0, close_m.start())
    think_start = open_m.end() if open_m is not None else 0
    think_end = close_m.start()
    cursor = think_start
    for raw_ln in full_text[think_start:think_end].splitlines(keepends=True):
        ln_end = cursor + len(raw_ln)
        m = _THINK_LINE_PREFIX_RE.match(raw_ln)
        if m is not None:
            k = int(m.group(1)) - 1
            for ci in range(cursor, ln_end):
                char_label[ci] = k
        cursor = ln_end

    # 2. After </think>: label each non-empty haiku line -> 0, 1, 2, ...
    after_start = close_m.end()
    cursor = after_start
    haiku_line_idx = 0
    for raw_ln in full_text[after_start:].splitlines(keepends=True):
        ln_end = cursor + len(raw_ln)
        if raw_ln.strip():
            for ci in range(cursor, ln_end):
                char_label[ci] = haiku_line_idx
            haiku_line_idx += 1
        cursor = ln_end

    # Assign each token the label of its first non-whitespace char (or start char if whitespace)
    out: list[int] = []
    for start, end in spans:
        if start >= end:
            out.append(-1)
            continue
        chosen = char_label[start]
        for ci in range(start, end):
            if not full_text[ci].isspace():
                chosen = char_label[ci]
                break
        out.append(chosen)
    return out


def compute_per_line_token_advantages(
    breakdowns: list[RewardBreakdown],
    comp_token_ids: list[list[int]],
    tokenizer,
    line_credit_weight: float = 0.50,
    eps: float = 1e-6,
    thinking_mode: bool = False,
) -> tuple[list[mx.array], bool]:
    """Build per-token advantage vectors blending global & line-level group advantages.

    When `line_credit_weight > 0`, tokens belonging to Line k (k in 0..2) receive:
        (1 - w) * A_global[i] + w * A_line[k][i]
    so an exact Line 2/3 in a (6, 7, 5) haiku is reinforced while Line 1 is pushed down.
    """
    rewards = [b.total_reward for b in breakdowns]
    global_advs, has_global_var = compute_group_advantages(rewards, eps=eps)

    w = max(0.0, min(1.0, line_credit_weight))
    line_advs_by_idx: list[list[float]] = []
    has_any_line_var = False

    if w > 0.0:
        for line_k in range(3):
            line_scores = [
                b.line_rewards[line_k]
                if (b.is_valid_format and len(b.line_rewards) > line_k)
                else 0.0
                for b in breakdowns
            ]
            l_adv, l_var = compute_group_advantages(line_scores, eps=eps)
            line_advs_by_idx.append(l_adv)
            if l_var:
                has_any_line_var = True

    has_variance = has_global_var or has_any_line_var
    token_advs: list[mx.array] = []

    for i, (bd, tids) in enumerate(zip(breakdowns, comp_token_ids)):
        if not tids:
            token_advs.append(mx.array([], dtype=mx.float32))
            continue
        g_adv = global_advs[i]
        if w <= 0.0 or (not bd.is_valid_format):
            token_advs.append(mx.full((len(tids),), g_adv, dtype=mx.float32))
            continue

        line_map = map_tokens_to_lines(tids, tokenizer, thinking_mode=thinking_mode)
        per_tok = [
            (1.0 - w) * g_adv + w * line_advs_by_idx[lk][i]
            if 0 <= lk < 3
            else g_adv
            for lk in line_map
        ]
        token_advs.append(mx.array(per_tok, dtype=mx.float32))

    return token_advs, has_variance


def kl_k3(new_logprobs: mx.array, ref_logprobs: mx.array) -> mx.array:
    """Unbiased, non-negative Schulman k3 KL estimator: exp(r) - r - 1, r = log(p_ref / p_new)."""
    log_ratio = mx.clip(ref_logprobs - new_logprobs, -20.0, 20.0)
    return mx.exp(log_ratio) - log_ratio - 1.0


def grpo_token_loss(
    new_logprobs: mx.array,
    old_logprobs: mx.array,
    ref_logprobs: mx.array,
    advantage: float | mx.array,
    kl_beta: float = 0.04,
    clip_eps: float = 0.2,
    clip_eps_high: float | None = None,
    token_entropy: mx.array | None = None,
    entropy_coef: float = 0.0,
) -> tuple[mx.array, mx.array]:
    """Compute mean per-token GRPO asymmetric-clipped surrogate + k3 KL - entropy bonus."""
    log_ratio = mx.clip(new_logprobs - old_logprobs, -20.0, 20.0)
    ratio = mx.exp(log_ratio)
    adv = (
        advantage.astype(mx.float32)
        if isinstance(advantage, mx.array)
        else mx.array(advantage, dtype=mx.float32)
    )

    hi = clip_eps if clip_eps_high is None else clip_eps_high
    surr1 = ratio * adv
    surr2 = mx.clip(ratio, 1.0 - clip_eps, 1.0 + hi) * adv
    policy_loss = -mx.minimum(surr1, surr2)

    kl = kl_k3(new_logprobs, ref_logprobs)
    total_loss = mx.mean(policy_loss + kl_beta * kl)
    if token_entropy is not None and entropy_coef > 0.0:
        total_loss = total_loss - entropy_coef * mx.mean(token_entropy.astype(mx.float32))
    return total_loss, mx.mean(kl)


# ---------------------------------------------------------------------------
# Zero-copy reference toggle & scoped gradient checkpointing
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def no_lora(model: nn.Module) -> Iterator[None]:
    """Temporarily zero out all LoRA scales to evaluate the frozen base reference model."""
    saved: list[tuple[nn.Module, float]] = []
    for _, module in model.named_modules():
        if isinstance(module, _LORA_TYPES):
            saved.append((module, module.scale))
            module.scale = 0.0
    try:
        yield
    finally:
        for module, orig_scale in saved:
            module.scale = orig_scale


def snapshot_lora_weights(model: nn.Module) -> list[tuple[str, mx.array]]:
    """Capture a detached in-memory copy of trainable LoRA tensors (~10-20 MB)."""
    flat = tree_flatten(model.trainable_parameters())
    snap = [(k, mx.array(v)) for k, v in flat]
    if snap:
        mx.eval([v for _, v in snap])
    return snap


def load_lora_weights_into_model(model: nn.Module, adapter_path: str | Path) -> None:
    """Load saved LoRA weights into an already LoRA-wrapped model."""
    weights_path = Path(adapter_path) / "adapters.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Adapter weights not found: {weights_path}")
    weights = mx.load(str(weights_path))
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.trainable_parameters())


@contextlib.contextmanager
def reference_policy(
    model: nn.Module,
    ref_lora_weights: list[tuple[str, mx.array]] | None,
) -> Iterator[None]:
    """Evaluate ref log-probs against an SFT LoRA snapshot (or base model if None)."""
    if ref_lora_weights is None:
        with no_lora(model):
            yield
        return

    current = snapshot_lora_weights(model)
    model.load_weights(ref_lora_weights, strict=False)
    mx.eval(model.trainable_parameters())
    try:
        yield
    finally:
        model.load_weights(current, strict=False)
        mx.eval(model.trainable_parameters())


@contextlib.contextmanager
def gradient_checkpointing(model: nn.Module, enabled: bool = True) -> Iterator[None]:
    """Enable MLX gradient checkpointing ONLY within this context (never during KV-cached rollouts)."""
    if not enabled or not hasattr(model, "layers") or not model.layers:
        yield
        return

    layer_cls = type(model.layers[0])
    orig_call = layer_cls.__call__

    def checkpointed_call(self, *args, **kwargs):
        def inner_fn(params, *inner_args, **inner_kwargs):
            self.update(params)
            return orig_call(self, *inner_args, **inner_kwargs)

        return mx.checkpoint(inner_fn)(self.trainable_parameters(), *args, **kwargs)

    layer_cls.__call__ = checkpointed_call
    try:
        yield
    finally:
        layer_cls.__call__ = orig_call


def clip_grad_norm(grads: dict, max_norm: float) -> tuple[dict, float]:
    """Clip global L2 norm of a nested gradient pytree."""
    flat = tree_flatten(grads)
    if not flat:
        return grads, 0.0
    sq_norm = sum(mx.sum(g.astype(mx.float32) ** 2) for _, g in flat)
    total_norm = mx.sqrt(sq_norm)
    norm_val = float(total_norm.item())
    if max_norm > 0.0 and norm_val > max_norm:
        scale = mx.array(max_norm / (norm_val + 1e-6), dtype=mx.float32)
        grads = tree_map(lambda g: (g * scale).astype(g.dtype), grads)
    return grads, norm_val


def sequence_completion_logprobs(
    model: nn.Module,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> mx.array:
    """Compute per-token log-probabilities for `completion_ids` given `prompt_ids`.

    Runs on a single sequence (batch=1) so peak activation & logits memory stay tiny.
    """
    full_ids = prompt_ids + completion_ids
    tokens = mx.array(full_ids, dtype=mx.int32)[None, :]  # (1, L)
    inputs = tokens[:, :-1]  # (1, L-1)
    targets = tokens[:, 1:]  # (1, L-1)

    logits = model(inputs)  # (1, L-1, V)
    start = len(prompt_ids) - 1
    comp_logits = logits[:, start:, :].astype(mx.float32)  # (1, T_comp, V)
    comp_targets = targets[:, start:]  # (1, T_comp)

    # -cross_entropy gives exact log_softmax(logits)[target] without keeping full log_probs graph
    nll = nn.losses.cross_entropy(comp_logits, comp_targets, reduction="none")
    return -nll[0]  # (T_comp,)


def sequence_completion_logprobs_and_entropy(
    model: nn.Module,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> tuple[mx.array, mx.array]:
    """Compute per-token target log-probs AND Shannon entropy on completion tokens (batch=1)."""
    full_ids = prompt_ids + completion_ids
    tokens = mx.array(full_ids, dtype=mx.int32)[None, :]
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    logits = model(inputs)
    start = len(prompt_ids) - 1
    comp_logits = logits[:, start:, :].astype(mx.float32)  # (1, T_comp, V)
    comp_targets = targets[:, start:]  # (1, T_comp)

    log_probs = comp_logits - mx.logsumexp(comp_logits, axis=-1, keepdims=True)
    target_lp = mx.take_along_axis(log_probs, comp_targets[..., None], axis=-1)[0, :, 0]
    probs = mx.exp(log_probs)
    entropy = -mx.sum(probs * log_probs, axis=-1)[0]
    return target_lp, entropy


# ---------------------------------------------------------------------------
# Adapter persistence (100% compatible with `mlx_lm.load(..., adapter_path=...)`)
# ---------------------------------------------------------------------------


def save_adapter(
    model: nn.Module,
    model_cfg: ModelConfig,
    out_dir: str | Path,
) -> Path:
    """Save trainable LoRA weights and adapter_config.json in mlx-lm standard format."""
    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(save_dir / "adapters.safetensors"), adapter_weights)

    num_layers = model_cfg.lora_layers
    if num_layers < 0 and hasattr(model, "layers"):
        num_layers = len(model.layers)

    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": model_cfg.lora_rank,
            "scale": model_cfg.lora_scale,
            "dropout": model_cfg.lora_dropout,
            "keys": list(model_cfg.lora_keys),
        },
    }
    with (save_dir / "adapter_config.json").open("w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)
    return save_dir


def normalize_completion_text(prompt_text: str, completion_text: str) -> str:
    """Return completion text cleanly stripped for direct 3-line haiku scoring."""
    return completion_text.strip()


def truncate_after_haiku_close(
    text: str, token_ids: list[int], tokenizer
) -> tuple[str, list[int]]:
    """Trim trailing gibberish if the model already closed `</haiku>` cleanly (optional safety)."""
    # We intentionally do NOT strip trailing text during training scoring so trailing text
    # gets penalized by `trailing_text_penalty`; we only guarantee non-empty token_ids.
    return text, token_ids


# ---------------------------------------------------------------------------
# Trainer setup & main GRPO loop
# ---------------------------------------------------------------------------


def build_lora_model(model_cfg: ModelConfig):
    """Load 4-bit base model, freeze base weights, and inject LoRA layers."""
    model, tokenizer = load(model_cfg.model_name)
    model.freeze()

    num_layers = model_cfg.lora_layers
    if num_layers < 0:
        num_layers = len(model.layers)
    num_layers = min(num_layers, len(model.layers))

    lora_params = {
        "rank": model_cfg.lora_rank,
        "scale": model_cfg.lora_scale,
        "dropout": model_cfg.lora_dropout,
        "keys": list(model_cfg.lora_keys),
    }
    linear_to_lora_layers(model, num_layers=num_layers, config=lora_params)
    model.train()
    print_trainable_parameters(model)
    return model, tokenizer


def compute_quick_diversity(haikus: list[str]) -> tuple[float, float]:
    """Return `(unique_haiku_ratio, distinct_2)` across a list of generated haikus."""
    non_empty = [h.strip().lower() for h in haikus if h.strip()]
    if not non_empty:
        return 0.0, 0.0
    unique_ratio = len(set(non_empty)) / len(non_empty)
    bigrams: list[tuple[str, str]] = []
    for h in non_empty:
        words = tokenize_words(h)
        for i in range(len(words) - 1):
            bigrams.append((words[i], words[i + 1]))
    distinct_2 = (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    return unique_ratio, distinct_2


def run_quick_eval(
    model: nn.Module,
    tokenizer,
    eval_examples: list[HaikuPromptExample],
    reward_cfg: RewardConfig,
    relevance_scorer: RelevanceScorer | None,
    max_tokens: int,
    subset_size: int = 50,
    batch_size: int = 16,
) -> dict[str, float]:
    """Run a fast greedy evaluation snapshot on a held-out subset including distinct-2."""
    subset = eval_examples[:subset_size]
    breakdowns: list[RewardBreakdown] = []

    model.eval()
    for start in range(0, len(subset), batch_size):
        chunk = subset[start : start + batch_size]
        prompt_texts = [
            format_chat_prompt(
                tokenizer,
                ex.prompt,
                thinking_mode=reward_cfg.thinking_mode,
                think_format=reward_cfg.think_format,
            )
            for ex in chunk
        ]
        prompt_ids_list = [
            tokenizer.encode(pt, add_special_tokens=False) for pt in prompt_texts
        ]
        resp = batch_generate(
            model,
            tokenizer,
            prompts=prompt_ids_list,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
            return_token_ids=False,
        )
        for ex, pt, raw_text in zip(chunk, prompt_texts, resp.texts):
            full_text = normalize_completion_text(pt, raw_text)
            bd = score_group(
                topic=ex.topic,
                completions=[full_text],
                cfg=reward_cfg,
                relevance_scorer=relevance_scorer,
            )[0]
            breakdowns.append(bd)
    model.train()

    n = max(1, len(breakdowns))
    haiku_texts = ["\n".join(b.haiku_lines) for b in breakdowns]
    unique_ratio, distinct_2 = compute_quick_diversity(haiku_texts)
    stats = {
        "eval_reward": sum(b.total_reward for b in breakdowns) / n,
        "eval_exact_575": sum(1 for b in breakdowns if b.is_exact_575) / n,
        "eval_valid_format": sum(1 for b in breakdowns if b.is_valid_format) / n,
        "eval_relevance": sum(b.raw_cosine_sim for b in breakdowns) / n,
        "eval_oov_rate": sum(1 for b in breakdowns if b.oov_words) / n,
        "eval_unique_ratio": unique_ratio,
        "eval_distinct_2": distinct_2,
    }
    if reward_cfg.thinking_mode:
        stats["eval_grounding"] = sum(b.think_grounding for b in breakdowns) / n
    return stats


def train_grpo(
    model_cfg: ModelConfig,
    reward_cfg: RewardConfig,
    grpo_cfg: GRPOConfig,
) -> Path:
    """Execute the full MLX GRPO training run and return the final adapter path."""
    random.seed(grpo_cfg.seed)
    mx.random.seed(grpo_cfg.seed)

    # Cap Metal buffer cache to stay comfortably inside 16GB unified memory
    if hasattr(mx, "set_cache_limit") and grpo_cfg.metal_cache_limit_gb > 0:
        mx.set_cache_limit(int(grpo_cfg.metal_cache_limit_gb * (1024**3)))

    run_dir = Path(grpo_cfg.runs_dir) / grpo_cfg.run_name
    adapters_dir = run_dir / "adapters"
    run_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir.mkdir(parents=True, exist_ok=True)

    # Persist exact run config for reproducibility
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": asdict(model_cfg),
                "reward": asdict(reward_cfg),
                "grpo": asdict(grpo_cfg),
            },
            f,
            indent=2,
        )

    train_examples = load_jsonl(grpo_cfg.train_file)
    eval_examples = load_jsonl(grpo_cfg.eval_file) if Path(grpo_cfg.eval_file).exists() else []
    if not train_examples:
        raise ValueError(f"No training examples found in {grpo_cfg.train_file}")

    print(f"Loading model: {model_cfg.model_name}")
    model, tokenizer = build_lora_model(model_cfg)

    # Resolve reference policy for KL independently of `--resume-adapter` so iterative runs
    # can anchor KL to a high-diversity policy (e.g. `sft_v1` or `base`) without ratcheting.
    ref_lora_weights: list[tuple[str, mx.array]] | None = None
    kl_ref_spec = grpo_cfg.kl_ref_adapter
    if kl_ref_spec is not None and kl_ref_spec.lower() in {"base", "none"}:
        print("Anchoring KL to frozen base model (explicit --kl-ref-adapter=base)")
        ref_lora_weights = None
    elif kl_ref_spec is not None:
        print(f"Anchoring KL to reference adapter: {kl_ref_spec}")
        init_snap = snapshot_lora_weights(model)
        load_lora_weights_into_model(model, kl_ref_spec)
        ref_lora_weights = snapshot_lora_weights(model)
        model.load_weights(init_snap, strict=False)
        mx.eval(model.trainable_parameters())

    if grpo_cfg.resume_adapter:
        print(f"Resuming trainable LoRA weights from: {grpo_cfg.resume_adapter}")
        load_lora_weights_into_model(model, grpo_cfg.resume_adapter)
        if kl_ref_spec is None:
            print("Anchoring KL to resumed adapter (pass --kl-ref-adapter to override)")
            ref_lora_weights = snapshot_lora_weights(model)

    relevance_scorer: RelevanceScorer | None = None
    if reward_cfg.relevance_weight > 0.0:
        print(f"Loading local MLX relevance embedder: {reward_cfg.embedding_model}")
        relevance_scorer = MLXEmbeddingRelevanceScorer(reward_cfg.embedding_model)

    optimizer = optim.AdamW(
        learning_rate=grpo_cfg.learning_rate,
        weight_decay=grpo_cfg.weight_decay,
    )
    sampler = make_sampler(temp=grpo_cfg.temperature, top_p=grpo_cfg.top_p)

    # Single-sequence value_and_grad function for micro-batch=1 accumulation
    def _single_seq_loss(
        model_arg: nn.Module,
        prompt_ids: list[int],
        completion_ids: list[int],
        old_lp: mx.array,
        ref_lp: mx.array,
        advantage: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if grpo_cfg.entropy_coef > 0.0:
            new_lp, tok_ent = sequence_completion_logprobs_and_entropy(
                model_arg, prompt_ids, completion_ids
            )
            mean_ent = mx.mean(tok_ent)
        else:
            new_lp = sequence_completion_logprobs(model_arg, prompt_ids, completion_ids)
            tok_ent = None
            mean_ent = mx.array(0.0, dtype=mx.float32)

        loss, mean_kl = grpo_token_loss(
            new_logprobs=new_lp,
            old_logprobs=old_lp,
            ref_logprobs=ref_lp,
            advantage=advantage,
            kl_beta=grpo_cfg.kl_beta,
            clip_eps=grpo_cfg.clip_eps,
            clip_eps_high=grpo_cfg.clip_eps_high,
            token_entropy=tok_ent,
            entropy_coef=grpo_cfg.entropy_coef,
        )
        return loss, mean_kl, mean_ent

    loss_and_grad_fn = nn.value_and_grad(model, _single_seq_loss)
    metrics_path = run_dir / "metrics.jsonl"
    cliche_tracker = ClicheTracker(
        window=reward_cfg.cliche_window,
        bigram_min_count=reward_cfg.cliche_bigram_min_count,
        starter1_min_count=reward_cfg.cliche_starter1_min_count,
        starter2_min_count=reward_cfg.cliche_starter2_min_count,
        unigram_min_count=reward_cfg.cliche_unigram_min_count,
    )

    rng = random.Random(grpo_cfg.seed)
    order = list(range(len(train_examples)))
    rng.shuffle(order)

    print(
        f"Starting GRPO training: steps={grpo_cfg.num_steps}, G={grpo_cfg.group_size}, "
        f"thinking={grpo_cfg.thinking_mode} ({grpo_cfg.think_format}), "
        f"lr={grpo_cfg.learning_rate}->{grpo_cfg.min_learning_rate}, kl_beta={grpo_cfg.kl_beta}, "
        f"clip=[{grpo_cfg.clip_eps},{grpo_cfg.clip_eps_high}], ent_coef={grpo_cfg.entropy_coef}, "
        f"line_credit_w={grpo_cfg.line_credit_weight}, relevance_w={reward_cfg.relevance_weight}, "
        f"exact_min_sim={reward_cfg.exact_bonus_min_sim}, arith_pen={reward_cfg.think_arith_penalty}, "
        f"max_tokens={grpo_cfg.max_completion_tokens}"
    )

    for step in range(1, grpo_cfg.num_steps + 1):
        t0 = time.perf_counter()
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()

        current_lr = cosine_lr(
            step=step,
            total_steps=grpo_cfg.num_steps,
            max_lr=grpo_cfg.learning_rate,
            min_lr=grpo_cfg.min_learning_rate,
        )
        optimizer.learning_rate = current_lr

        ex = train_examples[order[(step - 1) % len(order)]]
        if step % len(order) == 0:
            rng.shuffle(order)

        prompt_text = format_chat_prompt(
            tokenizer,
            ex.prompt,
            thinking_mode=grpo_cfg.thinking_mode,
            think_format=grpo_cfg.think_format,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # 1. Parallel batched rollout of G completions (no grad checkpointing during KV decode!)
        model.eval()
        rollout = batch_generate(
            model,
            tokenizer,
            prompts=[prompt_ids] * grpo_cfg.group_size,
            max_tokens=grpo_cfg.max_completion_tokens,
            sampler=sampler,
            return_token_ids=True,
            return_logprobs=False,
        )
        model.train()

        completions = [normalize_completion_text(prompt_text, txt) for txt in rollout.texts]
        comp_token_ids = rollout.token_ids or [
            tokenizer.encode(txt, add_special_tokens=False) for txt in rollout.texts
        ]

        # 2. Deterministic verifiable reward + fitness sharing + per-line credit assignment
        breakdowns = score_group(
            topic=ex.topic,
            completions=completions,
            cfg=reward_cfg,
            relevance_scorer=relevance_scorer,
            cliche_tracker=cliche_tracker,
            apply_fitness_sharing=True,
        )
        rewards = [b.total_reward for b in breakdowns]

        # Record the top valid completion from this step into the rolling cliche window
        valid_bds = [(i, b) for i, b in enumerate(breakdowns) if b.is_valid_format]
        if valid_bds:
            _, top_bd = max(valid_bds, key=lambda pair: pair[1].total_reward)
            cliche_tracker.record("\n".join(top_bd.haiku_lines), topic=ex.topic)
        token_advantages, has_variance = compute_per_line_token_advantages(
            breakdowns=breakdowns,
            comp_token_ids=comp_token_ids,
            tokenizer=tokenizer,
            line_credit_weight=grpo_cfg.line_credit_weight,
            eps=grpo_cfg.adv_eps,
            thinking_mode=grpo_cfg.thinking_mode,
        )

        # Filter out degenerate empty token rollouts
        valid_indices = [i for i, tids in enumerate(comp_token_ids) if len(tids) > 0]

        loss_val = 0.0
        kl_val = 0.0
        ent_val = 0.0
        grad_norm_val = 0.0
        skipped_update = (not has_variance) or (len(valid_indices) == 0)

        if not skipped_update:
            # 3. Reference log-probs (anchored to --kl-ref-adapter, SFT snapshot, or base model)
            ref_logprobs_list: dict[int, mx.array] = {}
            with reference_policy(model, ref_lora_weights):
                for idx in valid_indices:
                    ref_lp = mx.stop_gradient(
                        sequence_completion_logprobs(model, prompt_ids, comp_token_ids[idx])
                    )
                    mx.eval(ref_lp)
                    ref_logprobs_list[idx] = ref_lp

            # Capture exact old_logprobs on current policy before inner epochs
            old_logprobs_list: dict[int, mx.array] = {}
            for idx in valid_indices:
                old_lp = mx.stop_gradient(
                    sequence_completion_logprobs(model, prompt_ids, comp_token_ids[idx])
                )
                mx.eval(old_lp)
                old_logprobs_list[idx] = old_lp

            # 4. Micro-batch=1 gradient accumulation with scoped gradient checkpointing
            epoch_losses: list[float] = []
            epoch_kls: list[float] = []
            epoch_ents: list[float] = []
            inv_n = 1.0 / len(valid_indices)

            with gradient_checkpointing(model, enabled=model_cfg.grad_checkpoint):
                for _epoch in range(grpo_cfg.grpo_epochs):
                    accum_grads = None
                    step_loss = 0.0
                    step_kl = 0.0
                    step_ent = 0.0

                    for idx in valid_indices:
                        (seq_loss, seq_kl, seq_ent), grads = loss_and_grad_fn(
                            model,
                            prompt_ids,
                            comp_token_ids[idx],
                            old_logprobs_list[idx],
                            ref_logprobs_list[idx],
                            token_advantages[idx],
                        )
                        scaled_grads = tree_map(lambda g: g * inv_n, grads)
                        if accum_grads is None:
                            accum_grads = scaled_grads
                        else:
                            accum_grads = tree_map(lambda a, b: a + b, accum_grads, scaled_grads)

                        mx.eval(accum_grads, seq_loss, seq_kl, seq_ent)
                        step_loss += float(seq_loss.item()) * inv_n
                        step_kl += float(seq_kl.item()) * inv_n
                        step_ent += float(seq_ent.item()) * inv_n

                    if accum_grads is not None:
                        accum_grads, grad_norm_val = clip_grad_norm(
                            accum_grads, grpo_cfg.grad_clip_norm
                        )
                        optimizer.update(model, accum_grads)
                        mx.eval(model.parameters(), optimizer.state)

                    epoch_losses.append(step_loss)
                    epoch_kls.append(step_kl)
                    epoch_ents.append(step_ent)

            loss_val = sum(epoch_losses) / max(1, len(epoch_losses))
            kl_val = sum(epoch_kls) / max(1, len(epoch_kls))
            ent_val = sum(epoch_ents) / max(1, len(epoch_ents))

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

        step_sec = time.perf_counter() - t0
        peak_mem_gb = (
            mx.get_peak_memory() / (1024**3) if hasattr(mx, "get_peak_memory") else 0.0
        )

        # Telemetry
        n_g = len(breakdowns)
        reward_mean = sum(rewards) / n_g
        reward_max = max(rewards)
        exact_rate = sum(1 for b in breakdowns if b.is_exact_575) / n_g
        format_rate = sum(1 for b in breakdowns if b.is_valid_format) / n_g
        rel_mean = sum(b.raw_cosine_sim for b in breakdowns) / n_g
        oov_rate = sum(1 for b in breakdowns if b.oov_words) / n_g
        cliche_rate = sum(1 for b in breakdowns if b.cliche_penalty < 0.0) / n_g
        grounding_mean = sum(b.think_grounding for b in breakdowns) / n_g
        avg_comp_len = sum(len(t) for t in comp_token_ids) / n_g

        row = {
            "step": step,
            "topic": ex.topic,
            "reward_mean": round(reward_mean, 4),
            "reward_max": round(reward_max, 4),
            "exact_575_rate": round(exact_rate, 4),
            "format_rate": round(format_rate, 4),
            "relevance_sim": round(rel_mean, 4),
            "oov_rate": round(oov_rate, 4),
            "cliche_rate": round(cliche_rate, 4),
            "grounding": round(grounding_mean, 4),
            "loss": round(loss_val, 5),
            "kl": round(kl_val, 5),
            "entropy": round(ent_val, 4),
            "lr": round(current_lr, 8),
            "grad_norm": round(grad_norm_val, 4),
            "avg_tokens": round(avg_comp_len, 1),
            "skipped": skipped_update,
            "peak_mem_gb": round(peak_mem_gb, 3),
            "step_sec": round(step_sec, 2),
        }

        gnd_str = f" gnd={grounding_mean:.2f}" if grpo_cfg.thinking_mode else ""
        print(
            f"[step {step:04d}/{grpo_cfg.num_steps}] "
            f"r={reward_mean:+.3f} (max={reward_max:+.3f}) | "
            f"575={exact_rate*100:5.1f}% fmt={format_rate*100:5.1f}% rel={rel_mean:.2f} cl={cliche_rate*100:4.0f}%{gnd_str} | "
            f"kl={kl_val:.4f} ent={ent_val:.2f} gnorm={grad_norm_val:.2f} | "
            f"toks={avg_comp_len:5.1f} mem={peak_mem_gb:.2f}GB ({step_sec:.1f}s)"
            + (" [skip:0var]" if skipped_update else "")
        )

        if grpo_cfg.sample_every > 0 and (step == 1 or step % grpo_cfg.sample_every == 0):
            best_idx = max(range(n_g), key=lambda i: rewards[i])
            best = breakdowns[best_idx]
            print(f"  ── Sample (topic: {ex.topic!r}, counts={best.line_counts}, r={best.total_reward:+.2f}) ──")
            if grpo_cfg.thinking_mode and best.think_text:
                for tl in best.think_text.splitlines():
                    print(f"    [think] {tl}")
            if best.haiku_lines:
                for line in best.haiku_lines:
                    print(f"    {line}")
            else:
                print(f"    [unparsed] {completions[best_idx][:140]!r}")

        if (
            eval_examples
            and grpo_cfg.eval_every > 0
            and step % grpo_cfg.eval_every == 0
        ):
            eval_stats = run_quick_eval(
                model=model,
                tokenizer=tokenizer,
                eval_examples=eval_examples,
                reward_cfg=reward_cfg,
                relevance_scorer=relevance_scorer,
                max_tokens=grpo_cfg.max_completion_tokens,
                subset_size=grpo_cfg.eval_subset_size,
            )
            row.update({k: round(v, 4) for k, v in eval_stats.items()})
            eval_gnd = (
                f" | gnd={eval_stats['eval_grounding']:.2f}"
                if "eval_grounding" in eval_stats
                else ""
            )
            print(
                f"  ★ Eval@{step} ({min(len(eval_examples), grpo_cfg.eval_subset_size)} held-out): "
                f"r={eval_stats['eval_reward']:+.3f} | "
                f"575={eval_stats['eval_exact_575']*100:.1f}% | "
                f"d2={eval_stats['eval_distinct_2']:.3f} uniq={eval_stats['eval_unique_ratio']*100:.0f}% | "
                f"fmt={eval_stats['eval_valid_format']*100:.1f}% | "
                f"rel={eval_stats['eval_relevance']:.2f}{eval_gnd}"
            )
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if grpo_cfg.save_every > 0 and step % grpo_cfg.save_every == 0:
            ckpt = save_adapter(model, model_cfg, adapters_dir / f"step_{step:04d}")
            print(f"  Saved checkpoint -> {ckpt}")

    final_path = save_adapter(model, model_cfg, adapters_dir / "final")
    print(f"Training complete! Final LoRA adapter saved to: {final_path}")
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small MLX LLM with GRPO to write 5-7-5 haikus.")
    parser.add_argument("--model", type=str, default=ModelConfig.model_name)
    parser.add_argument("--train-file", type=str, default=GRPOConfig.train_file)
    parser.add_argument("--eval-file", type=str, default=GRPOConfig.eval_file)
    parser.add_argument("--resume-adapter", type=str, default=GRPOConfig.resume_adapter)
    parser.add_argument("--kl-ref-adapter", type=str, default=GRPOConfig.kl_ref_adapter)
    parser.add_argument("--run-name", type=str, default=GRPOConfig.run_name)
    parser.add_argument("--runs-dir", type=str, default=GRPOConfig.runs_dir)
    parser.add_argument("--steps", type=int, default=GRPOConfig.num_steps)
    parser.add_argument("--group-size", type=int, default=GRPOConfig.group_size)
    parser.add_argument("--grpo-epochs", type=int, default=GRPOConfig.grpo_epochs)
    parser.add_argument("--thinking-mode", action="store_true")
    parser.add_argument("--think-format", type=str, default="sum", choices=["sum", "cumulative"])
    parser.add_argument("--think-arith-penalty", type=float, default=RewardConfig.think_arith_penalty)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temp", type=float, default=GRPOConfig.temperature)
    parser.add_argument("--top-p", type=float, default=GRPOConfig.top_p)
    parser.add_argument("--lr", type=float, default=GRPOConfig.learning_rate)
    parser.add_argument("--min-lr", type=float, default=GRPOConfig.min_learning_rate)
    parser.add_argument("--kl-beta", type=float, default=GRPOConfig.kl_beta)
    parser.add_argument("--clip-high", type=float, default=GRPOConfig.clip_eps_high)
    parser.add_argument("--entropy-coef", type=float, default=GRPOConfig.entropy_coef)
    parser.add_argument("--line-credit-weight", type=float, default=GRPOConfig.line_credit_weight)
    parser.add_argument("--relevance-weight", type=float, default=RewardConfig.relevance_weight)
    parser.add_argument("--exact-min-sim", type=float, default=RewardConfig.exact_bonus_min_sim)
    parser.add_argument("--lora-rank", type=int, default=ModelConfig.lora_rank)
    parser.add_argument("--lora-scale", type=float, default=ModelConfig.lora_scale)
    parser.add_argument("--lora-layers", type=int, default=ModelConfig.lora_layers)
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--eval-every", type=int, default=GRPOConfig.eval_every)
    parser.add_argument("--eval-subset", type=int, default=GRPOConfig.eval_subset_size)
    parser.add_argument("--save-every", type=int, default=GRPOConfig.save_every)
    parser.add_argument("--sample-every", type=int, default=GRPOConfig.sample_every)
    parser.add_argument("--seed", type=int, default=GRPOConfig.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_max_tokens = (
        (256 if args.think_format == "cumulative" else 224)
        if args.thinking_mode
        else GRPOConfig.max_completion_tokens
    )
    max_tokens = args.max_tokens if args.max_tokens is not None else default_max_tokens
    model_cfg = ModelConfig(
        model_name=args.model,
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
        lora_layers=args.lora_layers,
        grad_checkpoint=not args.no_grad_checkpoint,
    )
    reward_cfg = RewardConfig(
        relevance_weight=args.relevance_weight,
        exact_bonus_min_sim=args.exact_min_sim,
        thinking_mode=args.thinking_mode,
        think_format=args.think_format,
        think_arith_penalty=args.think_arith_penalty,
    )
    grpo_cfg = GRPOConfig(
        train_file=args.train_file,
        eval_file=args.eval_file,
        resume_adapter=args.resume_adapter,
        kl_ref_adapter=args.kl_ref_adapter,
        run_name=args.run_name,
        runs_dir=args.runs_dir,
        thinking_mode=args.thinking_mode,
        think_format=args.think_format,
        num_steps=args.steps,
        group_size=args.group_size,
        grpo_epochs=args.grpo_epochs,
        max_completion_tokens=max_tokens,
        temperature=args.temp,
        top_p=args.top_p,
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        kl_beta=args.kl_beta,
        clip_eps_high=args.clip_high,
        entropy_coef=args.entropy_coef,
        line_credit_weight=args.line_credit_weight,
        eval_every=args.eval_every,
        eval_subset_size=args.eval_subset,
        save_every=args.save_every,
        sample_every=args.sample_every,
        seed=args.seed,
    )
    train_grpo(model_cfg, reward_cfg, grpo_cfg)


if __name__ == "__main__":
    main()
