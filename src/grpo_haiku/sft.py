"""Rejection-sampling data collector and LoRA SFT trainer for 5-7-5 haiku cold-start.

Uses the exact same 4-bit base model and attention+MLP LoRA architecture as `grpo.py`,
saving adapters in standard `mlx-lm` format so GRPO can resume directly from the SFT
checkpoint via `--resume-adapter`.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from grpo_haiku.config import ModelConfig, RewardConfig
from grpo_haiku.data import HaikuPromptExample, format_chat_prompt, load_jsonl
from grpo_haiku.grpo import (
    build_lora_model,
    clip_grad_norm,
    gradient_checkpointing,
    normalize_completion_text,
    save_adapter,
    sequence_completion_logprobs,
)
from grpo_haiku.relevance import MLXEmbeddingRelevanceScorer, RelevanceScorer
from grpo_haiku.rewards import RewardBreakdown, score_group


def is_sft_worthy(bd: RewardBreakdown) -> bool:
    """Strict quality filter: keep only exact 5-7-5, on-topic, clean, zero-OOV haikus."""
    return (
        bd.is_valid_format
        and bd.is_exact_575
        and bd.exact_bonus > 0.0
        and len(bd.oov_words) == 0
        and bd.echo_penalty == 0.0
        and bd.repetition_penalty == 0.0
    )


def collect_sft_haikus(
    model_name: str,
    train_file: str,
    out_file: str,
    adapter_path: str | None = None,
    num_prompts: int = 450,
    samples_per_prompt: int = 8,
    prompt_batch_size: int = 8,
    max_tokens: int = 64,
    temp: float = 0.95,
    top_p: float = 0.95,
    seed: int = 42,
    reward_cfg: RewardConfig | None = None,
    relevance_scorer: RelevanceScorer | None = None,
) -> list[dict]:
    """Generate candidate haikus on training prompts and keep verified exact 5-7-5 winners."""
    t0 = time.perf_counter()
    reward_cfg = reward_cfg or RewardConfig()
    examples = load_jsonl(train_file)
    rng = random.Random(seed)
    rng.shuffle(examples)
    examples = examples[:num_prompts]

    mx.random.seed(seed)
    print(
        f"Rejection sampling from {model_name} (adapter={adapter_path or 'NONE'}): "
        f"{len(examples)} prompts x {samples_per_prompt} samples = {len(examples) * samples_per_prompt} candidates..."
    )
    model, tokenizer = load(model_name, adapter_path=adapter_path)
    scorer = relevance_scorer
    if scorer is None and reward_cfg.relevance_weight > 0.0:
        scorer = MLXEmbeddingRelevanceScorer(reward_cfg.embedding_model)

    sampler = make_sampler(temp=temp, top_p=top_p)
    accepted: list[dict] = []
    seen_haikus: set[str] = set()
    total_candidates = 0

    for start in range(0, len(examples), prompt_batch_size):
        chunk = examples[start : start + prompt_batch_size]
        flat_prompt_ids: list[list[int]] = []
        flat_prompt_texts: list[str] = []
        for ex in chunk:
            pt = format_chat_prompt(tokenizer, ex.prompt)
            pids = tokenizer.encode(pt, add_special_tokens=False)
            for _ in range(samples_per_prompt):
                flat_prompt_texts.append(pt)
                flat_prompt_ids.append(pids)

        resp = batch_generate(
            model,
            tokenizer,
            prompts=flat_prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            return_token_ids=False,
        )

        # Score per prompt group to reuse cached topic embeddings
        for i, ex in enumerate(chunk):
            offset = i * samples_per_prompt
            raw_slice = resp.texts[offset : offset + samples_per_prompt]
            pt_slice = flat_prompt_texts[offset : offset + samples_per_prompt]
            completions = [
                normalize_completion_text(pt, txt) for pt, txt in zip(pt_slice, raw_slice)
            ]
            breakdowns = score_group(
                topic=ex.topic,
                completions=completions,
                cfg=reward_cfg,
                relevance_scorer=scorer,
            )
            total_candidates += len(breakdowns)

            # Keep up to 2 best distinct exact 5-7-5 haikus per prompt to preserve topic diversity
            winners = [
                (comp, bd)
                for comp, bd in zip(completions, breakdowns)
                if is_sft_worthy(bd)
            ]
            winners.sort(key=lambda pair: pair[1].total_reward, reverse=True)
            kept_for_prompt = 0
            for comp, bd in winners:
                key = "\n".join(bd.haiku_lines).strip().lower()
                if key in seen_haikus:
                    continue
                seen_haikus.add(key)
                accepted.append(
                    {
                        "id": ex.id,
                        "domain": ex.domain,
                        "topic": ex.topic,
                        "prompt": ex.prompt,
                        "completion": "\n".join(bd.haiku_lines),
                        "total_reward": bd.total_reward,
                        "raw_cosine_sim": bd.raw_cosine_sim,
                        "line_counts": list(bd.line_counts),
                    }
                )
                kept_for_prompt += 1
                if kept_for_prompt >= 2:
                    break

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

        done_prompts = min(start + prompt_batch_size, len(examples))
        if done_prompts % (prompt_batch_size * 5) == 0 or done_prompts == len(examples):
            rate = (len(accepted) / max(1, total_candidates)) * 100.0
            elapsed = time.perf_counter() - t0
            print(
                f"  [collect {done_prompts:4d}/{len(examples)}] "
                f"accepted={len(accepted)} ({rate:.1f}% of {total_candidates}) [{elapsed:.1f}s]"
            )

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in accepted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(accepted)} verified exact 5-7-5 SFT examples -> {out_path}")
    return accepted


def sft_sequence_loss(
    model: nn.Module,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> mx.array:
    """Mean negative log-likelihood on completion tokens only (prompt tokens masked out)."""
    logprobs = sequence_completion_logprobs(model, prompt_ids, completion_ids)
    return -mx.mean(logprobs)


def train_sft(
    model_cfg: ModelConfig,
    sft_file: str,
    run_name: str = "sft_v1",
    runs_dir: str = "runs",
    epochs: int = 2,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    grad_accum_steps: int = 4,
    grad_clip_norm: float = 1.0,
    seed: int = 42,
) -> Path:
    """Run memory-safe LoRA SFT on verified exact 5-7-5 haikus using identical LoRA layers."""
    random.seed(seed)
    mx.random.seed(seed)
    if hasattr(mx, "set_cache_limit"):
        mx.set_cache_limit(2 * (1024**3))

    rows: list[dict] = []
    with Path(sft_file).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No SFT examples found in {sft_file}")

    run_dir = Path(runs_dir) / run_name
    adapters_dir = run_dir / "adapters"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model for LoRA SFT: {model_cfg.model_name} ({len(rows)} examples, {epochs} epochs)")
    model, tokenizer = build_lora_model(model_cfg)

    # Pre-tokenize all (prompt_ids, completion_ids + eos) pairs
    eos_id = tokenizer.eos_token_id
    tokenized: list[tuple[list[int], list[int]]] = []
    for row in rows:
        pt = format_chat_prompt(tokenizer, row["prompt"])
        pids = tokenizer.encode(pt, add_special_tokens=False)
        cids = tokenizer.encode(row["completion"], add_special_tokens=False)
        if eos_id is not None and (not cids or cids[-1] != eos_id):
            cids = cids + [eos_id]
        if pids and cids:
            tokenized.append((pids, cids))

    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    loss_and_grad_fn = nn.value_and_grad(model, sft_sequence_loss)

    rng = random.Random(seed)
    opt_step = 0
    t0 = time.perf_counter()

    with gradient_checkpointing(model, enabled=model_cfg.grad_checkpoint):
        for epoch in range(1, epochs + 1):
            indices = list(range(len(tokenized)))
            rng.shuffle(indices)
            epoch_losses: list[float] = []

            for batch_start in range(0, len(indices), grad_accum_steps):
                batch_idx = indices[batch_start : batch_start + grad_accum_steps]
                inv_n = 1.0 / len(batch_idx)
                accum_grads = None
                batch_loss = 0.0

                for idx in batch_idx:
                    pids, cids = tokenized[idx]
                    loss_val, grads = loss_and_grad_fn(model, pids, cids)
                    scaled = tree_map(lambda g: g * inv_n, grads)
                    if accum_grads is None:
                        accum_grads = scaled
                    else:
                        accum_grads = tree_map(lambda a, b: a + b, accum_grads, scaled)
                    mx.eval(accum_grads, loss_val)
                    batch_loss += float(loss_val.item()) * inv_n

                if accum_grads is not None:
                    accum_grads, grad_norm = clip_grad_norm(accum_grads, grad_clip_norm)
                    optimizer.update(model, accum_grads)
                    mx.eval(model.parameters(), optimizer.state)
                    opt_step += 1
                    epoch_losses.append(batch_loss)

                    if opt_step == 1 or opt_step % 10 == 0:
                        peak_gb = (
                            mx.get_peak_memory() / (1024**3)
                            if hasattr(mx, "get_peak_memory")
                            else 0.0
                        )
                        print(
                            f"  [sft epoch {epoch}/{epochs} step {opt_step:03d}] "
                            f"nll={batch_loss:.4f} gnorm={grad_norm:.2f} mem={peak_gb:.2f}GB"
                        )

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()

            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            print(
                f"Epoch {epoch}/{epochs} complete: mean_nll={mean_loss:.4f} "
                f"({time.perf_counter() - t0:.1f}s elapsed)"
            )

    final_path = save_adapter(model, model_cfg, adapters_dir / "final")
    print(f"SFT complete! LoRA adapter saved -> {final_path}")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rejection-sampling data collection + LoRA SFT for exact 5-7-5 haikus."
    )
    parser.add_argument("--model", type=str, default=ModelConfig.model_name)
    parser.add_argument("--train-file", type=str, default="data/train.jsonl")
    parser.add_argument("--sft-file", type=str, default="data/sft_haikus.jsonl")
    parser.add_argument("--collect-adapter", type=str, default=None, help="Optional adapter for sampling.")
    parser.add_argument("--skip-collect", action="store_true", help="Reuse existing --sft-file.")
    parser.add_argument("--collect-only", action="store_true", help="Only collect SFT data.")
    parser.add_argument("--num-prompts", type=int, default=450)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--run-name", type=str, default="sft_v1")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=ModelConfig.lora_rank)
    parser.add_argument("--lora-scale", type=float, default=ModelConfig.lora_scale)
    parser.add_argument("--lora-layers", type=int, default=ModelConfig.lora_layers)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.skip_collect:
        collect_sft_haikus(
            model_name=args.model,
            train_file=args.train_file,
            out_file=args.sft_file,
            adapter_path=args.collect_adapter,
            num_prompts=args.num_prompts,
            samples_per_prompt=args.samples_per_prompt,
            prompt_batch_size=args.prompt_batch_size,
            seed=args.seed,
        )

    if not args.collect_only:
        model_cfg = ModelConfig(
            model_name=args.model,
            lora_rank=args.lora_rank,
            lora_scale=args.lora_scale,
            lora_layers=args.lora_layers,
        )
        train_sft(
            model_cfg=model_cfg,
            sft_file=args.sft_file,
            run_name=args.run_name,
            runs_dir=args.runs_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            grad_accum_steps=args.grad_accum,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
