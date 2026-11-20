"""Held-out evaluation harness & before/after comparison report for GRPO haikus."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from grpo_haiku.config import ModelConfig, RewardConfig
from grpo_haiku.data import format_chat_prompt, load_jsonl
from grpo_haiku.grpo import normalize_completion_text
from grpo_haiku.relevance import MLXEmbeddingRelevanceScorer
from grpo_haiku.rewards import RewardBreakdown, score_group
from grpo_haiku.syllables import tokenize_words


def compute_diversity_metrics(haikus: list[str]) -> dict[str, float]:
    """Measure output diversity across held-out prompts to verify zero mode collapse."""
    non_empty = [h.strip().lower() for h in haikus if h.strip()]
    if not non_empty:
        return {"unique_haiku_ratio": 0.0, "distinct_2": 0.0}

    unique_ratio = len(set(non_empty)) / len(non_empty)

    bigrams: list[tuple[str, str]] = []
    for h in non_empty:
        words = tokenize_words(h)
        for i in range(len(words) - 1):
            bigrams.append((words[i], words[i + 1]))

    distinct_2 = (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    return {
        "unique_haiku_ratio": round(unique_ratio, 4),
        "distinct_2": round(distinct_2, 4),
    }


def evaluate_model(
    model_name: str,
    eval_file: str,
    adapter_path: str | None = None,
    max_tokens: int = 64,
    batch_size: int = 8,
    limit: int | None = None,
    relevance_weight: float = RewardConfig.relevance_weight,
) -> dict:
    """Evaluate base or LoRA-adapted model on held-out haiku prompts."""
    t0 = time.perf_counter()
    examples = load_jsonl(eval_file)
    if limit is not None and limit > 0:
        examples = examples[:limit]

    print(f"Evaluating model={model_name} adapter={adapter_path or 'NONE'} on {len(examples)} prompts...")
    model, tokenizer = load(model_name, adapter_path=adapter_path)
    reward_cfg = RewardConfig(relevance_weight=relevance_weight)
    scorer = (
        MLXEmbeddingRelevanceScorer(reward_cfg.embedding_model)
        if relevance_weight > 0.0
        else None
    )

    sampler = make_sampler(temp=0.0)
    records: list[dict] = []
    breakdowns: list[RewardBreakdown] = []

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prompt_texts = [format_chat_prompt(tokenizer, ex.prompt) for ex in chunk]
        prompt_ids = [tokenizer.encode(pt, add_special_tokens=False) for pt in prompt_texts]

        resp = batch_generate(
            model,
            tokenizer,
            prompts=prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            return_token_ids=False,
        )

        for ex, pt, raw_text in zip(chunk, prompt_texts, resp.texts):
            completion = normalize_completion_text(pt, raw_text)
            bd = score_group(
                topic=ex.topic,
                completions=[completion],
                cfg=reward_cfg,
                relevance_scorer=scorer,
            )[0]
            breakdowns.append(bd)
            records.append(
                {
                    "id": ex.id,
                    "domain": ex.domain,
                    "topic": ex.topic,
                    "prompt": ex.prompt,
                    "completion": completion,
                    "breakdown": asdict(bd),
                }
            )

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    n = max(1, len(breakdowns))
    haiku_texts = ["\n".join(b.haiku_lines) for b in breakdowns]
    diversity = compute_diversity_metrics(haiku_texts)

    line1_acc = sum(1 for b in breakdowns if len(b.line_distances) >= 1 and b.line_distances[0] == 0) / n
    line2_acc = sum(1 for b in breakdowns if len(b.line_distances) >= 2 and b.line_distances[1] == 0) / n
    line3_acc = sum(1 for b in breakdowns if len(b.line_distances) >= 3 and b.line_distances[2] == 0) / n

    summary = {
        "model": model_name,
        "adapter_path": adapter_path,
        "num_examples": len(breakdowns),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "exact_575_rate": round(sum(1 for b in breakdowns if b.is_exact_575) / n, 4),
        "valid_format_rate": round(sum(1 for b in breakdowns if b.is_valid_format) / n, 4),
        "line1_exact_rate": round(line1_acc, 4),
        "line2_exact_rate": round(line2_acc, 4),
        "line3_exact_rate": round(line3_acc, 4),
        "mean_total_reward": round(sum(b.total_reward for b in breakdowns) / n, 4),
        "mean_syllable_reward": round(sum(b.syllable_reward for b in breakdowns) / n, 4),
        "mean_relevance_sim": round(sum(b.raw_cosine_sim for b in breakdowns) / n, 4),
        "oov_rate": round(sum(1 for b in breakdowns if b.oov_words) / n, 4),
        "echo_rate": round(sum(1 for b in breakdowns if b.echo_penalty < 0.0) / n, 4),
        "repetition_penalty_rate": round(
            sum(1 for b in breakdowns if b.repetition_penalty < 0.0) / n, 4
        ),
        **diversity,
    }
    return {"summary": summary, "records": records}


def print_summary(result: dict, baseline: dict | None = None) -> None:
    s = result["summary"]
    print("\n=== Held-Out Eval Summary ===")
    if baseline is None:
        for k, v in s.items():
            print(f"  {k:26s}: {v}")
    else:
        b = baseline["summary"]
        metrics = [
            ("exact_575_rate", True),
            ("valid_format_rate", True),
            ("line1_exact_rate", True),
            ("line2_exact_rate", True),
            ("line3_exact_rate", True),
            ("mean_total_reward", False),
            ("mean_relevance_sim", False),
            ("oov_rate", True),
            ("echo_rate", True),
            ("unique_haiku_ratio", True),
            ("distinct_2", False),
        ]
        print(f"{'Metric':26s} | {'Before':>10s} | {'After':>10s} | {'Delta':>10s}")
        print("-" * 66)
        for key, is_pct in metrics:
            bv = float(b.get(key, 0.0))
            av = float(s.get(key, 0.0))
            delta = av - bv
            if is_pct:
                print(f"{key:26s} | {bv*100:9.1f}% | {av*100:9.1f}% | {delta*100:+9.1f}%")
            else:
                print(f"{key:26s} | {bv:10.4f} | {av:10.4f} | {delta:+10.4f}")

    print("\n=== Sample Outputs (first 5) ===")
    for rec in result["records"][:5]:
        bd = rec["breakdown"]
        status = "EXACT 5-7-5" if bd["is_exact_575"] else f"counts={bd['line_counts']}"
        print(f"\nTopic: {rec['topic']}  [{status} | r={bd['total_reward']:+.2f} | sim={bd['raw_cosine_sim']:.2f}]")
        if bd["haiku_lines"]:
            for line in bd["haiku_lines"]:
                print(f"  {line}")
        else:
            print("  [no valid <haiku> block]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base or LoRA-adapted haiku model on held-out prompts.")
    parser.add_argument("--model", type=str, default=ModelConfig.model_name)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--eval-file", type=str, default="data/eval.jsonl")
    parser.add_argument("--out", type=str, default=None, help="Path to write JSON eval results.")
    parser.add_argument("--compare-with", type=str, default=None, help="Path to baseline eval JSON.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of eval prompts.")
    parser.add_argument("--relevance-weight", type=float, default=RewardConfig.relevance_weight)
    args = parser.parse_args()

    result = evaluate_model(
        model_name=args.model,
        eval_file=args.eval_file,
        adapter_path=args.adapter_path,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        limit=args.limit,
        relevance_weight=args.relevance_weight,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved eval report -> {out_path}")

    baseline = None
    if args.compare_with:
        with Path(args.compare_with).open("r", encoding="utf-8") as f:
            baseline = json.load(f)

    print_summary(result, baseline=baseline)


if __name__ == "__main__":
    main()
