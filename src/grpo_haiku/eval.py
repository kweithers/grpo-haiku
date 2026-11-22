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
from grpo_haiku.rewards import RewardBreakdown, analyze_think_counts, score_group
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
    max_tokens: int | None = None,
    batch_size: int = 8,
    limit: int | None = None,
    relevance_weight: float = RewardConfig.relevance_weight,
    thinking_mode: bool = False,
    think_format: str = "sum",
) -> dict:
    """Evaluate base or LoRA-adapted model on held-out haiku prompts."""
    t0 = time.perf_counter()
    default_max = (256 if think_format == "cumulative" else 224) if thinking_mode else 64
    resolved_max_tokens = max_tokens if max_tokens is not None else default_max
    examples = load_jsonl(eval_file)
    if limit is not None and limit > 0:
        examples = examples[:limit]

    print(
        f"Evaluating model={model_name} adapter={adapter_path or 'NONE'} "
        f"thinking={thinking_mode} ({think_format}) max_tokens={resolved_max_tokens} on {len(examples)} prompts..."
    )
    model, tokenizer = load(model_name, adapter_path=adapter_path)
    reward_cfg = RewardConfig(
        relevance_weight=relevance_weight,
        thinking_mode=thinking_mode,
        think_format=think_format,
    )
    scorer = (
        MLXEmbeddingRelevanceScorer(reward_cfg.embedding_model)
        if relevance_weight > 0.0
        else None
    )

    sampler = make_sampler(temp=0.0)
    records: list[dict] = []
    breakdowns: list[RewardBreakdown] = []
    think_token_counts: list[int] = []
    think_honest_hits = 0
    think_arith_hits = 0
    think_revision_hits = 0

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prompt_texts = [
            format_chat_prompt(
                tokenizer,
                ex.prompt,
                thinking_mode=thinking_mode,
                think_format=think_format,
            )
            for ex in chunk
        ]
        prompt_ids = [tokenizer.encode(pt, add_special_tokens=False) for pt in prompt_texts]

        resp = batch_generate(
            model,
            tokenizer,
            prompts=prompt_ids,
            max_tokens=resolved_max_tokens,
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
            rec = {
                "id": ex.id,
                "domain": ex.domain,
                "topic": ex.topic,
                "prompt": ex.prompt,
                "completion": completion,
                "breakdown": asdict(bd),
            }
            if thinking_mode:
                diag = analyze_think_counts(
                    bd.think_text,
                    targets=reward_cfg.line_targets,
                    think_format=think_format,
                )
                rec["think_diag"] = diag
                think_toks = (
                    len(tokenizer.encode(bd.think_text, add_special_tokens=False))
                    if bd.think_text
                    else 0
                )
                think_token_counts.append(think_toks)
                if diag["is_honest_575"]:
                    think_honest_hits += 1
                if diag["arith_consistent"]:
                    think_arith_hits += 1
                if diag["has_revision"]:
                    think_revision_hits += 1
            records.append(rec)

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
        "thinking_mode": thinking_mode,
        "think_format": think_format if thinking_mode else None,
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
    if thinking_mode:
        summary["mean_think_grounding"] = round(
            sum(b.think_grounding for b in breakdowns) / n, 4
        )
        summary["mean_think_tokens"] = round(
            sum(think_token_counts) / max(1, len(think_token_counts)), 1
        )
        summary["think_count_accuracy"] = round(think_honest_hits / n, 4)
        summary["think_arith_consistency"] = round(think_arith_hits / n, 4)
        summary["think_revision_rate"] = round(think_revision_hits / n, 4)
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

    if s.get("thinking_mode"):
        print(
            f"\n  <think> diagnostics ({s.get('think_format', 'sum')}): "
            f"grounding={s.get('mean_think_grounding', 0.0):.3f} | "
            f"avg_think_tokens={s.get('mean_think_tokens', 0.0):.1f} | "
            f"arith_ok={s.get('think_arith_consistency', 0.0)*100:.1f}% | "
            f"honest_575_counts={s.get('think_count_accuracy', 0.0)*100:.1f}% | "
            f"revision_rate={s.get('think_revision_rate', 0.0)*100:.1f}%"
        )

    print("\n=== Sample Outputs (first 5) ===")
    for rec in result["records"][:5]:
        bd = rec["breakdown"]
        status = "EXACT 5-7-5" if bd["is_exact_575"] else f"counts={bd['line_counts']}"
        print(f"\nTopic: {rec['topic']}  [{status} | r={bd['total_reward']:+.2f} | sim={bd['raw_cosine_sim']:.2f}]")
        if s.get("thinking_mode") and bd.get("think_text"):
            for tl in bd["think_text"].splitlines():
                print(f"  [think] {tl}")
        if bd["haiku_lines"]:
            for line in bd["haiku_lines"]:
                print(f"  {line}")
        else:
            print("  [no valid haiku block]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base or LoRA-adapted haiku model on held-out prompts.")
    parser.add_argument("--model", type=str, default=ModelConfig.model_name)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--eval-file", type=str, default="data/eval.jsonl")
    parser.add_argument("--out", type=str, default=None, help="Path to write JSON eval results.")
    parser.add_argument("--compare-with", type=str, default=None, help="Path to baseline eval JSON.")
    parser.add_argument("--thinking-mode", action="store_true")
    parser.add_argument("--think-format", type=str, default="sum", choices=["sum", "cumulative"])
    parser.add_argument("--max-tokens", type=int, default=None)
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
        thinking_mode=args.thinking_mode,
        think_format=args.think_format,
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
