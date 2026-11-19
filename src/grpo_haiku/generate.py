"""Interactive / CLI haiku generator with live CMUdict syllable verification."""

from __future__ import annotations

import argparse

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from grpo_haiku.config import ModelConfig, RewardConfig
from grpo_haiku.data import format_chat_prompt
from grpo_haiku.grpo import normalize_completion_text
from grpo_haiku.relevance import MLXEmbeddingRelevanceScorer
from grpo_haiku.rewards import score_completion
from grpo_haiku.syllables import analyze_line


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and verify a haiku on any topic.")
    parser.add_argument("--model", type=str, default=ModelConfig.model_name)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--topic", type=str, default="a winter snowy morning")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--no-relevance", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    user_prompt = f"Write a haiku about {args.topic}."
    prompt_text = format_chat_prompt(tokenizer, user_prompt)

    sampler = make_sampler(temp=args.temp, top_p=args.top_p)
    raw_out = generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=args.max_tokens,
        sampler=sampler,
        verbose=False,
    )
    completion = normalize_completion_text(prompt_text, raw_out)

    reward_cfg = RewardConfig(relevance_weight=0.0 if args.no_relevance else RewardConfig.relevance_weight)
    scorer = (
        None
        if args.no_relevance
        else MLXEmbeddingRelevanceScorer(reward_cfg.embedding_model)
    )
    bd = score_completion(args.topic, completion, cfg=reward_cfg, relevance_scorer=scorer)

    print(f"\n=== Prompt: {user_prompt} ===")
    if bd.think_text:
        print("\n<think>")
        print(bd.think_text)
        print("</think>")

    print("\n<haiku>")
    targets = reward_cfg.line_targets
    for i, line in enumerate(bd.haiku_lines):
        analysis = analyze_line(line)
        target = targets[i] if i < len(targets) else None
        counts_str = sorted(analysis.possible_counts)
        oov_str = f" OOV={list(analysis.oov_words)}" if analysis.oov_words else ""
        mark = "✓" if (target is not None and analysis.matches_exact(target)) else "✗"
        print(f"  {mark} {line:<45s}  [syllables={counts_str}, target={target}{oov_str}]")
    if not bd.haiku_lines:
        print(f"  [unparsed raw output]\n{completion}")
    print("</haiku>")

    print(
        f"\nVerdict: {'EXACT 5-7-5 ✓' if bd.is_exact_575 else 'NOT EXACT ✗'} | "
        f"total={bd.total_reward:+.3f} (fmt={bd.format_reward:+.2f}, syl={bd.syllable_reward:+.2f}, "
        f"exact={bd.exact_bonus:+.2f}, rel={bd.relevance_reward:+.2f} [sim={bd.raw_cosine_sim:.2f}], "
        f"oov={bd.oov_penalty:+.2f}, rep={bd.repetition_penalty:+.2f}, echo={bd.echo_penalty:+.2f})"
    )


if __name__ == "__main__":
    main()
