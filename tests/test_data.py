"""Unit tests for deterministic synthetic haiku dataset generation."""

from __future__ import annotations

from collections import Counter

from grpo_haiku.data import (
    DOMAIN_SPECS,
    THINK_CUMULATIVE_SYSTEM_PROMPT,
    THINK_SYSTEM_PROMPT,
    format_chat_prompt,
    generate_dataset,
    load_jsonl,
    save_jsonl,
)


def test_dataset_sizes_determinism_and_no_leakage():
    train1, eval1 = generate_dataset(seed=42, eval_per_domain=10)
    train2, eval2 = generate_dataset(seed=42, eval_per_domain=10)

    assert len(train1) == 900
    assert len(eval1) == 100
    assert train1 == train2
    assert eval1 == eval2

    train_topics = {ex.topic for ex in train1}
    eval_topics = {ex.topic for ex in eval1}
    assert len(train_topics) == 900
    assert len(eval_topics) == 100
    assert train_topics.isdisjoint(eval_topics)


def test_stratified_domain_balance():
    train, eval_set = generate_dataset(seed=42, eval_per_domain=10)
    train_counts = Counter(ex.domain for ex in train)
    eval_counts = Counter(ex.domain for ex in eval_set)

    assert set(train_counts.keys()) == set(DOMAIN_SPECS.keys())
    assert set(eval_counts.keys()) == set(DOMAIN_SPECS.keys())
    assert all(c == 90 for c in train_counts.values())
    assert all(c == 10 for c in eval_counts.values())


def test_jsonl_roundtrip(tmp_path):
    train, eval_set = generate_dataset(seed=42, eval_per_domain=10)
    p = tmp_path / "eval.jsonl"
    save_jsonl(eval_set, p)
    loaded = load_jsonl(p)
    assert loaded == eval_set


def test_format_chat_prompt_thinking_mode_strips_empty_qwen_stub():
    class FakeQwenTokenizer:
        chat_template = "qwen3"

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ):
            assert enable_thinking is False
            sys_txt = messages[0]["content"]
            usr_txt = messages[1]["content"]
            return (
                f"<|im_start|>system\n{sys_txt}<|im_end|>\n"
                f"<|im_start|>user\n{usr_txt}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )

    tok = FakeQwenTokenizer()
    direct = format_chat_prompt(tok, "Write a haiku about snow.", thinking_mode=False)
    assert direct.endswith("<think>\n\n</think>\n\n")

    think = format_chat_prompt(tok, "Write a haiku about snow.", thinking_mode=True)
    assert THINK_SYSTEM_PROMPT in think
    assert think.endswith("<|im_start|>assistant\n")
    assert "<think>\n\n</think>" not in think

    cum = format_chat_prompt(
        tok, "Write a haiku about snow.", thinking_mode=True, think_format="cumulative"
    )
    assert THINK_CUMULATIVE_SYSTEM_PROMPT in cum
    assert cum.endswith("<|im_start|>assistant\n")
    assert "<think>\n\n</think>" not in cum
