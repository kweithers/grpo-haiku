"""Unit tests for rejection-sampling SFT filtering and masked completion-only SFT loss."""

from __future__ import annotations

import json
import random

import mlx.core as mx
import mlx.nn as nn

from grpo_haiku.config import RewardConfig
from grpo_haiku.rewards import analyze_think_counts, score_completion
from grpo_haiku.sft import (
    format_line_syllable_trace,
    is_sft_worthy,
    reformat_think_sft_jsonl,
    sft_sequence_loss,
    synthesize_think_trace,
)


class _StubScorer:
    def __init__(self, sim: float) -> None:
        self.sim = sim

    def similarity(self, topic: str, haiku: str) -> float:
        return self.sim

    def batch_similarity(self, topic: str, haikus: list[str]) -> list[float]:
        return [self.sim for _ in haikus]


def test_is_sft_worthy_accepts_only_clean_on_topic_exact_575():
    good = "Frosted cedar branch\nQuiet snowflakes drift below\nDawn breaks pale and still"
    bd_good = score_completion("a winter snowy morning", good, relevance_scorer=_StubScorer(0.72))
    assert is_sft_worthy(bd_good)

    # Off-topic exact 5-7-5 rejected (sim < 0.42)
    bd_off = score_completion("a winter snowy morning", good, relevance_scorer=_StubScorer(0.35))
    assert not is_sft_worthy(bd_off)

    # Near-miss rejected
    near = "Silver frosted branches\nQuiet snowflakes drift below\nDawn breaks pale and still"
    bd_near = score_completion("a winter snowy morning", near, relevance_scorer=_StubScorer(0.75))
    assert not is_sft_worthy(bd_near)

    # Prompt echo rejected even if 5-7-5
    echo = "Cold breath in the air\nA winter snowy morning\nPine trees hold the frost"
    bd_echo = score_completion("a winter snowy morning", echo, relevance_scorer=_StubScorer(0.80))
    assert not is_sft_worthy(bd_echo)


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, dim: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.head(self.emb(tokens))


def test_sft_sequence_loss_runs_and_decreases_with_gradient_step():
    mx.random.seed(0)
    lm = _TinyLM()
    prompt_ids = [1, 2, 3]
    comp_ids = [4, 5, 6, 7]

    loss_fn = nn.value_and_grad(lm, sft_sequence_loss)
    l0, grads = loss_fn(lm, prompt_ids, comp_ids)
    mx.eval(l0, grads)
    assert float(l0.item()) > 0.0

    # Apply one small manual SGD step and verify NLL drops
    for k, g in grads.items():
        if isinstance(g, dict):
            for sk, sg in g.items():
                mod = getattr(lm, k)
                setattr(mod, sk, getattr(mod, sk) - 0.1 * sg)
    l1 = sft_sequence_loss(lm, prompt_ids, comp_ids)
    mx.eval(l1)
    assert float(l1.item()) < float(l0.item())


def test_synthesize_think_trace_exact_and_multi_pronunciation():
    # "fire" can be 1 or 2 syllables; in "Warm fire glows at dusk" -> Warm(1) fire(1) glows(1) at(1) dusk(1) = 5
    # In "Cold fire burns bright" -> Cold(1) fire(2) burns(1) bright(1) = 5
    tr1 = format_line_syllable_trace("Warm fire glows at dusk", target=5)
    tr2 = format_line_syllable_trace("Cold fire burns bright", target=5)
    assert tr1 is not None and tr1[1] == 5 and "fire(1)" in tr1[0]
    assert tr2 is not None and tr2[1] == 5 and "fire(2)" in tr2[0]

    lines = (
        "Frost bites the cold air",
        "Soft snowflakes drift to the ground",
        "Dawn breaks pale and still",
    )
    out = synthesize_think_trace(lines, revision_prob=0.0, rng=random.Random(1))
    assert "snowflakes(2)" in out
    assert "snow-flakes" not in out

    cfg = RewardConfig(thinking_mode=True)
    bd = score_completion("a winter snowy morning", out, cfg=cfg, relevance_scorer=_StubScorer(0.75))
    assert bd.is_valid_format
    assert bd.is_exact_575
    assert bd.think_grounding == 1.0
    assert bd.exact_bonus == cfg.exact_575_bonus

    diag = analyze_think_counts(bd.think_text)
    assert diag["is_honest_575"]
    assert not diag["has_revision"]


def test_synthesize_think_trace_revisions_from_real_draft_and_perturbation():
    lines = (
        "Frost bites the cold air",
        "Soft snowflakes drift to the ground",
        "Dawn breaks pale and still",
    )
    # 1) Real near-miss line (6 syllables -> 7 syllables)
    out_real = synthesize_think_trace(
        lines,
        draft_miss_lines={1: "Soft snowflakes drift down slow"},
        revision_prob=1.0,
        rng=random.Random(7),
    )
    assert "->" in out_real
    assert "= 6 ->" in out_real
    assert "= 7" in out_real
    cfg = RewardConfig(thinking_mode=True)
    bd_real = score_completion("a winter snowy morning", out_real, cfg=cfg, relevance_scorer=_StubScorer(0.75))
    assert bd_real.is_exact_575 and bd_real.think_grounding == 1.0
    diag_real = analyze_think_counts(bd_real.think_text)
    assert diag_real["is_honest_575"] and diag_real["has_revision"]

    # 2) Fallback deterministic perturbation (no real near-miss passed)
    out_pert = synthesize_think_trace(
        lines,
        draft_miss_lines=None,
        revision_prob=1.0,
        rng=random.Random(3),
    )
    assert "->" in out_pert
    bd_pert = score_completion("a winter snowy morning", out_pert, cfg=cfg, relevance_scorer=_StubScorer(0.75))
    assert bd_pert.is_exact_575 and bd_pert.think_grounding == 1.0
    diag_pert = analyze_think_counts(bd_pert.think_text)
    assert diag_pert["is_honest_575"] and diag_pert["has_revision"]


def test_cumulative_think_synthesizer_and_reformat_roundtrip(tmp_path):
    tr = format_line_syllable_trace("Frosted cedar branch", target=5, think_format="cumulative")
    assert tr == ("Frosted(2)=2 cedar(2)=4 branch(1)=5", 5)

    lines = (
        "Frosted cedar branch",
        "Quiet snowflakes drift below",
        "Dawn breaks pale and still",
    )
    out = synthesize_think_trace(
        lines,
        draft_miss_lines={1: "Quiet snowflakes drift slow"},
        revision_prob=1.0,
        think_format="cumulative",
        forced_rev_line=1,
        rng=random.Random(1),
    )
    assert "1: Frosted(2)=2 cedar(2)=4 branch(1)=5" in out
    assert "2: Quiet(2)=2 snowflakes(2)=4 drift(1)=5 slow(1)=6 -> Quiet(2)=2 snowflakes(2)=4 drift(1)=5 below(2)=7" in out
    assert "3: Dawn(1)=1 breaks(1)=2 pale(1)=3 and(1)=4 still(1)=5" in out

    cfg = RewardConfig(thinking_mode=True, think_format="cumulative", think_arith_penalty=-0.20)
    bd = score_completion("a winter snowy morning", out, cfg=cfg, relevance_scorer=_StubScorer(0.75))
    assert bd.is_valid_format and bd.is_exact_575 and bd.think_penalty == 0.0
    diag = analyze_think_counts(bd.think_text, think_format="cumulative")
    assert diag["is_honest_575"] and diag["arith_consistent"] and diag["has_revision"]

    # Roundtrip reformat from classic "sum" SFT JSONL preserves real near-miss draft verbatim
    src_sum = synthesize_think_trace(
        lines,
        draft_miss_lines={1: "Quiet snowflakes drift slow"},
        revision_prob=1.0,
        think_format="sum",
        forced_rev_line=1,
        rng=random.Random(1),
    )
    src_no_rev = synthesize_think_trace(lines, revision_prob=0.0, think_format="sum")
    src_file = tmp_path / "src_sum.jsonl"
    out_file = tmp_path / "out_cum.jsonl"
    with src_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "a", "prompt": "p1", "topic": "t1", "completion": src_sum, "haiku": "\n".join(lines), "raw_cosine_sim": 0.72}) + "\n")
        f.write(json.dumps({"id": "b", "prompt": "p2", "topic": "t2", "completion": src_no_rev, "haiku": "\n".join(lines), "raw_cosine_sim": 0.74}) + "\n")

    rows = reformat_think_sft_jsonl(str(src_file), str(out_file), think_format="cumulative")
    assert len(rows) == 2
    assert "Quiet(2)=2 snowflakes(2)=4 drift(1)=5 slow(1)=6 -> Quiet(2)=2 snowflakes(2)=4 drift(1)=5 below(2)=7" in rows[0]["completion"]
    assert "->" not in rows[1]["completion"]
    assert "1: Frosted(2)=2 cedar(2)=4 branch(1)=5" in rows[1]["completion"]
    for row in rows:
        bd_r = score_completion("a winter snowy morning", row["completion"], cfg=cfg, relevance_scorer=_StubScorer(0.75))
        assert bd_r.is_exact_575 and bd_r.think_penalty == 0.0
        assert analyze_think_counts(bd_r.think_text, think_format="cumulative")["is_honest_575"]
