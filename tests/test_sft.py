"""Unit tests for rejection-sampling SFT filtering and masked completion-only SFT loss."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from grpo_haiku.rewards import score_completion
from grpo_haiku.sft import is_sft_worthy, sft_sequence_loss


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
