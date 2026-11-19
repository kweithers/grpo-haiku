"""Unit tests for tag parsing, shaped rewards, and anti-hacking guardrails."""

from __future__ import annotations

from grpo_haiku.config import RewardConfig
from grpo_haiku.relevance import has_prompt_echo
from grpo_haiku.rewards import parse_completion, score_completion, score_group


class DummyRelevanceScorer:
    """Deterministic stub scorer for fast offline unit tests."""

    def __init__(self, fixed_sim: float = 0.72) -> None:
        self.fixed_sim = fixed_sim

    def similarity(self, topic: str, haiku: str) -> float:
        return self.fixed_sim

    def batch_similarity(self, topic: str, haikus: list[str]) -> list[float]:
        return [self.fixed_sim if h.strip() else 0.0 for h in haikus]


def test_exact_valid_haiku_gets_full_shaped_reward():
    completion = (
        "<think>\n"
        "Line 1: Frosted cedar branch (2+2+1=5)\n"
        "Line 2: Quiet snowflakes drift below (2+2+1+2=7)\n"
        "Line 3: Dawn breaks pale and still (1+1+1+1+1=5)\n"
        "</think>\n"
        "<haiku>\n"
        "Frosted cedar branch\n"
        "Quiet snowflakes drift below\n"
        "Dawn breaks pale and still\n"
        "</haiku>"
    )
    res = score_completion(
        topic="a quiet winter snowy morning",
        completion=completion,
        relevance_scorer=DummyRelevanceScorer(0.78),
    )
    assert res.is_valid_format
    assert res.is_exact_575
    assert res.oov_words == ()
    assert res.format_reward == 0.30
    assert res.syllable_reward == 1.20
    assert res.exact_bonus == 1.00
    assert res.relevance_reward == 0.50
    assert res.oov_penalty == 0.0
    assert res.repetition_penalty == 0.0
    assert res.echo_penalty == 0.0
    assert res.total_reward == 3.00


def test_near_miss_gets_smooth_partial_credit():
    # Line 1: 6 syllables (off by 1), Line 2: 7 (exact), Line 3: 5 (exact)
    completion = (
        "<think>drafting</think>\n"
        "<haiku>\n"
        "Silver frosted branches\n"
        "Quiet snowflakes drift below\n"
        "Dawn breaks pale and still\n"
        "</haiku>"
    )
    res = score_completion(
        topic="a quiet winter snowy morning",
        completion=completion,
        relevance_scorer=DummyRelevanceScorer(0.60),
    )
    assert res.is_valid_format
    assert not res.is_exact_575
    assert res.exact_bonus == 0.0
    assert res.line_distances == (1, 0, 0)
    # 0.40*0.75 + 0.40 + 0.40 = 1.10
    assert abs(res.syllable_reward - 1.10) < 1e-5


def test_missing_haiku_tag_penalized():
    completion = "<think>Thinking about winter...</think>\nSnow falls on the pines"
    res = score_completion(
        topic="a quiet winter snowy morning",
        completion=completion,
        relevance_scorer=DummyRelevanceScorer(0.75),
    )
    assert not res.is_valid_format
    assert not res.is_exact_575
    assert res.total_reward < 0.0


def test_oov_blocks_jackpot_and_incurs_penalty():
    completion = (
        "<think>drafting</think>\n"
        "<haiku>\n"
        "Blorptastic cold snow\n"
        "Quiet snowflakes drift below\n"
        "Dawn breaks pale and still\n"
        "</haiku>"
    )
    res = score_completion(
        topic="a quiet winter snowy morning",
        completion=completion,
        relevance_scorer=DummyRelevanceScorer(0.70),
    )
    assert not res.is_exact_575
    assert res.exact_bonus == 0.0
    assert res.oov_penalty == -0.20
    assert res.oov_words == ("blorptastic",)


def test_duplicate_lines_and_monosyllable_monotony_penalized():
    completion = (
        "<think>drafting</think>\n"
        "<haiku>\n"
        "The cold sun shines bright\n"
        "The cold wind blows in the sky\n"
        "The cold sun shines bright\n"
        "</haiku>"
    )
    res = score_completion(
        topic="a quiet winter snowy morning",
        completion=completion,
        relevance_scorer=DummyRelevanceScorer(0.65),
    )
    assert not res.is_exact_575  # duplicate lines block jackpot
    assert res.repetition_penalty <= -0.40


def test_prompt_echo_detected_and_penalized():
    topic = "a quiet winter snowy morning"
    assert has_prompt_echo(topic, "A winter snowy morning shines\nSoft ice on the roof\nCold breath in the air")
    assert not has_prompt_echo(topic, "Frosted cedar branch\nQuiet snowflakes drift below\nDawn breaks pale and still")

    echo_completion = (
        "<think>drafting</think>\n"
        "<haiku>\n"
        "Cold breath in the air\n"
        "A winter snowy morning\n"
        "Pine trees hold the frost\n"
        "</haiku>"
    )
    res = score_completion(
        topic=topic,
        completion=echo_completion,
        relevance_scorer=DummyRelevanceScorer(0.85),
    )
    assert res.echo_penalty == RewardConfig().prompt_echo_penalty


def test_fewshot_copy_and_off_topic_cannot_farm_jackpot():
    # Copying the 1-shot autumn-leaves example is heavily penalized and denied exact jackpot
    copied = (
        "<think>L1: Crisp red leaves drift down</think>\n"
        "<haiku>\n"
        "Crisp red leaves drift down\n"
        "Cold wind sweeps the quiet lane\n"
        "Dusk cools the damp earth\n"
        "</haiku>"
    )
    res_copy = score_completion(
        topic="a simmering skillet of roasted chestnuts",
        completion=copied,
        relevance_scorer=DummyRelevanceScorer(0.42),
    )
    assert not res_copy.is_exact_575
    assert res_copy.exact_bonus == 0.0
    assert res_copy.echo_penalty == RewardConfig().fewshot_copy_penalty

    # Original 5-7-5 haiku that is off-topic (sim < exact_bonus_min_sim) also does not get jackpot
    off_topic = (
        "<think>drafting</think>\n"
        "<haiku>\n"
        "Frosted cedar branch\n"
        "Quiet snowflakes drift below\n"
        "Dawn breaks pale and still\n"
        "</haiku>"
    )
    res_off = score_completion(
        topic="a simmering skillet of roasted chestnuts",
        completion=off_topic,
        relevance_scorer=DummyRelevanceScorer(0.40),
    )
    assert res_off.is_exact_575
    assert res_off.exact_bonus == 0.0


def test_score_group_batches_cleanly():
    completions = [
        "<think>ok</think><haiku>\nAn old silent pond\nA frog jumps into the pond\nSplash silence again\n</haiku>",
        "<think>oops</think>no haiku here",
    ]
    results = score_group("a frog in a quiet pond", completions, relevance_scorer=DummyRelevanceScorer(0.70))
    assert len(results) == 2
    assert results[0].total_reward > results[1].total_reward
    assert parse_completion(completions[0]).think_before_haiku
