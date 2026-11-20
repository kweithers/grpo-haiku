"""Unit tests for direct 3-line haiku parsing, shaped rewards, and anti-hacking guardrails."""

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


def test_exact_valid_haiku_gets_full_shaped_reward_with_zero_tag_bonus():
    completion = (
        "Frosted cedar branch\n"
        "Quiet snowflakes drift below\n"
        "Dawn breaks pale and still"
    )
    res = score_completion(
        topic="a quiet winter snowy morning",
        completion=completion,
        relevance_scorer=DummyRelevanceScorer(0.78),
    )
    assert res.is_valid_format
    assert res.is_exact_575
    assert res.oov_words == ()
    assert res.format_reward == 0.0
    assert res.syllable_reward == 1.20
    assert res.exact_bonus == 2.00
    assert res.relevance_reward == 0.40
    assert res.oov_penalty == 0.0
    assert res.repetition_penalty == 0.0
    assert res.echo_penalty == 0.0
    assert res.total_reward == 3.60


def test_near_miss_gets_sharpened_partial_credit():
    # Line 1: 6 syllables (off by 1), Line 2: 7 (exact), Line 3: 5 (exact)
    completion = (
        "Silver frosted branches\n"
        "Quiet snowflakes drift below\n"
        "Dawn breaks pale and still"
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
    # 0.40 * (1 - 0.45) + 0.40 + 0.40 = 0.22 + 0.80 = 1.02
    assert abs(res.syllable_reward - 1.02) < 1e-5


def test_wrong_line_count_and_stray_tags_penalized():
    empty_res = score_completion(
        topic="a quiet winter snowy morning",
        completion="   \n  ",
        relevance_scorer=DummyRelevanceScorer(0.75),
    )
    assert not empty_res.is_valid_format
    assert empty_res.total_reward == RewardConfig().empty_output_penalty

    four_lines = (
        "Here is your haiku:\n"
        "Frosted cedar branch\n"
        "Quiet snowflakes drift below\n"
        "Dawn breaks pale and still"
    )
    res4 = score_completion(
        topic="a quiet winter snowy morning",
        completion=four_lines,
        relevance_scorer=DummyRelevanceScorer(0.75),
    )
    assert not res4.is_valid_format
    assert not res4.is_exact_575
    assert res4.format_reward == -0.35

    stray_tags = (
        "<haiku>\n"
        "Frosted cedar branch\n"
        "Quiet snowflakes drift below\n"
        "Dawn breaks pale and still\n"
        "</haiku>"
    )
    res_tags = score_completion(
        topic="a quiet winter snowy morning",
        completion=stray_tags,
        relevance_scorer=DummyRelevanceScorer(0.75),
    )
    assert not res_tags.is_valid_format
    assert res_tags.format_reward < 0.0


def test_oov_blocks_jackpot_and_incurs_penalty():
    completion = (
        "Blorptastic cold snow\n"
        "Quiet snowflakes drift below\n"
        "Dawn breaks pale and still"
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
        "The cold sun shines bright\n"
        "The cold wind blows in the sky\n"
        "The cold sun shines bright"
    )
    res = score_completion(
        topic="a quiet winter snowy morning",
        completion=completion,
        relevance_scorer=DummyRelevanceScorer(0.65),
    )
    assert not res.is_exact_575
    assert res.repetition_penalty <= -0.40


def test_prompt_echo_and_off_topic_jackpot_gate():
    topic = "a quiet winter snowy morning"
    assert has_prompt_echo(topic, "A winter snowy morning shines\nSoft ice on the roof\nCold breath in the air")
    assert not has_prompt_echo(topic, "Frosted cedar branch\nQuiet snowflakes drift below\nDawn breaks pale and still")

    echo_completion = (
        "Cold breath in the air\n"
        "A winter snowy morning\n"
        "Pine trees hold the frost"
    )
    res = score_completion(
        topic=topic,
        completion=echo_completion,
        relevance_scorer=DummyRelevanceScorer(0.85),
    )
    assert res.echo_penalty == RewardConfig().prompt_echo_penalty

    off_topic = (
        "Frosted cedar branch\n"
        "Quiet snowflakes drift below\n"
        "Dawn breaks pale and still"
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
        "An old silent pond\nA frog jumps into the pond\nSplash silence again",
        "just one line",
    ]
    results = score_group("a frog in a quiet pond", completions, relevance_scorer=DummyRelevanceScorer(0.70))
    assert len(results) == 2
    assert results[0].total_reward > results[1].total_reward
    assert len(parse_completion(completions[0]).lines) == 3
