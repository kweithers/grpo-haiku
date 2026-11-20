"""Unit tests for direct 3-line haiku parsing, shaped rewards, and anti-hacking guardrails."""

from __future__ import annotations

from grpo_haiku.config import RewardConfig
from grpo_haiku.relevance import has_prompt_echo
from grpo_haiku.rewards import (
    ClicheTracker,
    extract_content_bigrams,
    parse_completion,
    score_completion,
    score_group,
)


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
    assert res.syllable_reward == 1.80
    assert res.exact_bonus == 2.50
    assert res.relevance_reward == 0.20
    assert res.oov_penalty == 0.0
    assert res.repetition_penalty == 0.0
    assert res.echo_penalty == 0.0
    assert res.total_reward == 4.50


def test_near_miss_gets_sharpened_partial_credit_and_beats_worse_syllables_with_max_theme():
    # (6, 7, 5): Line 1 off by 1, Line 2 exact, Line 3 exact, moderate relevance
    better_syl = (
        "Silver frosted branches\n"
        "Quiet snowflakes drift below\n"
        "Dawn breaks pale and still"
    )
    res_better = score_completion(
        topic="a quiet winter snowy morning",
        completion=better_syl,
        relevance_scorer=DummyRelevanceScorer(0.60),
    )
    assert res_better.is_valid_format
    assert not res_better.is_exact_575
    assert res_better.exact_bonus == 0.0
    assert res_better.line_distances == (1, 0, 0)
    # 0.60 * (1 - 0.55) + 0.60 + 0.60 = 0.27 + 1.20 = 1.47
    assert abs(res_better.syllable_reward - 1.47) < 1e-5

    # (6, 7, 6): Two lines off by 1, even with MAX relevance (0.85), must lose to (6, 7, 5)
    worse_syl = (
        "Silver frosted branches\n"
        "Quiet snowflakes drift below\n"
        "Morning breaks pale and still"
    )
    res_worse = score_completion(
        topic="a quiet winter snowy morning",
        completion=worse_syl,
        relevance_scorer=DummyRelevanceScorer(0.85),
    )
    assert res_better.total_reward > res_worse.total_reward


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
    assert res4.syllable_reward == 0.0
    assert res4.total_reward <= -0.35

    # Exact v2 hack: 4-line poem with "5-7-5 syllable haiku:" preamble must get negative reward
    v2_hack = (
        "5-7-5 syllable haiku:\n"
        "Whispers of dawn in the breeze,\n"
        "lightning in the mist,\n"
        "night returns."
    )
    res_hack = score_completion(
        topic="a sun-warmed glass marble catching sunlight",
        completion=v2_hack,
        relevance_scorer=DummyRelevanceScorer(0.50),
    )
    assert not res_hack.is_valid_format
    assert not res_hack.is_exact_575
    assert res_hack.syllable_reward == 0.0
    assert res_hack.total_reward <= -0.60

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
    # Indirect poetic imagery (sim=0.48 >= 0.42) keeps the exact 5-7-5 jackpot
    res_poetic = score_completion(
        topic="a simmering skillet of roasted chestnuts",
        completion=off_topic,
        relevance_scorer=DummyRelevanceScorer(0.48),
    )
    assert res_poetic.is_exact_575
    assert res_poetic.exact_bonus == RewardConfig().exact_575_bonus

    # Clearly off-topic (sim=0.35 < 0.42) is denied the jackpot
    res_off = score_completion(
        topic="a simmering skillet of roasted chestnuts",
        completion=off_topic,
        relevance_scorer=DummyRelevanceScorer(0.35),
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


def test_cliche_tracker_penalizes_overused_non_topic_bigrams_and_expires():
    tracker = ClicheTracker(window=4, min_count=2)
    # Record two unrelated haikus that both lean on "silence holds"
    tracker.record("Wind cools the harbor\nLanterns sway above the pier\nSilence holds the dusk", topic="a harbor at dusk")
    tracker.record("Coffee cools in glass\nRain taps on the brick window\nSilence holds the night", topic="rainy coffee shop")

    # On a third unrelated topic, "silence holds" is flagged as a cliche
    cliche_haiku = "Frosted cedar branch\nQuiet snowflakes drift below\nSilence holds the dawn"
    fresh_haiku = "Frosted cedar branch\nQuiet snowflakes drift below\nDawn breaks pale and still"

    res_cliche = score_completion(
        "a winter snowy morning",
        cliche_haiku,
        relevance_scorer=DummyRelevanceScorer(0.75),
        cliche_tracker=tracker,
    )
    res_fresh = score_completion(
        "a winter snowy morning",
        fresh_haiku,
        relevance_scorer=DummyRelevanceScorer(0.75),
        cliche_tracker=tracker,
    )
    assert res_cliche.cliche_penalty == RewardConfig().cliche_penalty_per_bigram
    assert res_fresh.cliche_penalty == 0.0
    assert res_fresh.total_reward > res_cliche.total_reward

    # If the topic itself asks for "silence", topic words are excluded from cliche penalty
    res_topic_ok = score_completion(
        "a glacier valley in silence",
        cliche_haiku,
        relevance_scorer=DummyRelevanceScorer(0.75),
        cliche_tracker=tracker,
    )
    assert res_topic_ok.cliche_penalty == 0.0

    # Window eviction: push 4 fresh haikus -> "silence holds" expires cleanly
    for i in range(4):
        tracker.record(f"Bright sun warms the hill\nBees hum over clover fields\nMeadow wakes at noon", topic=f"topic {i}")
    assert tracker.overused_bigrams(cliche_haiku, topic="a winter snowy morning") == []
    assert ("in", "the") not in extract_content_bigrams("Wind blows in the trees")
