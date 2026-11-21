"""Unit tests for direct 3-line haiku parsing, shaped rewards, and anti-hacking guardrails."""

from __future__ import annotations

from grpo_haiku.config import RewardConfig
from grpo_haiku.relevance import has_prompt_echo
from grpo_haiku.rewards import (
    ClicheTracker,
    analyze_think_counts,
    compute_think_grounding,
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


def test_cliche_tracker_discounts_jackpot_for_overused_starters_bigrams_and_words():
    tracker = ClicheTracker(
        window=4,
        bigram_min_count=2,
        starter1_min_count=2,
        starter2_min_count=2,
        unigram_min_count=2,
    )
    # Record two unrelated haikus that both lean on "Silence holds..."
    tracker.record("Wind cools the harbor\nLanterns sway above the pier\nSilence holds the dusk", topic="a harbor at dusk")
    tracker.record("Coffee cools in glass\nRain taps on the brick window\nSilence holds the night", topic="rainy coffee shop")

    # On a third unrelated topic, "Silence holds" triggers both starter/bigram hits -> heavy jackpot discount
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
    assert res_cliche.cliche_penalty < 0.0
    assert res_cliche.exact_bonus < RewardConfig().exact_575_bonus * 0.5
    assert res_fresh.cliche_penalty == 0.0
    assert res_fresh.exact_bonus == RewardConfig().exact_575_bonus
    assert res_fresh.total_reward - res_cliche.total_reward > 1.5

    # If the topic itself asks for "silence", using "silence" with a fresh verb is NOT penalized
    topic_silence_haiku = "Frosted cedar branch\nQuiet snowflakes drift below\nSilence fills the dawn"
    res_topic_ok = score_completion(
        "a glacier valley in silence",
        topic_silence_haiku,
        relevance_scorer=DummyRelevanceScorer(0.75),
        cliche_tracker=tracker,
    )
    assert res_topic_ok.cliche_penalty == 0.0

    # Window eviction: push 4 fresh haikus -> "silence holds" expires cleanly
    for i in range(4):
        tracker.record("Bright sun warms the hill\nBees hum over clover fields\nMeadow wakes at noon", topic=f"topic {i}")
    assert tracker.cliche_hits(cliche_haiku, topic="a winter snowy morning") == []
    assert ("in", "the") not in extract_content_bigrams("Wind blows in the trees")


def test_fitness_sharing_rewards_unique_exact_haiku_over_template_herd():
    # Three exact 5-7-5 haikus sharing the "Nightfall claims..." template + one fresh exact 5-7-5
    herd_1 = "Nightfall claims the snow\nQuiet snowflakes drift below\nDawn breaks pale and still"
    herd_2 = "Nightfall claims the frost\nSilver branches bend in wind\nMorning wakes the pines"
    herd_3 = "Nightfall claims the hill\nFrozen footprints line the road\nWinter breathes out cold"
    fresh = "Cedar boughs hold white\nCrisp air wakes the sleeping pines\nSunlight melts the rime"

    results = score_group(
        topic="a winter snowy morning",
        completions=[herd_1, herd_2, herd_3, fresh],
        relevance_scorer=DummyRelevanceScorer(0.75),
        apply_fitness_sharing=True,
    )
    # Fresh haiku keeps the full unshared jackpot; herd copies split it
    assert results[3].exact_bonus == RewardConfig().exact_575_bonus
    for i in range(3):
        assert results[i].exact_bonus < RewardConfig().exact_575_bonus * 0.5
        assert results[3].total_reward > results[i].total_reward + 1.0


def test_cliche_tracker_disables_unigrams_when_threshold_zero():
    cfg = RewardConfig()
    assert cfg.cliche_unigram_min_count == 0
    tracker = ClicheTracker(
        window=16,
        bigram_min_count=2,
        starter1_min_count=2,
        starter2_min_count=2,
        unigram_min_count=cfg.cliche_unigram_min_count,
    )
    # Use common poetic word "light" in mid-line positions across several unrelated poems
    tracker.record("Boats rock near the light\nTides pull pebbles off the shore\nGulls call from the dock", topic="a coastal pier")
    tracker.record("Steam curls in the light\nKettle hums above the stove\nToast warms on a plate", topic="a morning kitchen")
    tracker.record("Leaves shine in the light\nBreezes rock the maple crown\nGrass bends near the fence", topic="a windy park")

    # Mid-line reuse of "light" with fresh starter & bigrams must NOT trigger any cliche penalty
    fresh_with_light = "Cedar boughs hold light\nQuiet snowflakes drift below\nDawn breaks pale and still"
    res = score_completion(
        "a winter snowy morning",
        fresh_with_light,
        cfg=cfg,
        relevance_scorer=DummyRelevanceScorer(0.75),
        cliche_tracker=tracker,
    )
    assert tracker.cliche_hits(fresh_with_light, topic="a winter snowy morning") == []
    assert res.cliche_penalty == 0.0
    assert res.exact_bonus == cfg.exact_575_bonus

    # Meanwhile, repeating a 2-word starter or content bigram still triggers cleanly
    tracker.record("Nightfall claims the pier\nCold spray settles on the wood\nLanterns blink at dusk", topic="a foggy dock")
    tracker.record("Nightfall claims the road\nHeadlights vanish past the bend\nCrickets start to sing", topic="a highway")
    cliche_starter = "Nightfall claims the snow\nQuiet snowflakes drift below\nDawn breaks pale and still"
    hits = tracker.cliche_hits(cliche_starter, topic="a winter snowy morning")
    assert any(h.startswith("start2:") or h.startswith("bg:") for h in hits)
    assert not any(h.startswith("word:") for h in hits)


def test_grounded_think_mode_rewards_honest_trace_and_blocks_fake_boilerplate():
    cfg = RewardConfig(thinking_mode=True)

    honest = (
        "<think>\n"
        "1: Frosted(2) cedar(2) branch(1) = 5\n"
        "2: Quiet(2) snowflakes(2) drift(1) slow(1) = 6 -> Quiet(2) snowflakes(2) drift(1) below(2) = 7\n"
        "3: Dawn(1) breaks(1) pale(1) and(1) still(1) = 5\n"
        "</think>\n"
        "Frosted cedar branch\n"
        "Quiet snowflakes drift below\n"
        "Dawn breaks pale and still"
    )
    res_honest = score_completion(
        topic="a quiet winter snowy morning",
        completion=honest,
        cfg=cfg,
        relevance_scorer=DummyRelevanceScorer(0.78),
    )
    assert res_honest.is_valid_format
    assert res_honest.is_exact_575
    assert res_honest.format_reward == 0.0
    assert res_honest.think_grounding == 1.0
    assert res_honest.think_penalty == 0.0
    assert res_honest.exact_bonus == cfg.exact_575_bonus
    assert res_honest.total_reward == 4.50

    diag = analyze_think_counts(res_honest.think_text)
    assert diag["all_lines_present"]
    assert diag["counts_valid_cmudict"]
    assert diag["sums_match_target"]
    assert diag["is_honest_575"]
    assert diag["has_revision"]

    # Fake Run-1 style autumn-leaves copy hack inside <think> must forfeit the jackpot!
    fake_think = (
        "<think>\n"
        "1: Autumn(2) leaves(1) drift(1) down(1) = 5\n"
        "2: Crimson(2) maples(2) paint(1) the(1) ground(1) = 7\n"
        "3: Cold(1) wind(1) calls(1) the(1) night(1) = 5\n"
        "</think>\n"
        "Frosted cedar branch\n"
        "Quiet snowflakes drift below\n"
        "Dawn breaks pale and still"
    )
    res_fake = score_completion(
        topic="a quiet winter snowy morning",
        completion=fake_think,
        cfg=cfg,
        relevance_scorer=DummyRelevanceScorer(0.78),
    )
    assert res_fake.is_valid_format
    assert res_fake.think_grounding < cfg.grounding_min_overlap
    assert res_fake.exact_bonus == 0.0
    assert res_fake.think_penalty == cfg.ungrounded_think_penalty
    assert res_honest.total_reward > res_fake.total_reward + 2.5

    # Missing or unclosed <think> gets missing_think_penalty and zero syllable credit
    truncated = "<think>\n1: Frosted(2) cedar(2) branch(1) = 5\n2: Quiet(2) snowflakes(2)"
    res_trunc = score_completion(
        topic="a quiet winter snowy morning",
        completion=truncated,
        cfg=cfg,
        relevance_scorer=DummyRelevanceScorer(0.78),
    )
    assert not res_trunc.is_valid_format
    assert not res_trunc.is_exact_575
    assert res_trunc.format_reward == cfg.missing_think_penalty
    assert res_trunc.syllable_reward == 0.0
    assert res_trunc.haiku_lines == ()

    # Digits after </think> are still rejected even though digits inside <think> are allowed
    digits_after = (
        "<think>\n"
        "1: Frosted(2) cedar(2) branch(1) = 5\n"
        "2: Quiet(2) snowflakes(2) drift(1) below(2) = 7\n"
        "3: Dawn(1) breaks(1) pale(1) and(1) still(1) = 5\n"
        "</think>\n"
        "1: Frosted cedar branch\n"
        "2: Quiet snowflakes drift below\n"
        "3: Dawn breaks pale and still"
    )
    res_digits = score_completion(
        topic="a quiet winter snowy morning",
        completion=digits_after,
        cfg=cfg,
        relevance_scorer=DummyRelevanceScorer(0.78),
    )
    assert not res_digits.is_valid_format
    assert res_digits.syllable_reward == 0.0
    assert res_digits.exact_bonus == 0.0

    # Grounding helper direct check
    assert (
        compute_think_grounding(
            "frosted cedar branch quiet snowflakes drift below dawn breaks pale still",
            res_honest.haiku_lines,
        )
        == 1.0
    )
