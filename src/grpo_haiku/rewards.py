"""Direct 3-line haiku parsing and shaped RLVR reward computation (no tag rewards)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from grpo_haiku.config import RewardConfig
from grpo_haiku.relevance import RelevanceScorer, has_prompt_echo
from grpo_haiku.syllables import LineSyllableAnalysis, analyze_line, tokenize_words

# Detect stray XML/markdown/think artifacts or digits (e.g. "5-7-5 syllable haiku:")
_STRAY_TAG_RE = re.compile(r"</?(?:think|haiku)\b[^>]*>", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d")


@dataclass(frozen=True)
class ParsedCompletion:
    """Structured parse of a direct 3-line haiku completion."""

    haiku_text: str
    lines: tuple[str, ...]
    has_stray_tags: bool
    has_digits: bool


@dataclass(frozen=True)
class RewardBreakdown:
    """Full transparent breakdown of every reward & penalty component."""

    total_reward: float
    format_reward: float
    syllable_reward: float
    exact_bonus: float
    relevance_reward: float
    oov_penalty: float
    repetition_penalty: float
    echo_penalty: float
    raw_cosine_sim: float
    is_valid_format: bool
    is_exact_575: bool
    line_counts: tuple[int, ...]
    line_distances: tuple[int, ...]
    line_rewards: tuple[float, ...]
    oov_words: tuple[str, ...]
    haiku_lines: tuple[str, ...]


def parse_completion(text: str) -> ParsedCompletion:
    """Parse raw completion directly into non-empty lines (no tag extraction bonus)."""
    has_stray_tags = bool(_STRAY_TAG_RE.search(text))
    has_digits = bool(_DIGIT_RE.search(text))
    cleaned = text.strip()
    lines = tuple(line.strip() for line in cleaned.splitlines() if line.strip())
    return ParsedCompletion(
        haiku_text="\n".join(lines),
        lines=lines,
        has_stray_tags=has_stray_tags,
        has_digits=has_digits,
    )


def _compute_repetition_penalty(
    lines: tuple[str, ...],
    analyses: list[LineSyllableAnalysis],
    cfg: RewardConfig,
) -> float:
    """Penalize duplicate lines, repetitive words, and 1-syllable monotony."""
    penalty = 0.0

    # Duplicate normalized lines
    norm_lines = [" ".join(tokenize_words(line)) for line in lines]
    if len(set(norm_lines)) < len(norm_lines):
        penalty += cfg.duplicate_line_penalty

    # Word diversity across the whole haiku
    all_words = [w for a in analyses for w in a.words]
    if all_words:
        unique_ratio = len(set(all_words)) / len(all_words)
        if unique_ratio < cfg.low_diversity_ratio_threshold:
            penalty += cfg.low_diversity_penalty

        # Monosyllable monotony check: penalize if nearly every known word is 1-syllable
        known_counts = [
            c for a in analyses for c in a.word_counts if c is not None
        ]
        if len(known_counts) >= 6:
            mono_count = sum(1 for c in known_counts if max(c) == 1)
            if (mono_count / len(known_counts)) >= cfg.monosyllable_ratio_threshold:
                penalty += cfg.monosyllable_penalty

    return max(cfg.repetition_penalty_cap, penalty)


def score_completion(
    topic: str,
    completion: str,
    cfg: RewardConfig | None = None,
    relevance_scorer: RelevanceScorer | None = None,
    precomputed_sim: float | None = None,
) -> RewardBreakdown:
    """Compute shaped RLVR reward for a direct 3-line haiku (zero tag rewards)."""
    cfg = cfg or RewardConfig()
    parsed = parse_completion(completion)

    if not parsed.lines:
        return RewardBreakdown(
            total_reward=cfg.empty_output_penalty,
            format_reward=cfg.empty_output_penalty,
            syllable_reward=0.0,
            exact_bonus=0.0,
            relevance_reward=0.0,
            oov_penalty=0.0,
            repetition_penalty=0.0,
            echo_penalty=0.0,
            raw_cosine_sim=0.0,
            is_valid_format=False,
            is_exact_575=False,
            line_counts=(),
            line_distances=(),
            line_rewards=(),
            oov_words=(),
            haiku_lines=(),
        )

    # No positive tag/format reward: 0.0 when cleanly 3 lines with no stray tags or digits,
    # penalty only when line count is wrong, stray tags appear, or digits appear.
    has_three_lines = len(parsed.lines) == len(cfg.line_targets)
    format_reward = 0.0
    if not has_three_lines:
        line_diff = abs(len(parsed.lines) - len(cfg.line_targets))
        format_reward += max(
            cfg.wrong_line_count_penalty_cap,
            cfg.wrong_line_count_penalty_per_line * line_diff,
        )
    if parsed.has_stray_tags:
        format_reward += -0.25
    if parsed.has_digits:
        format_reward += -0.25

    is_valid_format = has_three_lines and (not parsed.has_stray_tags) and (not parsed.has_digits)

    # Sharpened per-line syllable shaping (gated on valid 3-line format so 4-line
    # preambles like "5-7-5 syllable haiku:\n..." cannot farm partial syllable credit!)
    analyses = [analyze_line(line) for line in parsed.lines]
    scored_pairs = list(zip(analyses, cfg.line_targets))

    syllable_reward = 0.0
    line_counts: list[int] = []
    line_distances: list[int] = []
    line_rewards: list[float] = []
    all_lines_exact = is_valid_format

    for analysis, target in scored_pairs:
        dist = analysis.min_distance(target)
        closest = analysis.closest_count(target)
        line_counts.append(closest)
        line_distances.append(dist)
        if is_valid_format:
            line_score = cfg.line_max_reward * max(0.0, 1.0 - cfg.syllable_error_slope * dist)
            syllable_reward += line_score
            line_rewards.append(round(line_score, 6))
        else:
            line_rewards.append(0.0)
        if not analysis.matches_exact(target):
            all_lines_exact = False

    # OOV penalty across all lines
    all_oov = tuple(w for a in analyses for w in a.oov_words)
    oov_penalty = max(cfg.oov_penalty_cap, cfg.oov_penalty_per_word * len(all_oov))

    # Repetition & monotony penalty
    repetition_penalty = _compute_repetition_penalty(parsed.lines, analyses, cfg)

    haiku_plain = parsed.haiku_text

    # Prompt relevance (only when weight > 0 and haiku has 3 valid lines)
    raw_sim = 0.0
    relevance_reward = 0.0
    sim_evaluated = False
    if cfg.relevance_weight > 0.0 and is_valid_format:
        if precomputed_sim is not None:
            raw_sim = float(precomputed_sim)
            sim_evaluated = True
        elif relevance_scorer is not None:
            raw_sim = float(relevance_scorer.similarity(topic, haiku_plain))
            sim_evaluated = True
        denom = max(1e-6, cfg.relevance_max_sim - cfg.relevance_min_sim)
        scaled = min(1.0, max(0.0, (raw_sim - cfg.relevance_min_sim) / denom))
        relevance_reward = cfg.relevance_weight * scaled

    # Exact 5-7-5 jackpot (strictly gated: 3 clean lines, 0 OOV, no dup lines, on-topic)
    norm_lines = [" ".join(tokenize_words(line)) for line in parsed.lines]
    has_dup_lines = len(set(norm_lines)) < len(norm_lines)
    is_exact_575 = all_lines_exact and (len(all_oov) == 0) and (not has_dup_lines)
    on_topic_for_bonus = (not sim_evaluated) or (raw_sim >= cfg.exact_bonus_min_sim)
    exact_bonus = cfg.exact_575_bonus if (is_exact_575 and on_topic_for_bonus) else 0.0

    # Prompt-echo guardrail
    echo_penalty = (
        cfg.prompt_echo_penalty
        if has_prompt_echo(topic, haiku_plain, ngram=cfg.prompt_echo_ngram)
        else 0.0
    )

    total_reward = (
        format_reward
        + syllable_reward
        + exact_bonus
        + relevance_reward
        + oov_penalty
        + repetition_penalty
        + echo_penalty
    )

    return RewardBreakdown(
        total_reward=round(total_reward, 6),
        format_reward=round(format_reward, 6),
        syllable_reward=round(syllable_reward, 6),
        exact_bonus=round(exact_bonus, 6),
        relevance_reward=round(relevance_reward, 6),
        oov_penalty=round(oov_penalty, 6),
        repetition_penalty=round(repetition_penalty, 6),
        echo_penalty=round(echo_penalty, 6),
        raw_cosine_sim=round(raw_sim, 6),
        is_valid_format=is_valid_format,
        is_exact_575=is_exact_575,
        line_counts=tuple(line_counts),
        line_distances=tuple(line_distances),
        line_rewards=tuple(line_rewards),
        oov_words=all_oov,
        haiku_lines=parsed.lines,
    )


def score_group(
    topic: str,
    completions: list[str],
    cfg: RewardConfig | None = None,
    relevance_scorer: RelevanceScorer | None = None,
) -> list[RewardBreakdown]:
    """Batch-score a group of completions for the same topic (batches embedding calls)."""
    cfg = cfg or RewardConfig()
    parsed_list = [parse_completion(c) for c in completions]

    sims: list[float | None] = [None] * len(completions)
    if cfg.relevance_weight > 0.0 and relevance_scorer is not None:
        haiku_texts = [
            p.haiku_text
            if (len(p.lines) == len(cfg.line_targets) and not p.has_stray_tags and not p.has_digits)
            else ""
            for p in parsed_list
        ]
        batch_sims = relevance_scorer.batch_similarity(topic, haiku_texts)
        sims = [float(s) for s in batch_sims]

    return [
        score_completion(
            topic=topic,
            completion=comp,
            cfg=cfg,
            relevance_scorer=None,
            precomputed_sim=sim,
        )
        for comp, sim in zip(completions, sims)
    ]
