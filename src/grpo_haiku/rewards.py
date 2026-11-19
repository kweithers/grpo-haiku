"""Tag parsing and shaped RLVR reward computation for haiku generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from grpo_haiku.config import RewardConfig
from grpo_haiku.data import FEWSHOT_ASSISTANT
from grpo_haiku.relevance import RelevanceScorer, has_prompt_echo
from grpo_haiku.syllables import LineSyllableAnalysis, analyze_line, tokenize_words

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_HAIKU_RE = re.compile(r"<haiku>(.*?)</haiku>", re.DOTALL | re.IGNORECASE)
_FEWSHOT_HAIKU_TEXT = (
    _HAIKU_RE.search(FEWSHOT_ASSISTANT).group(1).strip()
    if _HAIKU_RE.search(FEWSHOT_ASSISTANT)
    else ""
)


@dataclass(frozen=True)
class ParsedCompletion:
    """Structured parse of a model completion containing <think> and <haiku> tags."""

    has_think: bool
    think_text: str
    has_haiku: bool
    haiku_text: str
    lines: tuple[str, ...]
    think_before_haiku: bool
    has_trailing_text: bool


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
    oov_words: tuple[str, ...]
    haiku_lines: tuple[str, ...]
    think_text: str


def parse_completion(text: str) -> ParsedCompletion:
    """Extract <think>...</think> and <haiku>...</haiku> blocks cleanly."""
    think_match = _THINK_RE.search(text)
    haiku_match = _HAIKU_RE.search(text)

    has_think = think_match is not None and bool(think_match.group(1).strip())
    think_text = think_match.group(1).strip() if think_match else ""

    has_haiku = haiku_match is not None
    haiku_text = haiku_match.group(1).strip() if haiku_match else ""
    lines = tuple(line.strip() for line in haiku_text.splitlines() if line.strip())

    think_before_haiku = (
        think_match is not None
        and haiku_match is not None
        and think_match.end() <= haiku_match.start()
    )
    has_trailing_text = (
        haiku_match is not None and bool(text[haiku_match.end() :].strip())
    )

    return ParsedCompletion(
        has_think=has_think,
        think_text=think_text,
        has_haiku=has_haiku,
        haiku_text=haiku_text,
        lines=lines,
        think_before_haiku=think_before_haiku,
        has_trailing_text=has_trailing_text,
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
    """Compute shaped RLVR reward and detailed diagnostics for a single completion."""
    cfg = cfg or RewardConfig()
    parsed = parse_completion(completion)

    # Case 1: Missing or empty <haiku> block
    if not parsed.has_haiku or not parsed.lines:
        fmt = cfg.missing_haiku_penalty + (cfg.think_tag_reward if parsed.has_think else 0.0)
        return RewardBreakdown(
            total_reward=fmt,
            format_reward=fmt,
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
            oov_words=(),
            haiku_lines=parsed.lines,
            think_text=parsed.think_text,
        )

    # Format reward
    format_reward = 0.0
    if parsed.has_think and parsed.think_before_haiku:
        format_reward += cfg.think_tag_reward
    elif parsed.has_think:
        format_reward += 0.5 * cfg.think_tag_reward

    has_three_lines = len(parsed.lines) == len(cfg.line_targets)
    if has_three_lines:
        format_reward += cfg.haiku_tag_reward
    else:
        # Partial credit for outputting <haiku> tags, minus line-count error penalty
        line_diff = abs(len(parsed.lines) - len(cfg.line_targets))
        format_reward += max(-0.30, 0.05 - 0.15 * line_diff)

    if parsed.has_trailing_text:
        format_reward += cfg.trailing_text_penalty

    is_valid_format = (
        parsed.has_think
        and parsed.think_before_haiku
        and parsed.has_haiku
        and has_three_lines
        and not parsed.has_trailing_text
    )

    # Syllable analysis across lines (score up to 3 lines against 5-7-5)
    analyses = [analyze_line(line) for line in parsed.lines]
    scored_pairs = list(zip(analyses, cfg.line_targets))

    syllable_reward = 0.0
    line_counts: list[int] = []
    line_distances: list[int] = []
    all_lines_exact = has_three_lines

    for analysis, target in scored_pairs:
        dist = analysis.min_distance(target)
        closest = analysis.closest_count(target)
        line_counts.append(closest)
        line_distances.append(dist)
        line_score = cfg.line_max_reward * max(0.0, 1.0 - cfg.syllable_error_slope * dist)
        syllable_reward += line_score
        if not analysis.matches_exact(target):
            all_lines_exact = False

    # OOV penalty across all lines in <haiku>
    all_oov = tuple(w for a in analyses for w in a.oov_words)
    oov_penalty = max(cfg.oov_penalty_cap, cfg.oov_penalty_per_word * len(all_oov))

    # Repetition & monotony penalty
    repetition_penalty = _compute_repetition_penalty(parsed.lines, analyses, cfg)

    haiku_plain = "\n".join(parsed.lines)

    # Anti-hacking: detect copying the 1-shot example haiku from the prompt
    is_fewshot_copy = bool(_FEWSHOT_HAIKU_TEXT) and has_prompt_echo(
        _FEWSHOT_HAIKU_TEXT, haiku_plain, ngram=cfg.prompt_echo_ngram
    )

    # Prompt relevance (only when weight > 0 and haiku has 3 lines)
    raw_sim = 0.0
    relevance_reward = 0.0
    sim_evaluated = False
    if cfg.relevance_weight > 0.0 and has_three_lines:
        if precomputed_sim is not None:
            raw_sim = float(precomputed_sim)
            sim_evaluated = True
        elif relevance_scorer is not None:
            raw_sim = float(relevance_scorer.similarity(topic, haiku_plain))
            sim_evaluated = True
        denom = max(1e-6, cfg.relevance_max_sim - cfg.relevance_min_sim)
        scaled = min(1.0, max(0.0, (raw_sim - cfg.relevance_min_sim) / denom))
        relevance_reward = cfg.relevance_weight * scaled

    # Exact 5-7-5 bonus (strictly gated: 3 lines, 0 OOV, no dup lines, not fewshot copy, on-topic)
    norm_lines = [" ".join(tokenize_words(line)) for line in parsed.lines]
    has_dup_lines = len(set(norm_lines)) < len(norm_lines)
    is_exact_575 = (
        all_lines_exact
        and (len(all_oov) == 0)
        and (not has_dup_lines)
        and (not is_fewshot_copy)
    )
    on_topic_for_bonus = (not sim_evaluated) or (raw_sim >= cfg.exact_bonus_min_sim)
    exact_bonus = cfg.exact_575_bonus if (is_exact_575 and on_topic_for_bonus) else 0.0

    # Prompt-echo & few-shot copy guardrail
    echo_penalty = 0.0
    if has_prompt_echo(topic, haiku_plain, ngram=cfg.prompt_echo_ngram):
        echo_penalty += cfg.prompt_echo_penalty
    if is_fewshot_copy:
        echo_penalty += cfg.fewshot_copy_penalty

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
        oov_words=all_oov,
        haiku_lines=parsed.lines,
        think_text=parsed.think_text,
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
        haiku_texts = ["\n".join(p.lines) if len(p.lines) == len(cfg.line_targets) else "" for p in parsed_list]
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
