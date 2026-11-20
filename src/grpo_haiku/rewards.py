"""Direct 3-line haiku parsing and shaped RLVR reward computation (no tag rewards)."""

from __future__ import annotations

from collections import Counter, deque
import re
from dataclasses import dataclass

from grpo_haiku.config import RewardConfig
from grpo_haiku.relevance import RelevanceScorer, has_prompt_echo
from grpo_haiku.syllables import LineSyllableAnalysis, analyze_line, tokenize_words

# Detect stray XML/markdown/think artifacts or digits (e.g. "5-7-5 syllable haiku:")
_STRAY_TAG_RE = re.compile(r"</?(?:think|haiku)\b[^>]*>", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d")
_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "by",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "be",
        "and",
        "or",
        "but",
        "not",
        "no",
        "it",
        "its",
        "this",
        "that",
        "they",
        "their",
        "into",
        "over",
        "under",
        "through",
        "upon",
    }
)


def extract_content_bigrams(
    text: str, exclude_words: set[str] | frozenset[str] = frozenset()
) -> set[tuple[str, str]]:
    """Extract unique non-trivial bigrams where at least one word is a content word
    and not both words belong to the current prompt topic."""
    words = tokenize_words(text)
    bigrams: set[tuple[str, str]] = set()
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 in _FUNCTION_WORDS and w2 in _FUNCTION_WORDS:
            continue
        if exclude_words and (w1 in exclude_words or w2 in exclude_words):
            continue
        # Only track bigrams where both words aren't tiny 1-2 char fillers or at least one is substantive
        if (w1 not in _FUNCTION_WORDS) and (w2 not in _FUNCTION_WORDS):
            bigrams.add((w1, w2))
    return bigrams


def extract_content_words(
    text: str, exclude_words: set[str] | frozenset[str] = frozenset()
) -> set[str]:
    """Extract unique non-function, non-topic content words."""
    return {
        w
        for w in tokenize_words(text)
        if (w not in _FUNCTION_WORDS) and (w not in exclude_words) and len(w) >= 3
    }


def extract_line_starters(
    lines: tuple[str, ...] | list[str],
    exclude_words: set[str] | frozenset[str] = frozenset(),
) -> tuple[set[str], set[tuple[str, str]]]:
    """Extract 1-word and 2-word non-topic line openers from haiku lines."""
    starters1: set[str] = set()
    starters2: set[tuple[str, str]] = set()
    for line in lines:
        words = tokenize_words(line)
        if not words:
            continue
        w0 = words[0]
        if w0 not in _FUNCTION_WORDS and w0 not in exclude_words and len(w0) >= 3:
            starters1.add(w0)
        if len(words) >= 2:
            w1 = words[1]
            if not (w0 in _FUNCTION_WORDS and w1 in _FUNCTION_WORDS):
                if (w0 not in exclude_words) and (w1 not in exclude_words):
                    starters2.add((w0, w1))
    return starters1, starters2


class ClicheTracker:
    """Sliding-window tracker of recent cross-prompt starters, bigrams, and content words."""

    def __init__(
        self,
        window: int = 64,
        bigram_min_count: int = 3,
        starter1_min_count: int = 5,
        starter2_min_count: int = 3,
        unigram_min_count: int = 0,
    ) -> None:
        self.window = max(1, window)
        self.bigram_min_count = max(0, bigram_min_count)
        self.starter1_min_count = max(0, starter1_min_count)
        self.starter2_min_count = max(0, starter2_min_count)
        self.unigram_min_count = max(0, unigram_min_count)
        self._history: deque[
            tuple[set[tuple[str, str]], set[str], set[tuple[str, str]], set[str]]
        ] = deque()
        self._bigram_counts: Counter[tuple[str, str]] = Counter()
        self._starter1_counts: Counter[str] = Counter()
        self._starter2_counts: Counter[tuple[str, str]] = Counter()
        self._unigram_counts: Counter[str] = Counter()

    def _extract_features(
        self, haiku_text: str, topic: str
    ) -> tuple[set[tuple[str, str]], set[str], set[tuple[str, str]], set[str]]:
        topic_words = set(tokenize_words(topic))
        lines = [ln.strip() for ln in haiku_text.splitlines() if ln.strip()]
        bigrams = extract_content_bigrams(haiku_text, exclude_words=topic_words)
        starters1, starters2 = extract_line_starters(lines, exclude_words=topic_words)
        unigrams = extract_content_words(haiku_text, exclude_words=topic_words)
        return bigrams, starters1, starters2, unigrams

    def cliche_hits(self, haiku_text: str, topic: str = "") -> list[str]:
        """Return deduplicated list of overused starter/bigram/unigram hits."""
        bigrams, starters1, starters2, unigrams = self._extract_features(haiku_text, topic)
        hits: list[str] = []
        if self.starter2_min_count > 0:
            for s2 in sorted(starters2):
                if self._starter2_counts.get(s2, 0) >= self.starter2_min_count:
                    hits.append(f"start2:{s2[0]} {s2[1]}")
        if self.bigram_min_count > 0:
            for bg in sorted(bigrams):
                if self._bigram_counts.get(bg, 0) >= self.bigram_min_count:
                    label = f"{bg[0]} {bg[1]}"
                    if f"start2:{label}" not in hits:
                        hits.append(f"bg:{label}")
        if self.starter1_min_count > 0:
            for s1 in sorted(starters1):
                if self._starter1_counts.get(s1, 0) >= self.starter1_min_count:
                    hits.append(f"start1:{s1}")
        if self.unigram_min_count > 0:
            for u in sorted(unigrams):
                if self._unigram_counts.get(u, 0) >= self.unigram_min_count:
                    if f"start1:{u}" not in hits:
                        hits.append(f"word:{u}")
        return hits

    def overused_bigrams(
        self, haiku_text: str, topic: str = ""
    ) -> list[tuple[str, str]]:
        """Return content bigrams in `haiku_text` already seen >= `bigram_min_count` times."""
        if self.bigram_min_count <= 0:
            return []
        topic_words = set(tokenize_words(topic))
        bigrams = extract_content_bigrams(haiku_text, exclude_words=topic_words)
        return sorted(
            bg for bg in bigrams if self._bigram_counts.get(bg, 0) >= self.bigram_min_count
        )

    def record(self, haiku_text: str, topic: str = "") -> None:
        """Record one completed haiku's non-topic features into the rolling window."""
        feats = self._extract_features(haiku_text, topic)
        if len(self._history) >= self.window:
            old_bg, old_s1, old_s2, old_u = self._history.popleft()
            for k in old_bg:
                self._bigram_counts[k] -= 1
                if self._bigram_counts[k] <= 0:
                    del self._bigram_counts[k]
            for k in old_s1:
                self._starter1_counts[k] -= 1
                if self._starter1_counts[k] <= 0:
                    del self._starter1_counts[k]
            for k in old_s2:
                self._starter2_counts[k] -= 1
                if self._starter2_counts[k] <= 0:
                    del self._starter2_counts[k]
            for k in old_u:
                self._unigram_counts[k] -= 1
                if self._unigram_counts[k] <= 0:
                    del self._unigram_counts[k]

        self._history.append(feats)
        bg, s1, s2, u = feats
        for k in bg:
            self._bigram_counts[k] += 1
        for k in s1:
            self._starter1_counts[k] += 1
        for k in s2:
            self._starter2_counts[k] += 1
        for k in u:
            self._unigram_counts[k] += 1


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
    cliche_penalty: float
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
    cliche_tracker: ClicheTracker | None = None,
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
            cliche_penalty=0.0,
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

    # Cross-step cliche novelty gate & penalty (active during GRPO when tracker is provided)
    cliche_penalty = 0.0
    if cliche_tracker is not None and is_valid_format:
        hits = cliche_tracker.cliche_hits(haiku_plain, topic=topic)
        if hits:
            # Strongly discount the exact 5-7-5 jackpot so templates can't outscore fresh 5-7-5s
            if exact_bonus > 0.0:
                exact_bonus = exact_bonus * (cfg.cliche_jackpot_decay ** len(hits))
            cliche_penalty = max(
                cfg.cliche_penalty_cap,
                cfg.cliche_penalty_per_hit * len(hits),
            )

    total_reward = (
        format_reward
        + syllable_reward
        + exact_bonus
        + relevance_reward
        + oov_penalty
        + repetition_penalty
        + echo_penalty
        + cliche_penalty
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
        cliche_penalty=round(cliche_penalty, 6),
        raw_cosine_sim=round(raw_sim, 6),
        is_valid_format=is_valid_format,
        is_exact_575=is_exact_575,
        line_counts=tuple(line_counts),
        line_distances=tuple(line_distances),
        line_rewards=tuple(line_rewards),
        oov_words=all_oov,
        haiku_lines=parsed.lines,
    )


def _apply_fitness_sharing(
    topic: str,
    breakdowns: list[RewardBreakdown],
    cfg: RewardConfig,
) -> list[RewardBreakdown]:
    """Within-group niching: divide `exact_bonus` among near-duplicate templates in the group.

    Two valid rollouts are treated as template peers if they share a non-topic 2-word
    line opener (e.g. "Nightfall claims...") OR have non-topic content-word Jaccard
    similarity >= `cfg.fitness_sharing_jaccard`. Solitary fresh exact 5-7-5 poems keep
    100% of `exact_bonus`, while herds of template copies split it.
    """
    if (not cfg.fitness_sharing) or len(breakdowns) <= 1:
        return breakdowns

    topic_words = set(tokenize_words(topic))
    feats: list[tuple[set[tuple[str, str]], set[str]]] = []
    for bd in breakdowns:
        if bd.is_valid_format and bd.haiku_lines:
            _, s2 = extract_line_starters(bd.haiku_lines, exclude_words=topic_words)
            cwords = extract_content_words("\n".join(bd.haiku_lines), exclude_words=topic_words)
            feats.append((s2, cwords))
        else:
            feats.append((set(), set()))

    adjusted: list[RewardBreakdown] = []
    for i, bd in enumerate(breakdowns):
        if bd.exact_bonus <= 0.0 or not bd.is_valid_format:
            adjusted.append(bd)
            continue

        s2_i, cw_i = feats[i]
        cluster_size = 1
        for j, other in enumerate(breakdowns):
            if j == i or not other.is_valid_format:
                continue
            s2_j, cw_j = feats[j]
            shared_opener = bool(s2_i and s2_j and (s2_i & s2_j))
            union = cw_i | cw_j
            jaccard = (len(cw_i & cw_j) / len(union)) if union else 0.0
            if shared_opener or jaccard >= cfg.fitness_sharing_jaccard:
                cluster_size += 1

        if cluster_size == 1:
            adjusted.append(bd)
            continue

        new_exact = round(bd.exact_bonus / (cluster_size ** cfg.fitness_sharing_exponent), 6)
        delta = new_exact - bd.exact_bonus
        adjusted.append(
            RewardBreakdown(
                total_reward=round(bd.total_reward + delta, 6),
                format_reward=bd.format_reward,
                syllable_reward=bd.syllable_reward,
                exact_bonus=new_exact,
                relevance_reward=bd.relevance_reward,
                oov_penalty=bd.oov_penalty,
                repetition_penalty=bd.repetition_penalty,
                echo_penalty=bd.echo_penalty,
                cliche_penalty=bd.cliche_penalty,
                raw_cosine_sim=bd.raw_cosine_sim,
                is_valid_format=bd.is_valid_format,
                is_exact_575=bd.is_exact_575,
                line_counts=bd.line_counts,
                line_distances=bd.line_distances,
                line_rewards=bd.line_rewards,
                oov_words=bd.oov_words,
                haiku_lines=bd.haiku_lines,
            )
        )
    return adjusted


def score_group(
    topic: str,
    completions: list[str],
    cfg: RewardConfig | None = None,
    relevance_scorer: RelevanceScorer | None = None,
    cliche_tracker: ClicheTracker | None = None,
    apply_fitness_sharing: bool = False,
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

    breakdowns = [
        score_completion(
            topic=topic,
            completion=comp,
            cfg=cfg,
            relevance_scorer=None,
            precomputed_sim=sim,
            cliche_tracker=cliche_tracker,
        )
        for comp, sim in zip(completions, sims)
    ]
    if apply_fitness_sharing:
        breakdowns = _apply_fitness_sharing(topic, breakdowns, cfg)
    return breakdowns
