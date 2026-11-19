"""Deterministic phonetic syllable counter backed by CMU Pronouncing Dictionary.

Design choices to prevent RL reward hacking:
- Every word is verified against CMUdict (plus strict morphological handling for
  standard possessives, hyphenated compounds, and common contractions).
- Words with multiple valid pronunciations in CMUdict (e.g. "fire" -> {1, 2},
  "every" -> {2, 3}, "hour" -> {1, 2}) contribute ALL valid counts, combined via
  dynamic programming across the line so legitimate pronunciations are never
  unfairly penalized.
- Out-of-vocabulary (OOV) / invented words are NEVER guessed with a gameable
  vowel heuristic; they are explicitly surfaced in `oov_words` so the reward
  function can penalize them and block the exact 5-7-5 jackpot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import cmudict

# Matches words including internal apostrophes or hyphens (e.g., "don't", "snow-kissed")
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
_SIBILANT_ENDINGS = ("s", "x", "z", "ch", "sh", "ce", "ge", "se", "ze")

# Small curated set of standard poetic/everyday contractions missing from some CMUdict builds
_EXTRA_LEXICON: dict[str, frozenset[int]] = {
    "o'er": frozenset({1}),
    "ne'er": frozenset({1}),
    "e'er": frozenset({1}),
    "'tis": frozenset({1}),
    "'twas": frozenset({1}),
    "cannot": frozenset({2}),
    "haiku": frozenset({2}),
    "haikus": frozenset({2}),
    "sunlit": frozenset({2}),
    "moonlit": frozenset({2}),
    "starlit": frozenset({2}),
    "frostbite": frozenset({2}),
    "snowfall": frozenset({2}),
    "snowdrift": frozenset({2}),
    "snowdrifts": frozenset({2}),
    "snowflakes": frozenset({2}),
    "raindrop": frozenset({2}),
    "raindrops": frozenset({2}),
    "dewdrop": frozenset({2}),
    "dewdrops": frozenset({2}),
    "daybreak": frozenset({2}),
    "nightfall": frozenset({2}),
    "twilight": frozenset({2}),
    "firefly": frozenset({3}),
    "fireflies": frozenset({3}),
    "footsteps": frozenset({2}),
    "heartbeat": frozenset({2}),
    "heartbeats": frozenset({2}),
    "teacup": frozenset({2}),
    "teacups": frozenset({2}),
    "windchime": frozenset({2}),
    "windchimes": frozenset({2}),
    "sunbeams": frozenset({2}),
    "moonbeams": frozenset({2}),
    "starlight": frozenset({2}),
    "firelight": frozenset({3}),
    "candlelight": frozenset({3}),
    "lamplight": frozenset({2}),
    "streetlamp": frozenset({2}),
    "streetlamps": frozenset({2}),
    "seabird": frozenset({2}),
    "seabirds": frozenset({2}),
    "pinecone": frozenset({2}),
    "pinecones": frozenset({2}),
    "evergreens": frozenset({3}),
    "cobblestone": frozenset({3}),
    "cobblestones": frozenset({3}),
    "rainstorm": frozenset({2}),
    "snowstorm": frozenset({2}),
    "sunbeam": frozenset({2}),
    "moonbeam": frozenset({2}),
    "wildflower": frozenset({3, 4}),
    "wildflowers": frozenset({3, 4}),
}


@lru_cache(maxsize=1)
def _load_cmu_syllables() -> dict[str, frozenset[int]]:
    """Load CMUdict once and map each lowercase word to its valid syllable counts."""
    raw = cmudict.dict()
    table: dict[str, frozenset[int]] = {}
    for word, pronunciations in raw.items():
        counts = frozenset(
            sum(1 for phoneme in pron if phoneme and phoneme[-1].isdigit())
            for pron in pronunciations
        )
        valid = frozenset(c for c in counts if c > 0)
        if valid:
            table[word.lower()] = valid
    for word, counts in _EXTRA_LEXICON.items():
        if word in table:
            table[word] = table[word] | counts
        else:
            table[word] = counts
    return table


def tokenize_words(text: str) -> list[str]:
    """Extract normalized alphabetic tokens from a line of text."""
    normalized = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u2014", " ").replace("\u2013", " ")
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(normalized)]


def lookup_word_syllables(word: str) -> frozenset[int] | None:
    """Return all valid syllable counts for a word, or None if OOV."""
    table = _load_cmu_syllables()
    w = word.lower().strip("'-")
    if not w:
        return None

    if w in table:
        return table[w]

    # Hyphenated compound (e.g. "snow-covered", "soft-spoken"): sum valid counts of parts
    if "-" in w:
        parts = [p for p in w.split("-") if p]
        if len(parts) >= 2:
            sums: set[int] = {0}
            for part in parts:
                part_counts = lookup_word_syllables(part)
                if part_counts is None:
                    return None
                sums = {a + b for a in sums for b in part_counts}
            return frozenset(sums)

    # Possessive "'s" or plural possessive "s'"
    if w.endswith("'s") and len(w) > 2:
        base = w[:-2]
        base_counts = table.get(base)
        if base_counts is not None:
            extra = 1 if base.endswith(_SIBILANT_ENDINGS) else 0
            return frozenset(c + extra for c in base_counts)

    # Regular plural "-s" / "-es" fallback ONLY when the base stem is in CMUdict
    if w.endswith("es") and len(w) > 3:
        base = w[:-2]
        if base in table and base.endswith(("s", "x", "z", "ch", "sh")):
            return frozenset(c + 1 for c in table[base])
        base_e = w[:-1]
        if base_e in table:
            extra = 1 if base_e.endswith(("ce", "ge", "se", "ze")) else 0
            return frozenset(c + extra for c in table[base_e])
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        base = w[:-1]
        if base in table:
            extra = 1 if base.endswith(_SIBILANT_ENDINGS) else 0
            return frozenset(c + extra for c in table[base])

    return None


@dataclass(frozen=True)
class LineSyllableAnalysis:
    """Detailed phonetic analysis of a single haiku line."""

    text: str
    words: tuple[str, ...]
    word_counts: tuple[frozenset[int] | None, ...]
    possible_counts: frozenset[int]
    oov_words: tuple[str, ...]

    @property
    def has_oov(self) -> bool:
        return len(self.oov_words) > 0

    def min_distance(self, target: int) -> int:
        """Minimum absolute syllable distance to target across valid pronunciations."""
        if not self.words or not self.possible_counts:
            return target
        # Each OOV word contributes at least 1 unknown syllable; penalize distance conservatively
        # if OOV exists by adding OOV count to possible_counts as a minimal 1-syllable placeholder
        if self.oov_words:
            # We already track OOV separately; estimate minimum distance assuming >=1 syl per OOV
            shifted = {c + len(self.oov_words) for c in self.possible_counts}
            return min(abs(c - target) for c in shifted)
        return min(abs(c - target) for c in self.possible_counts)

    def closest_count(self, target: int) -> int:
        """Return the valid syllable count closest to target (0 if empty)."""
        if not self.possible_counts:
            return 0
        counts = (
            {c + len(self.oov_words) for c in self.possible_counts}
            if self.oov_words
            else self.possible_counts
        )
        return min(counts, key=lambda c: (abs(c - target), c))

    def matches_exact(self, target: int) -> bool:
        """True iff there are no OOV words and target is an exact valid pronunciation sum."""
        return (not self.has_oov) and (len(self.words) > 0) and (target in self.possible_counts)


def analyze_line(line: str) -> LineSyllableAnalysis:
    """Analyze a single line into words, valid syllable count sums, and OOV tokens."""
    words = tuple(tokenize_words(line))
    if not words:
        return LineSyllableAnalysis(
            text=line,
            words=(),
            word_counts=(),
            possible_counts=frozenset(),
            oov_words=(),
        )

    word_counts: list[frozenset[int] | None] = []
    oov_words: list[str] = []
    possible: set[int] = {0}

    for w in words:
        counts = lookup_word_syllables(w)
        word_counts.append(counts)
        if counts is None:
            oov_words.append(w)
        else:
            possible = {a + b for a in possible for b in counts}

    return LineSyllableAnalysis(
        text=line,
        words=words,
        word_counts=tuple(word_counts),
        possible_counts=frozenset(possible),
        oov_words=tuple(oov_words),
    )
