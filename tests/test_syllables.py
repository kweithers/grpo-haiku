"""Unit tests for CMUdict phonetic syllable counter."""

from __future__ import annotations

from grpo_haiku.syllables import analyze_line, lookup_word_syllables


def test_classic_haiku_lines():
    l1 = analyze_line("An old silent pond")
    l2 = analyze_line("A frog jumps into the pond")
    l3 = analyze_line("Splash silence again")

    assert l1.matches_exact(5)
    assert l2.matches_exact(7)
    assert l3.matches_exact(5)
    assert not l1.has_oov
    assert not l2.has_oov
    assert not l3.has_oov


def test_multi_pronunciation_words():
    # "fire" can be 1 or 2 syllables in CMUdict; "every" can be 2 or 3
    fire_counts = lookup_word_syllables("fire")
    assert fire_counts is not None
    assert 1 in fire_counts and 2 in fire_counts

    # "cold fire burns so bright" -> 1 + {1,2} + 1 + 1 + 1 = {5, 6}
    line = analyze_line("Cold fire burns so bright")
    assert 5 in line.possible_counts
    assert line.matches_exact(5)


def test_contractions_possessives_and_hyphens():
    assert lookup_word_syllables("don't") == frozenset({1})
    assert lookup_word_syllables("it's") == frozenset({1})
    assert lookup_word_syllables("winter's") == frozenset({2})
    assert lookup_word_syllables("snow-covered") == frozenset({3})

    line = analyze_line("Winter's snow-covered pines")
    assert not line.has_oov
    assert 6 in line.possible_counts


def test_oov_gibberish_strictly_flagged():
    line = analyze_line("Blorptastic flibber shines")
    assert line.has_oov
    assert "blorptastic" in line.oov_words
    assert "flibber" in line.oov_words
    assert not line.matches_exact(5)
    assert not line.matches_exact(7)


def test_punctuation_and_smart_quotes():
    line = analyze_line("Soft snow falls—quiet, bright!")
    assert not line.has_oov
    assert line.words == ("soft", "snow", "falls", "quiet", "bright")
    # quiet is 2 syllables -> 1 + 1 + 1 + 2 + 1 = 6
    assert 6 in line.possible_counts
    assert line.min_distance(5) == 1
