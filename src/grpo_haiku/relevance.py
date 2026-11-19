"""Lightweight MLX semantic relevance scorer & prompt-echo detector."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from grpo_haiku.syllables import tokenize_words

_PROMPT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "about",
        "write",
        "compose",
        "create",
        "craft",
        "haiku",
        "poem",
        "capture",
        "capturing",
        "describing",
        "into",
        "over",
        "under",
        "after",
        "before",
        "during",
        "and",
        "or",
        "as",
        "is",
        "are",
    }
)


class RelevanceScorer(Protocol):
    """Protocol for computing cosine similarity in [-1, 1] between topic and haiku."""

    def similarity(self, topic: str, haiku: str) -> float: ...

    def batch_similarity(self, topic: str, haikus: list[str]) -> list[float]: ...


def has_prompt_echo(topic: str, haiku: str, ngram: int = 3) -> bool:
    """Return True if the haiku copies `ngram` consecutive words verbatim from the topic.

    Checks both raw word n-grams (for length >= ngram) and content-word n-grams so
    the model cannot farm cosine similarity by parroting "a winter snowy morning"
    directly into a haiku line.
    """
    if ngram <= 0:
        return False

    topic_words = tokenize_words(topic)
    haiku_words = tokenize_words(haiku)
    if len(topic_words) < ngram or len(haiku_words) < ngram:
        # Also catch short 2-content-word topics echoed verbatim if topic is very short
        topic_content = [w for w in topic_words if w not in _PROMPT_STOPWORDS]
        haiku_content = [w for w in haiku_words if w not in _PROMPT_STOPWORDS]
        if len(topic_content) >= ngram and len(haiku_content) >= ngram:
            topic_ngrams = {
                tuple(topic_content[i : i + ngram])
                for i in range(len(topic_content) - ngram + 1)
            }
            for i in range(len(haiku_content) - ngram + 1):
                if tuple(haiku_content[i : i + ngram]) in topic_ngrams:
                    return True
        return False

    topic_raw_ngrams = {
        tuple(topic_words[i : i + ngram])
        for i in range(len(topic_words) - ngram + 1)
    }
    for i in range(len(haiku_words) - ngram + 1):
        gram = tuple(haiku_words[i : i + ngram])
        if gram in topic_raw_ngrams:
            # Require at least 2 non-stopword tokens in the echoed span so harmless
            # phrases like "in the morning" aren't over-penalized
            non_stop = sum(1 for w in gram if w not in _PROMPT_STOPWORDS)
            if non_stop >= max(2, ngram - 1):
                return True

    # Also check contiguous content-word n-grams (catches dropping "a"/"the")
    topic_content = [w for w in topic_words if w not in _PROMPT_STOPWORDS]
    haiku_content = [w for w in haiku_words if w not in _PROMPT_STOPWORDS]
    if len(topic_content) >= ngram and len(haiku_content) >= ngram:
        topic_c_ngrams = {
            tuple(topic_content[i : i + ngram])
            for i in range(len(topic_content) - ngram + 1)
        }
        for i in range(len(haiku_content) - ngram + 1):
            if tuple(haiku_content[i : i + ngram]) in topic_c_ngrams:
                return True

    return False


class MLXEmbeddingRelevanceScorer:
    """Lazy-loaded local MLX embedder (`bge-small`, ~65 MB) with topic caching."""

    def __init__(self, registry_name: str = "bge-small") -> None:
        self.registry_name = registry_name
        self._model = None
        self._topic_cache: dict[str, np.ndarray] = {}

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from mlx_embedding_models.embedding import EmbeddingModel

            self._model = EmbeddingModel.from_registry(self.registry_name)
            # transformers>=5 removed `batch_encode_plus`; `__call__` is its exact replacement
            if not hasattr(self._model.tokenizer, "batch_encode_plus"):
                self._model.tokenizer.batch_encode_plus = self._model.tokenizer.__call__

    def _encode_normalized(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        vecs = np.asarray(self._model.encode(texts, show_progress=False), dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        return vecs / np.maximum(norms, 1e-8)

    def _topic_vec(self, topic: str) -> np.ndarray:
        if topic not in self._topic_cache:
            # Keep cache bounded so memory stays negligible
            if len(self._topic_cache) > 2048:
                self._topic_cache.clear()
            self._topic_cache[topic] = self._encode_normalized([topic])[0]
        return self._topic_cache[topic]

    def similarity(self, topic: str, haiku: str) -> float:
        if not haiku.strip():
            return 0.0
        t_vec = self._topic_vec(topic)
        h_vec = self._encode_normalized([haiku])[0]
        return float(np.dot(t_vec, h_vec))

    def batch_similarity(self, topic: str, haikus: list[str]) -> list[float]:
        if not haikus:
            return []
        t_vec = self._topic_vec(topic)
        non_empty_indices = [i for i, h in enumerate(haikus) if h.strip()]
        scores = [0.0] * len(haikus)
        if non_empty_indices:
            batch_texts = [haikus[i] for i in non_empty_indices]
            h_vecs = self._encode_normalized(batch_texts)
            sims = h_vecs @ t_vec
            for idx, sim in zip(non_empty_indices, sims.tolist()):
                scores[idx] = float(sim)
        return scores
