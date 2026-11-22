"""Rejection-sampling data collector and LoRA SFT trainer for 5-7-5 haiku cold-start.

Uses the exact same 4-bit base model and attention+MLP LoRA architecture as `grpo.py`,
saving adapters in standard `mlx-lm` format so GRPO can resume directly from the SFT
checkpoint via `--resume-adapter`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from grpo_haiku.config import ModelConfig, RewardConfig
from grpo_haiku.data import HaikuPromptExample, format_chat_prompt, load_jsonl
from grpo_haiku.grpo import (
    build_lora_model,
    clip_grad_norm,
    cosine_lr,
    gradient_checkpointing,
    normalize_completion_text,
    save_adapter,
    sequence_completion_logprobs,
)
from grpo_haiku.relevance import MLXEmbeddingRelevanceScorer, RelevanceScorer
from grpo_haiku.rewards import (
    RewardBreakdown,
    extract_content_bigrams,
    extract_line_starters,
    parse_completion,
    score_group,
)
from grpo_haiku.syllables import LineSyllableAnalysis, analyze_line, tokenize_words

_SURFACE_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
_TRACE_LINE_RE = re.compile(r"^\s*([123])\s*:\s*(.+)$")
_TRACE_WORD_RE = re.compile(r"([A-Za-z]+(?:['’-][A-Za-z]+)*)\s*\(\d+\)")
_DROPPABLE_ONE_SYL = frozenset(
    {"the", "a", "an", "and", "in", "on", "of", "to", "at", "by", "for", "with", "its", "their"}
)


def _extract_surface_words(line: str) -> list[str]:
    """Extract alphabetic tokens preserving original surface capitalization."""
    normalized = (
        line.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u2014", " ")
        .replace("\u2013", " ")
    )
    return [m.group(0) for m in _SURFACE_WORD_RE.finditer(normalized)]


def _solve_word_counts(
    word_counts: Sequence[frozenset[int] | None],
    target: int | None = None,
    avoid_target: int | None = None,
) -> list[int] | None:
    """Backtrack a valid per-word CMUdict syllable count sequence matching `target` (or avoiding it)."""
    if not word_counts or any(wc is None for wc in word_counts):
        return None
    opts = [sorted(wc) for wc in word_counts if wc is not None]
    n = len(opts)

    if target is not None:
        # Suffix min/max pruning makes backtracking O(n) in practice
        suffix_min = [0] * (n + 1)
        suffix_max = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_min[i] = suffix_min[i + 1] + opts[i][0]
            suffix_max[i] = suffix_max[i + 1] + opts[i][-1]

        chosen: list[int] = []

        def dfs(idx: int, rem: int) -> bool:
            if idx == n:
                return rem == 0
            if rem < suffix_min[idx] or rem > suffix_max[idx]:
                return False
            for c in opts[idx]:
                chosen.append(c)
                if dfs(idx + 1, rem - c):
                    return True
                chosen.pop()
            return False

        return chosen if dfs(0, target) else None

    # No fixed target: pick minimal counts, adjusting if sum accidentally equals avoid_target
    chosen = [o[0] for o in opts]
    if avoid_target is not None and sum(chosen) == avoid_target:
        for i, o in enumerate(opts):
            if len(o) > 1:
                chosen[i] = o[1]
                if sum(chosen) != avoid_target:
                    break
        if sum(chosen) == avoid_target:
            return None
    return chosen


def format_line_syllable_trace(
    line: str,
    target: int | None = None,
    avoid_target: int | None = None,
    think_format: str = "sum",
) -> tuple[str, int] | None:
    """Format a single line using exact CMUdict counts in `sum` or `cumulative` format."""
    analysis: LineSyllableAnalysis = analyze_line(line)
    surface_words = _extract_surface_words(line)
    if analysis.has_oov or not surface_words or len(surface_words) != len(analysis.words):
        return None
    counts = _solve_word_counts(analysis.word_counts, target=target, avoid_target=avoid_target)
    if counts is None:
        return None
    if think_format == "cumulative":
        running = 0
        parts: list[str] = []
        for w, c in zip(surface_words, counts):
            running += c
            parts.append(f"{w}({c})={running}")
        return " ".join(parts), running
    total = sum(counts)
    body = " ".join(f"{w}({c})" for w, c in zip(surface_words, counts))
    return f"{body} = {total}", total


def _make_perturbed_draft(line: str, target: int, rng: random.Random) -> str | None:
    """Create a natural off-by-1/2 draft by dropping one short word from a valid line."""
    surface_words = _extract_surface_words(line)
    analysis = analyze_line(line)
    if analysis.has_oov or len(surface_words) < 4:
        return None
    counts = _solve_word_counts(analysis.word_counts, target=target)
    if counts is None:
        return None

    # Prefer dropping a 1-syllable function word (not the very first word) for natural syntax
    candidates: list[int] = [
        i
        for i in range(1, len(surface_words))
        if surface_words[i].lower() in _DROPPABLE_ONE_SYL and counts[i] == 1
    ]
    if not candidates:
        candidates = [
            i for i in range(1, len(surface_words) - 1) if counts[i] in (1, 2)
        ]
    if not candidates:
        return None

    drop_idx = rng.choice(candidates)
    draft_words = [w for i, w in enumerate(surface_words) if i != drop_idx]
    return " ".join(draft_words)


def synthesize_think_trace(
    haiku_lines: Sequence[str],
    targets: tuple[int, int, int] = (5, 7, 5),
    draft_miss_lines: dict[int, str] | None = None,
    revision_prob: float = 0.40,
    think_format: str = "sum",
    forced_rev_line: int | None = None,
    rng: random.Random | None = None,
) -> str:
    """Synthesize a concise CMUdict `<think>...</think>` trace + final 3-line haiku.

    Uses exact surface words (no invented hyphenation) and optionally injects a realistic
    off-by-1/2 self-revision (`draft -> final`) on one line.
    """
    if len(haiku_lines) != len(targets):
        raise ValueError(f"Expected {len(targets)} haiku lines, got {len(haiku_lines)}")
    rng = rng or random.Random(42)

    exact_traces: list[str] = []
    for k, (line, target) in enumerate(zip(haiku_lines, targets)):
        formatted = format_line_syllable_trace(line, target=target, think_format=think_format)
        if formatted is None:
            raise ValueError(f"Line {k + 1} cannot be traced cleanly to {target} syllables: {line!r}")
        exact_traces.append(formatted[0])

    # Optionally augment one line with a realistic off-by-1/2 self-revision
    rev_idx: int | None = None
    rev_draft_trace: str | None = None
    if revision_prob > 0.0 and (forced_rev_line is not None or rng.random() < revision_prob):
        if forced_rev_line is not None:
            weighted_order = [forced_rev_line]
        else:
            # Weight toward Line 2 (7 syllables is hardest), preferring real near-miss drafts
            weighted_order = [1, 0, 2]
            if rng.random() < 0.5:
                weighted_order = [1, 2, 0]
            if rng.random() < 0.35:
                rng.shuffle(weighted_order)

        # First pass: try real near-miss lines collected from the same prompt
        if draft_miss_lines:
            for k in weighted_order:
                cand = draft_miss_lines.get(k)
                if not cand or cand.strip().lower() == haiku_lines[k].strip().lower():
                    continue
                draft_fmt = format_line_syllable_trace(
                    cand,
                    target=None,
                    avoid_target=targets[k],
                    think_format=think_format,
                )
                if draft_fmt is not None and 1 <= abs(draft_fmt[1] - targets[k]) <= 2:
                    rev_idx = k
                    rev_draft_trace = draft_fmt[0]
                    break

        # Fallback: deterministically perturb one line by dropping a 1-2 syllable word
        if rev_idx is None:
            for k in weighted_order:
                perturbed = _make_perturbed_draft(haiku_lines[k], targets[k], rng)
                if not perturbed:
                    continue
                draft_fmt = format_line_syllable_trace(
                    perturbed,
                    target=None,
                    avoid_target=targets[k],
                    think_format=think_format,
                )
                if draft_fmt is not None and 1 <= abs(draft_fmt[1] - targets[k]) <= 2:
                    rev_idx = k
                    rev_draft_trace = draft_fmt[0]
                    break

    think_lines: list[str] = []
    for k, exact_str in enumerate(exact_traces):
        if k == rev_idx and rev_draft_trace is not None:
            think_lines.append(f"{k + 1}: {rev_draft_trace} -> {exact_str}")
        else:
            think_lines.append(f"{k + 1}: {exact_str}")

    haiku_block = "\n".join(ln.strip() for ln in haiku_lines)
    return "<think>\n" + "\n".join(think_lines) + "\n</think>\n" + haiku_block


def _extract_existing_draft_revisions(completion: str) -> dict[int, str]:
    """Extract any pre-`->` draft lines from an existing `<think>` SFT completion."""
    parsed = parse_completion(completion, thinking_mode=True)
    if not parsed.has_valid_think:
        return {}
    drafts: dict[int, str] = {}
    for raw_line in parsed.think_text.splitlines():
        m = _TRACE_LINE_RE.match(raw_line)
        if not m or "->" not in m.group(2):
            continue
        line_idx = int(m.group(1)) - 1
        draft_seg = m.group(2).split("->", 1)[0]
        words = _TRACE_WORD_RE.findall(draft_seg)
        if words:
            drafts[line_idx] = " ".join(words)
    return drafts


def reformat_think_sft_jsonl(
    src_file: str,
    out_file: str,
    think_format: str = "cumulative",
    targets: tuple[int, int, int] = (5, 7, 5),
    seed: int = 42,
) -> list[dict]:
    """Re-wrap an existing verified SFT JSONL into a new `<think>` format (exact 1:1 A/B).

    Preserves the exact same haikus and any existing real near-miss drafts before `->`.
    """
    rng = random.Random(seed)
    reformatted: list[dict] = []
    revisions = 0

    with Path(src_file).open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for row in rows:
        haiku_text = row.get("haiku")
        if not haiku_text:
            parsed = parse_completion(row["completion"], thinking_mode=bool(row.get("thinking_mode")))
            haiku_lines = parsed.lines
        else:
            haiku_lines = tuple(ln.strip() for ln in haiku_text.splitlines() if ln.strip())

        existing_drafts = _extract_existing_draft_revisions(row.get("completion", ""))
        if existing_drafts:
            rev_k = next(iter(existing_drafts))
            new_completion = synthesize_think_trace(
                haiku_lines=haiku_lines,
                targets=targets,
                draft_miss_lines=existing_drafts,
                revision_prob=1.0,
                think_format=think_format,
                forced_rev_line=rev_k,
                rng=rng,
            )
        else:
            new_completion = synthesize_think_trace(
                haiku_lines=haiku_lines,
                targets=targets,
                revision_prob=0.0,
                think_format=think_format,
                rng=rng,
            )
        if "->" in new_completion:
            revisions += 1

        updated = dict(row)
        updated["completion"] = new_completion
        updated["haiku"] = "\n".join(haiku_lines)
        updated["thinking_mode"] = True
        updated["think_format"] = think_format
        reformatted.append(updated)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in reformatted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mean_sim = sum(r.get("raw_cosine_sim", 0.0) for r in reformatted) / max(1, len(reformatted))
    print(
        f"Reformatted {len(reformatted)} SFT examples ({src_file} -> {out_path}) "
        f"[think_format={think_format}, mean_sim={mean_sim:.3f}, revisions={revisions}/{len(reformatted)}]"
    )
    return reformatted


def is_sft_worthy(bd: RewardBreakdown) -> bool:
    """Strict quality filter: keep only exact 5-7-5, on-topic, clean, zero-OOV haikus."""
    return (
        bd.is_valid_format
        and bd.is_exact_575
        and bd.exact_bonus > 0.0
        and len(bd.oov_words) == 0
        and bd.echo_penalty == 0.0
        and bd.repetition_penalty == 0.0
    )


def collect_sft_haikus(
    model_name: str,
    train_file: str,
    out_file: str,
    adapter_path: str | None = None,
    num_prompts: int = 450,
    samples_per_prompt: int = 8,
    prompt_batch_size: int = 8,
    max_tokens: int = 64,
    temp: float = 0.95,
    top_p: float = 0.95,
    max_line_repeats: int = 4,
    max_bigram_repeats: int = 3,
    max_starter_repeats: int = 0,
    min_sim: float = 0.0,
    thinking_mode: bool = False,
    think_format: str = "sum",
    revision_prob: float = 0.40,
    seed: int = 42,
    reward_cfg: RewardConfig | None = None,
    relevance_scorer: RelevanceScorer | None = None,
) -> list[dict]:
    """Generate candidate haikus on training prompts and keep verified exact 5-7-5 winners.

    When `thinking_mode=True`, candidates are sampled from the direct donor model, filtered
    for exact 5-7-5 + diversity + relevance, and programmatically wrapped in CMUdict
    `<think>...</think>` traces (reusing real near-miss lines from the same prompt for
    self-revision augmentation).
    """
    t0 = time.perf_counter()
    # Donor sampling always uses direct 3-line scoring to harvest clean exact haikus + near-misses
    donor_reward_cfg = reward_cfg or RewardConfig(thinking_mode=False)
    if donor_reward_cfg.thinking_mode:
        donor_reward_cfg = RewardConfig(
            relevance_weight=donor_reward_cfg.relevance_weight,
            exact_bonus_min_sim=donor_reward_cfg.exact_bonus_min_sim,
            thinking_mode=False,
        )
    examples = load_jsonl(train_file)
    rng = random.Random(seed)
    rng.shuffle(examples)
    examples = examples[:num_prompts]

    mx.random.seed(seed)
    print(
        f"Rejection sampling from {model_name} (adapter={adapter_path or 'NONE'}, "
        f"thinking_sft={thinking_mode}, min_sim={min_sim:.2f}, max_starter={max_starter_repeats}): "
        f"{len(examples)} prompts x {samples_per_prompt} samples = {len(examples) * samples_per_prompt} candidates..."
    )
    model, tokenizer = load(model_name, adapter_path=adapter_path)
    scorer = relevance_scorer
    if scorer is None and donor_reward_cfg.relevance_weight > 0.0:
        scorer = MLXEmbeddingRelevanceScorer(donor_reward_cfg.embedding_model)

    sampler = make_sampler(temp=temp, top_p=top_p)
    accepted: list[dict] = []
    seen_haikus: set[str] = set()
    line_counts_seen: dict[str, int] = {}
    bigram_counts_seen: dict[tuple[str, str], int] = {}
    starter1_counts_seen: dict[str, int] = {}
    total_candidates = 0
    revisions_synthesized = 0

    for start in range(0, len(examples), prompt_batch_size):
        chunk = examples[start : start + prompt_batch_size]
        flat_prompt_ids: list[list[int]] = []
        flat_prompt_texts: list[str] = []
        for ex in chunk:
            pt = format_chat_prompt(tokenizer, ex.prompt, thinking_mode=False)
            pids = tokenizer.encode(pt, add_special_tokens=False)
            for _ in range(samples_per_prompt):
                flat_prompt_texts.append(pt)
                flat_prompt_ids.append(pids)

        resp = batch_generate(
            model,
            tokenizer,
            prompts=flat_prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            return_token_ids=False,
        )

        # Score per prompt group to reuse cached topic embeddings
        for i, ex in enumerate(chunk):
            offset = i * samples_per_prompt
            raw_slice = resp.texts[offset : offset + samples_per_prompt]
            pt_slice = flat_prompt_texts[offset : offset + samples_per_prompt]
            completions = [
                normalize_completion_text(pt, txt) for pt, txt in zip(pt_slice, raw_slice)
            ]
            breakdowns = score_group(
                topic=ex.topic,
                completions=completions,
                cfg=donor_reward_cfg,
                relevance_scorer=scorer,
            )
            total_candidates += len(breakdowns)

            # Harvest real off-by-1/2 near-miss lines from the same prompt for <think> revisions
            near_miss_pool: dict[int, list[str]] = {0: [], 1: [], 2: []}
            if thinking_mode:
                for bd in breakdowns:
                    if bd.is_valid_format and not bd.oov_words and len(bd.haiku_lines) == 3:
                        for k in range(3):
                            if 1 <= bd.line_distances[k] <= 2:
                                near_miss_pool[k].append(bd.haiku_lines[k])

            # Keep up to 2 best distinct exact 5-7-5 haikus per prompt to preserve topic diversity
            winners = [
                (comp, bd)
                for comp, bd in zip(completions, breakdowns)
                if is_sft_worthy(bd) and (min_sim <= 0.0 or bd.raw_cosine_sim >= min_sim)
            ]
            winners.sort(key=lambda pair: pair[1].raw_cosine_sim, reverse=True)
            kept_for_prompt = 0
            for comp, bd in winners:
                key = "\n".join(bd.haiku_lines).strip().lower()
                if key in seen_haikus:
                    continue
                norm_lines = [ln.strip().lower().rstrip(".,;!?—-") for ln in bd.haiku_lines]
                if max_line_repeats > 0 and any(
                    line_counts_seen.get(ln, 0) >= max_line_repeats for ln in norm_lines
                ):
                    continue
                topic_words = set(tokenize_words(ex.topic))
                bigrams = extract_content_bigrams(key, exclude_words=topic_words)
                if max_bigram_repeats > 0 and any(
                    bigram_counts_seen.get(bg, 0) >= max_bigram_repeats for bg in bigrams
                ):
                    continue
                starters1, _ = extract_line_starters([bd.haiku_lines[0]], exclude_words=topic_words)
                if max_starter_repeats > 0 and any(
                    starter1_counts_seen.get(s1, 0) >= max_starter_repeats for s1 in starters1
                ):
                    continue

                if thinking_mode:
                    draft_misses = {
                        k: rng.choice(pool)
                        for k, pool in near_miss_pool.items()
                        if pool
                    }
                    try:
                        target_completion = synthesize_think_trace(
                            haiku_lines=bd.haiku_lines,
                            targets=donor_reward_cfg.line_targets,
                            draft_miss_lines=draft_misses,
                            revision_prob=revision_prob,
                            think_format=think_format,
                            rng=rng,
                        )
                    except ValueError:
                        continue
                    if "->" in target_completion:
                        revisions_synthesized += 1
                else:
                    target_completion = "\n".join(bd.haiku_lines)

                seen_haikus.add(key)
                for ln in norm_lines:
                    line_counts_seen[ln] = line_counts_seen.get(ln, 0) + 1
                for bg in bigrams:
                    bigram_counts_seen[bg] = bigram_counts_seen.get(bg, 0) + 1
                for s1 in starters1:
                    starter1_counts_seen[s1] = starter1_counts_seen.get(s1, 0) + 1
                accepted.append(
                    {
                        "id": ex.id,
                        "domain": ex.domain,
                        "topic": ex.topic,
                        "prompt": ex.prompt,
                        "completion": target_completion,
                        "haiku": "\n".join(bd.haiku_lines),
                        "thinking_mode": thinking_mode,
                        "total_reward": bd.total_reward,
                        "raw_cosine_sim": bd.raw_cosine_sim,
                        "line_counts": list(bd.line_counts),
                    }
                )
                kept_for_prompt += 1
                if kept_for_prompt >= 2:
                    break

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

        done_prompts = min(start + prompt_batch_size, len(examples))
        if done_prompts % (prompt_batch_size * 5) == 0 or done_prompts == len(examples):
            rate = (len(accepted) / max(1, total_candidates)) * 100.0
            elapsed = time.perf_counter() - t0
            rev_str = f" revisions={revisions_synthesized}" if thinking_mode else ""
            print(
                f"  [collect {done_prompts:4d}/{len(examples)}] "
                f"accepted={len(accepted)} ({rate:.1f}% of {total_candidates}){rev_str} [{elapsed:.1f}s]"
            )

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in accepted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if accepted:
        mean_sim = sum(r["raw_cosine_sim"] for r in accepted) / len(accepted)
        print(
            f"Saved {len(accepted)} verified exact 5-7-5 SFT examples -> {out_path} "
            f"(mean_sim={mean_sim:.3f}"
            + (f", revisions={revisions_synthesized}/{len(accepted)}" if thinking_mode else "")
            + ")"
        )
    else:
        print(f"Saved 0 SFT examples -> {out_path}")
    return accepted


def sft_sequence_loss(
    model: nn.Module,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> mx.array:
    """Mean negative log-likelihood on completion tokens only (prompt tokens masked out)."""
    logprobs = sequence_completion_logprobs(model, prompt_ids, completion_ids)
    return -mx.mean(logprobs)


def train_sft(
    model_cfg: ModelConfig,
    sft_file: str,
    run_name: str = "sft_v1",
    runs_dir: str = "runs",
    epochs: int = 2,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    grad_accum_steps: int = 4,
    grad_clip_norm: float = 1.0,
    thinking_mode: bool = False,
    think_format: str = "sum",
    seed: int = 42,
) -> Path:
    """Run memory-safe LoRA SFT on verified exact 5-7-5 haikus using identical LoRA layers."""
    random.seed(seed)
    mx.random.seed(seed)
    if hasattr(mx, "set_cache_limit"):
        mx.set_cache_limit(2 * (1024**3))

    rows: list[dict] = []
    with Path(sft_file).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No SFT examples found in {sft_file}")

    run_dir = Path(runs_dir) / run_name
    adapters_dir = run_dir / "adapters"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading model for LoRA SFT: {model_cfg.model_name} "
        f"({len(rows)} examples, {epochs} epochs, thinking_mode={thinking_mode}, think_format={think_format})"
    )
    model, tokenizer = build_lora_model(model_cfg)

    # Pre-tokenize all (prompt_ids, completion_ids + eos) pairs
    eos_id = tokenizer.eos_token_id
    tokenized: list[tuple[list[int], list[int]]] = []
    for row in rows:
        pt = format_chat_prompt(
            tokenizer,
            row["prompt"],
            thinking_mode=thinking_mode,
            think_format=think_format,
        )
        pids = tokenizer.encode(pt, add_special_tokens=False)
        cids = tokenizer.encode(row["completion"], add_special_tokens=False)
        if eos_id is not None and (not cids or cids[-1] != eos_id):
            cids = cids + [eos_id]
        if pids and cids:
            tokenized.append((pids, cids))

    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    loss_and_grad_fn = nn.value_and_grad(model, sft_sequence_loss)

    rng = random.Random(seed)
    steps_per_epoch = max(1, (len(tokenized) + grad_accum_steps - 1) // grad_accum_steps)
    total_opt_steps = epochs * steps_per_epoch
    opt_step = 0
    t0 = time.perf_counter()

    with gradient_checkpointing(model, enabled=model_cfg.grad_checkpoint):
        for epoch in range(1, epochs + 1):
            indices = list(range(len(tokenized)))
            rng.shuffle(indices)
            epoch_losses: list[float] = []

            for batch_start in range(0, len(indices), grad_accum_steps):
                batch_idx = indices[batch_start : batch_start + grad_accum_steps]
                inv_n = 1.0 / len(batch_idx)
                accum_grads = None
                batch_loss = 0.0

                for idx in batch_idx:
                    pids, cids = tokenized[idx]
                    loss_val, grads = loss_and_grad_fn(model, pids, cids)
                    scaled = tree_map(lambda g: g * inv_n, grads)
                    if accum_grads is None:
                        accum_grads = scaled
                    else:
                        accum_grads = tree_map(lambda a, b: a + b, accum_grads, scaled)
                    mx.eval(accum_grads, loss_val)
                    batch_loss += float(loss_val.item()) * inv_n

                if accum_grads is not None:
                    opt_step += 1
                    optimizer.learning_rate = cosine_lr(
                        opt_step, total_opt_steps, learning_rate, 0.1 * learning_rate
                    )
                    accum_grads, grad_norm = clip_grad_norm(accum_grads, grad_clip_norm)
                    optimizer.update(model, accum_grads)
                    mx.eval(model.parameters(), optimizer.state)
                    epoch_losses.append(batch_loss)

                    if opt_step == 1 or opt_step % 10 == 0:
                        peak_gb = (
                            mx.get_peak_memory() / (1024**3)
                            if hasattr(mx, "get_peak_memory")
                            else 0.0
                        )
                        print(
                            f"  [sft epoch {epoch}/{epochs} step {opt_step:03d}] "
                            f"nll={batch_loss:.4f} gnorm={grad_norm:.2f} mem={peak_gb:.2f}GB"
                        )

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()

            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            print(
                f"Epoch {epoch}/{epochs} complete: mean_nll={mean_loss:.4f} "
                f"({time.perf_counter() - t0:.1f}s elapsed)"
            )

    final_path = save_adapter(model, model_cfg, adapters_dir / "final")
    print(f"SFT complete! LoRA adapter saved -> {final_path}")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rejection-sampling data collection + LoRA SFT for exact 5-7-5 haikus."
    )
    parser.add_argument("--model", type=str, default=ModelConfig.model_name)
    parser.add_argument("--train-file", type=str, default="data/train.jsonl")
    parser.add_argument("--sft-file", type=str, default="data/sft_haikus.jsonl")
    parser.add_argument("--collect-adapter", type=str, default=None, help="Optional adapter for sampling.")
    parser.add_argument("--skip-collect", action="store_true", help="Reuse existing --sft-file.")
    parser.add_argument("--collect-only", action="store_true", help="Only collect SFT data.")
    parser.add_argument("--num-prompts", type=int, default=450)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--max-line-repeats", type=int, default=4)
    parser.add_argument("--max-bigram-repeats", type=int, default=3)
    parser.add_argument("--max-starter-repeats", type=int, default=0)
    parser.add_argument("--min-sim", type=float, default=0.0)
    parser.add_argument("--thinking-mode", action="store_true")
    parser.add_argument("--think-format", type=str, default="sum", choices=["sum", "cumulative"])
    parser.add_argument("--reformat-from", type=str, default=None, help="Re-wrap existing SFT JSONL into --think-format without re-sampling.")
    parser.add_argument("--revision-prob", type=float, default=0.40)
    parser.add_argument("--run-name", type=str, default="sft_v1")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=ModelConfig.lora_rank)
    parser.add_argument("--lora-scale", type=float, default=ModelConfig.lora_scale)
    parser.add_argument("--lora-layers", type=int, default=ModelConfig.lora_layers)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.reformat_from:
        reformat_think_sft_jsonl(
            src_file=args.reformat_from,
            out_file=args.sft_file,
            think_format=args.think_format,
            seed=args.seed,
        )
    elif not args.skip_collect:
        collect_sft_haikus(
            model_name=args.model,
            train_file=args.train_file,
            out_file=args.sft_file,
            adapter_path=args.collect_adapter,
            num_prompts=args.num_prompts,
            samples_per_prompt=args.samples_per_prompt,
            prompt_batch_size=args.prompt_batch_size,
            max_line_repeats=args.max_line_repeats,
            max_bigram_repeats=args.max_bigram_repeats,
            max_starter_repeats=args.max_starter_repeats,
            min_sim=args.min_sim,
            thinking_mode=args.thinking_mode,
            think_format=args.think_format,
            revision_prob=args.revision_prob,
            seed=args.seed,
        )

    if not args.collect_only:
        model_cfg = ModelConfig(
            model_name=args.model,
            lora_rank=args.lora_rank,
            lora_scale=args.lora_scale,
            lora_layers=args.lora_layers,
        )
        train_sft(
            model_cfg=model_cfg,
            sft_file=args.sft_file,
            run_name=args.run_name,
            runs_dir=args.runs_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            grad_accum_steps=args.grad_accum,
            thinking_mode=args.thinking_mode,
            think_format=args.think_format,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
