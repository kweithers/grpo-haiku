"""Configuration dataclasses for direct-generation GRPO haiku fine-tuning (16GB M5)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model and LoRA adapter configuration tuned for 16GB unified memory."""

    model_name: str = "mlx-community/Qwen3-1.7B-4bit"
    lora_rank: int = 8
    lora_scale: float = 8.0
    lora_dropout: float = 0.0
    lora_layers: int = 16  # Number of top transformer layers to adapt (-1 for all)
    lora_keys: list[str] = field(
        default_factory=lambda: [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
    )
    grad_checkpoint: bool = True


@dataclass
class RewardConfig:
    """Shaped reward weights and anti-hacking guardrail parameters (no tag rewards)."""

    # Structural line-count penalties (zero free reward for tags/format)
    empty_output_penalty: float = -0.60
    wrong_line_count_penalty_per_line: float = -0.35
    wrong_line_count_penalty_cap: float = -0.60

    # Sharpened 5-7-5 syllable shaping (exact line jump +0.33 > max theme bonus +0.20)
    line_targets: tuple[int, int, int] = (5, 7, 5)
    line_max_reward: float = 0.60  # * 3 lines = 1.80 max
    syllable_error_slope: float = 0.55  # 1 off -> 0.45x (+0.27), 2+ off -> 0.00
    exact_575_bonus: float = 2.50

    # Semantic prompt relevance (tie-breaker & off-topic guardrail, never overrides syllables)
    relevance_weight: float = 0.20
    relevance_min_sim: float = 0.45
    relevance_max_sim: float = 0.78
    exact_bonus_min_sim: float = 0.42  # Blocks off-topic poems without penalizing poetic imagery
    embedding_model: str = "bge-small"

    # Anti-hacking penalties
    oov_penalty_per_word: float = -0.20
    oov_penalty_cap: float = -0.60
    duplicate_line_penalty: float = -0.40
    low_diversity_ratio_threshold: float = 0.65
    low_diversity_penalty: float = -0.25
    monosyllable_ratio_threshold: float = 0.90
    monosyllable_penalty: float = -0.15
    repetition_penalty_cap: float = -0.60
    prompt_echo_ngram: int = 3
    prompt_echo_penalty: float = -0.35

    # Cross-step cliche guardrail (line starters + content bigrams + overused content words)
    cliche_window: int = 64
    cliche_bigram_min_count: int = 3
    cliche_starter1_min_count: int = 5
    cliche_starter2_min_count: int = 3
    cliche_unigram_min_count: int = 8
    cliche_jackpot_decay: float = 0.45  # Each cliche hit multiplies exact_bonus by 0.45x
    cliche_penalty_per_hit: float = -0.15
    cliche_penalty_cap: float = -0.45

    # Within-group fitness sharing (niching): split exact jackpot among near-duplicate templates
    fitness_sharing: bool = True
    fitness_sharing_jaccard: float = 0.45
    fitness_sharing_exponent: float = 0.85


@dataclass
class GRPOConfig:
    """GRPO training loop and memory-management hyperparameters."""

    train_file: str = "data/train.jsonl"
    eval_file: str = "data/eval.jsonl"
    resume_adapter: str | None = None
    run_name: str = "qwen3_1.7b_r2_sft_grpo_v5"
    runs_dir: str = "runs"

    # GRPO rollout & optimization
    num_steps: int = 200
    group_size: int = 8
    grpo_epochs: int = 1
    max_completion_tokens: int = 64
    temperature: float = 0.9
    top_p: float = 0.95
    learning_rate: float = 4e-6
    min_learning_rate: float = 5e-7
    weight_decay: float = 0.01
    kl_beta: float = 0.03
    clip_eps: float = 0.2
    grad_clip_norm: float = 1.0
    adv_eps: float = 1e-6
    line_credit_weight: float = 0.50  # Blend of per-line vs global advantage per token

    # Checkpointing, logging, and memory limits
    eval_every: int = 50
    eval_subset_size: int = 25
    save_every: int = 50
    sample_every: int = 10
    metal_cache_limit_gb: float = 2.0
    seed: int = 42
