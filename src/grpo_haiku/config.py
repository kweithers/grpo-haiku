"""Configuration dataclasses for GRPO haiku fine-tuning on Apple Silicon (16GB)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model and LoRA adapter configuration tuned for 16GB unified memory."""

    model_name: str = "mlx-community/Qwen3-1.7B-4bit"
    lora_rank: int = 8
    lora_scale: float = 16.0
    lora_dropout: float = 0.0
    lora_layers: int = 16  # Number of top transformer layers to adapt (-1 for all)
    lora_keys: list[str] = field(
        default_factory=lambda: [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]
    )
    grad_checkpoint: bool = True


@dataclass
class RewardConfig:
    """Shaped reward weights and anti-hacking guardrail parameters."""

    # Format reward
    think_tag_reward: float = 0.15
    haiku_tag_reward: float = 0.15
    missing_haiku_penalty: float = -0.50
    trailing_text_penalty: float = -0.10

    # Syllable shaping
    line_targets: tuple[int, int, int] = (5, 7, 5)
    line_max_reward: float = 0.40  # * 3 lines = 1.20 max
    syllable_error_slope: float = 0.25  # 1 off -> 0.75x, 2 off -> 0.50x, 4+ off -> 0
    exact_575_bonus: float = 1.00

    # Semantic prompt relevance
    relevance_weight: float = 0.50
    relevance_min_sim: float = 0.45
    relevance_max_sim: float = 0.78
    exact_bonus_min_sim: float = 0.52  # Off-topic 5-7-5 poems cannot farm the jackpot
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
    fewshot_copy_penalty: float = -1.00


@dataclass
class GRPOConfig:
    """GRPO training loop and memory-management hyperparameters."""

    train_file: str = "data/train.jsonl"
    eval_file: str = "data/eval.jsonl"
    run_name: str = "qwen3_1.7b_v1"
    runs_dir: str = "runs"

    # GRPO rollout & optimization
    num_steps: int = 250
    group_size: int = 4
    grpo_epochs: int = 1
    max_completion_tokens: int = 512
    temperature: float = 0.9
    top_p: float = 0.95
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    kl_beta: float = 0.04
    clip_eps: float = 0.2
    grad_clip_norm: float = 1.0
    adv_eps: float = 1e-6

    # Checkpointing, logging, and memory limits
    eval_every: int = 50
    eval_subset_size: int = 20
    save_every: int = 50
    sample_every: int = 10
    metal_cache_limit_gb: float = 2.0
    seed: int = 42
