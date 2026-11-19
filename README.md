# grpo-haiku

RLVR (GRPO) fine-tuning of small open-weights LLMs (`mlx-community/Qwen3-1.7B-4bit` by default, or `mlx-community/gemma-3-1b-it-4bit`) to write **5-7-5 haikus** using a **`<think>...</think>` scratchpad**, built for **Apple Silicon 16GB Unified Memory** with MLX.

## Quickstart

```bash
# 1. Sync environment (Python 3.12 + MLX)
uv sync --python 3.12 --extra dev

# 2. Run unit tests
uv run pytest -v

# 3. Generate deterministic 900 train / 100 held-out eval prompts
uv run python -m grpo_haiku.data --out-dir data

# 4. Run baseline held-out eval (Step 0, base model)
uv run python -m grpo_haiku.eval --eval-file data/eval.jsonl --out runs/baseline_eval.json

# 5. Train with GRPO (memory-safe on 16GB unified memory)
uv run python -m grpo_haiku.grpo --steps 250 --group-size 4 --run-name qwen3_1.7b_v1

# 6. Evaluate fine-tuned LoRA adapter vs baseline
uv run python -m grpo_haiku.eval \
  --eval-file data/eval.jsonl \
  --adapter-path runs/qwen3_1.7b_v1/adapters/final \
  --out runs/qwen3_1.7b_v1/eval_after.json \
  --compare-with runs/baseline_eval.json

# 7. Try a custom topic interactively
uv run python -m grpo_haiku.generate \
  --adapter-path runs/qwen3_1.7b_v1/adapters/final \
  --topic "a winter snowy morning"
```

## Switching Models or Ablations

- **Try Gemma 3 1B** *(requires Hugging Face access & `HF_TOKEN`)*:
  ```bash
  uv run python -m grpo_haiku.grpo --model mlx-community/gemma-3-1b-it-4bit --run-name gemma3_1b_v1
  ```
- **Ablate prompt relevance shaping**:
  ```bash
  uv run python -m grpo_haiku.grpo --relevance-weight 0.0 --run-name no_relevance
  ```
