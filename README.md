# grpo-haiku

Training a small, quantized open-weights LLM (**Qwen3-1.7B, 4-bit**) via **RLVR (GRPO)** and **LoRA** to write evocative, on-topic English **5-7-5 haikus** — entirely locally on an **Apple M5 MacBook Pro (16GB Unified Memory)** using Apple MLX.

---

## Why Haikus for RLVR?

LLMs are notoriously bad at haikus: subword tokenization hides syllable boundaries and English phonology, and autoregressive generation makes counting-while-writing surprisingly hard. At the same time, haiku meter is fast and deterministic to verify phonetically — making it an ideal, transparent playground for studying RLVR, reward shaping, credit assignment, and reward hacking.

### Experimental Setup
- **Hardware:** Apple M5 MacBook Pro, 16GB Unified Memory (~2–5 GB peak Metal memory across all runs, zero swap).
- **Base Model:** `mlx-community/Qwen3-1.7B-4bit` (~1 GB 4-bit weights) with attention + MLP LoRA adapters, zero-copy reference KL (swapping tiny LoRA weights instead of loading a second base model), micro-batch=1 gradient accumulation, and scoped gradient checkpointing.
- **Deterministic Verifier:** CMU Pronouncing Dictionary (`cmudict`) with multi-pronunciation dynamic programming (e.g., *fire* ∈ {1, 2}, *every* ∈ {2, 3}), contraction/possessive/hyphenated compound handling, and strict out-of-vocabulary (OOV) penalties rather than gameable vowel heuristics.
- **Semantic Theme Scorer:** Local MLX `BAAI/bge-small-en-v1.5` (~65 MB) cosine similarity between prompt topic and haiku, paired with an n-gram prompt-echo penalty so the model cannot parrot the prompt.
- **Dataset:** 1,000 synthetic prompts stratified across 10 domains (seasons/weather, nature/wildlife, ocean/mountains, urban moments, food/kitchen, home objects, emotions/memory, coding/tech, travel/transit, night/cosmos), split **900 train / 100 held-out eval** with zero topic overlap.

---

## Phase 1: Research & Prototyping

Phase 1 explored thinking vs. direct generation, reward geometry, per-line credit assignment, iterative rejection-sampling SFT → GRPO, and anti-collapse guardrails across 100 unseen held-out topics.

### Held-Out Progression (100 Unseen Topics)

| Run | Mode / Key Change | Exact 5-7-5 | Line 1 / 2 / 3 | Valid Fmt | Relevance | OOV / Echo | Unique | Distinct-2 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (`<think>`)** | Qwen3-1.7B + `<think>` + `<haiku>` | `6.0%` | `22 / 28 / 25%` | `63%` | `0.471` | `3% / 10%` | `97%` | `0.758` |
| **`v1` (`<think>`)** | 120-step GRPO (`G=4`, attn LoRA) | `6.0%` | `49 / 32 / 37%` | `99%` | `0.741` | `9% / 1%` | `100%` | `0.719` |
| **Base (Direct)** | `enable_thinking=False`, no tag rewards | `3.0%` | `29 / 32 / 39%` | `100%` | `0.704` | `6% / 5%` | `100%` | `0.762` |
| **`v2` (Direct)** | 300-step GRPO (`G=8`, MLP LoRA, `lr=1e-5`) | `0.0%` | *collapsed* | `0%` | `0.000` | — | — | — |
| **`v3` (Direct)** | Stabilized (`lr=2e-6`, `scale=8`, `kl=0.05`) | `5.0%` | `31 / 30 / 48%` | `100%` | `0.745` | `4% / 6%` | `100%` | `0.770` |
| **`sft_v1`** | Rejection-sampling SFT (137 exact haikus) | `12.0%` | `48 / 34 / 46%` | `100%` | `0.719` | `3% / 2%` | `100%` | `0.760` |
| **`v4` (R1 GRPO)** | Resumed `sft_v1` + **per-line credit** (`@100`) | `16.0%` | `52 / 39 / 64%` | `100%` | `0.711` | `1% / 3%` | `100%` | `0.710` |
| **`sft_v2`** | R2 SFT (262 exact haikus from `v4`) | `23.0%` | `54 / 48 / 71%` | `100%` | `0.708` | `2% / 2%` | `100%` | `0.663` |
| **`v5` (R2 GRPO)** | **Syllable-first reward** + cosine LR decay | `36.0%` | `68 / 61 / 73%` | `100%` | `0.690` | `2% / 1%` | `100%` | `0.615` |
| **`v6` (R3 GRPO)** | Resumed `v5` + mild `-0.15` cliché penalty | `49.0%` | `84 / 68 / 87%` | `100%` | `0.657` | `1% / 1%` | `98%` | `0.553` |
| **`v7` (from `v6`)** | Resumed `v6` + fitness sharing + DAPO + KL→`sft_v1` | **`80.0%`** | **`99 / 91 / 88%`** | `100%` | `0.585` | `2% / 0%` | `96%` | `0.357` |

### Key Lessons Learned

1. **Unverified scratchpads get reward-hacked immediately (`v1`):**
   - Native Qwen3 thinking mode rambled past 768 tokens before emitting a haiku (`[skip:0var]`).
   - Adding a 1-shot autumn-leaves example fixed truncation — and GRPO promptly discovered a loophole: **paste the 1-shot autumn-leaves poem verbatim inside `<think>`** to escape truncation and collect `+0.30` format reward in 80 tokens, then write the actual `<haiku>` completely blind without counting a single syllable.
   - *Fix:* Never include a copyable poem in the prompt, give zero free reward for XML tags, and either disable thinking (Direct track) or strictly ground `<think>` against the final haiku's words (Grounded `<think>` track).

2. **Short completions + MLP LoRA amplify effective step size ~10× (`v2`):**
   - Dropping from ~170-token thinking traces to ~22-token direct haikus while adding MLP LoRA (`4.98M` params) caused KL to explode from `0.00 → 1.12` in 5 steps at `lr=1e-5`.
   - After breaking formatting, GRPO discovered a second classic exploit: outputting a 4-line poem starting with `"5-7-5 syllable haiku:"` — because digits were ignored, `"syl-la-ble hai-ku"` scored **5 exact syllables**, farming `+0.85` on every prompt forever.
   - *Fix:* Gate syllable credit strictly on valid 3-line digit-free output, lower `lora_scale` (`16 → 8`), and calibrate LR (`2e-6` to `4e-6`) with cosine decay.

3. **Per-line token credit assignment unlocks 7-syllable Line 2 (`sft_v1` → `v4`):**
   - Sequence-level GRPO assigns one scalar advantage to all ~22 tokens, punishing exact lines inside near-miss poems like `(6, 7, 5)`.
   - Mapping tokens to line boundaries (`\n`) and blending global + per-line group advantages (`line_credit_w=0.5`) gave every step 3–4 clean winners per line, lifting Line 2 (`32% → 41%`) and Line 3 (`39% → 66%`) immediately.

4. **Reward geometry matters more than headline weights (`v4` → `v5`):**
   - Even when theme was only `11%` of max reward (`+0.40` vs `+3.60`), its `0.40` swing was **more than 2×** the `+0.18` step from an off-by-1 line to an exact line — so among near-misses, GRPO preferred worse syllables with literal keywords.
   - A strict `0.52` cosine similarity gate also robbed evocative indirect haikus (e.g., *Oak knots weave the grain...* on *"wooden rocking chair"*, `sim=0.51`) of their exact jackpot.
   - *Fix:* Sharpened syllable slopes (`line_max=0.60, slope=0.55`, `+0.33` jump per exact line), dialed theme to a tie-breaker (`0.20`), and lowered the off-topic gate to `0.42` — jumping exact 5-7-5 from `23% → 36%` while holding relevance at `0.69`.

5. **Template collapse is driven by LoRA capacity + the KL ratchet, and anti-collapse guardrails are *preventative*, not curative (`v6` → `v7`):**
   - Across rounds, re-anchoring KL to already-narrowed GRPO checkpoints caused KL to *protect* the `"Nightfall claims the [X]"` 5-syllable shortcut (`2+1+1+1=5`).
   - Once `v6` already had `cl=100%` (every rollout in `G=8` using the template), even strong within-group fitness sharing + starter/word jackpot discounting + DAPO Clip-Higher + KL anchored back to `sft_v1` had **zero fresh rollouts inside the group to reinforce** — so GRPO sharpened the template to **`80.0%` exact 5-7-5 (`99% / 91% / 88%` per line)** while crashing `distinct_2` to **`0.357`**.
   - *Fix for Phase 2:* Start completely clean from the untouched base model with **full-depth LoRA (`all 28 layers, rank=16`)**, lock KL permanently to the clean initial SFT snapshot (never re-anchor to a GRPO checkpoint), and keep fitness sharing + cliché jackpot decay + DAPO Clip-Higher + entropy bonus active from **Step 1** while the policy is broad.

---

## Phase 2: Long Post-Training Runs

Clean, from-scratch post-training runs incorporating every Phase 1 lesson from Step 1, evaluated apples-to-apples on the same 100 held-out topics.

- **Track A — Clean From-Scratch Non-Thinking (Direct) Model:** Detailed protocol in [docs/direct_grpo_plan.md](docs/direct_grpo_plan.md).
- **Track B — Grounded `<think>` Model:** Detailed protocol in [docs/think_grpo_plan.md](docs/think_grpo_plan.md).

### Held-Out Results Comparison (100 Unseen Topics)

| Track / Checkpoint | Exact 5-7-5 | Line 1 / 2 / 3 | Valid Fmt | Relevance | OOV / Echo | Unique | Distinct-2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (Direct, Step 0)** | `3.0%` | `29 / 32 / 39%` | `100%` | `0.704` | `6% / 5%` | `100%` | `0.762` |
| **Track A — Clean Direct SFT** *(all 28L, `r=16`, 229 base haikus)* | `12.0%` | `57 / 38 / 55%` | `100%` | `0.692` | `3% / 0%` | `100%` | `0.719` |
| **Track A — Clean Direct GRPO** *(500 steps, `G=8`, anti-collapse)* | **`31.0%`** | **`74 / 59 / 75%`** | **`100%`** | `0.658` | **`2% / 0%`** | **`100%`** | **`0.714`** |
| **Track B — Grounded `<think>` SFT** | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* |
| **Track B — Grounded `<think>` GRPO** | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* |

#### Track A Findings (500-Step Full-Depth LoRA Direct GRPO)

- **Zero template collapse over 500 steps:** Exact 5-7-5 climbed monotonically across checkpoints (`3.0%` Base → `12.0%` SFT → `21.0%` @200 → `28.0%` @350 → **`31.0%` @500**) while `distinct_2` held steady at **`0.714`** (virtually unchanged from SFT's `0.719`, vs. Phase 1 `v7`'s collapse to `0.357`). `unique_haiku_ratio` remained **`100%`**, `echo_rate` dropped to **`0.0%`**, peak Metal RAM stayed at **`2.71 GB`**, and no Line-1 starter word appeared more than **8%** of the time across all 100 unseen prompts.
- **Genuine meter learning, mostly off by ±1 syllable:** Per-line accuracy reached **`74% / 59% / 75%`** (`0.74 × 0.59 × 0.75 = 32.7% ≈ 31.0%`, confirming independent line composition rather than memorized frames), with **80%** of all unseen haikus landing within ±1 total syllable across the entire 17-syllable poem (`31%` exact + `49%` off by 1).
- **Why Track A plateaued around 31% (next lever):** Even with `distinct_2 > 0.71`, the rolling unigram cliché tracker (`min_count = 8` per 64 rollouts) fired on **74%–88%** of steps on common poetic words (*wind*, *light*, *night*, *still*), dampening the `+2.50` exact-575 jackpot, while `kl_beta = 0.05` permanently pinned `kl ≈ 0.23` near the 12%-accuracy SFT anchor. Relaxing unigram cliché thresholds (keeping starter/bigram decay), running a clean Round 2 rejection-sampling SFT from this uncollapsed `31%` policy, or testing Track B (Grounded `<think>`) are the clearest paths beyond 55% at `distinct_2 >= 0.70`.

### Selected Held-Out Samples (Phase 2 — Track A, `final`)

- **a quiet autumn rainstorm on glass** *(5-7-5, relevance `0.81`)*
  > Rain dashes on glass,
  > silver threads in trembling air,
  > quiet as a sigh.
- **a coastal ferry cutting silver wake** *(5-7-5, relevance `0.77`)*
  > Ferry glides through mist,
  > Silver wake a fleeting light,
  > Sea whispers its night.
- **a sunlit pigeon strutting on brick** *(5-7-5, relevance `0.74`)*
  > Sunlit wings on brick,
  > Shadow stretches, bold and bright,
  > Feathers catch the light.
- **a rugged granite ridge above clouds** *(5-7-5, relevance `0.73`)*
  > Rock crests pierce the sky,
  > Whispers of wind in the stone,
  > Clouds bow to the height.
- **a simmering loaf of sourdough cooling** *(5-7-5, relevance `0.70`)*
  > Bread rises in heat,
  > Flour dreams in golden sleep,
  > Steam slips like a sigh.

---

## Repository Structure

- `src/grpo_haiku/` — Phonetic CMUdict verifier (`syllables.py`), local MLX semantic relevance & prompt-echo detector (`relevance.py`), shaped rewards with fitness sharing & cliché tracking (`rewards.py`), dataset builder (`data.py`), rejection-sampling LoRA SFT (`sft.py`), memory-safe MLX GRPO trainer (`grpo.py`), held-out evaluator (`eval.py`), and interactive verifier (`generate.py`).
- `docs/` — Phase 2 design plans for [direct_grpo_plan.md](docs/direct_grpo_plan.md) and [think_grpo_plan.md](docs/think_grpo_plan.md).
- `data/` — Deterministic stratified 900 train / 100 held-out eval JSONL prompt sets.
- `tests/` — Unit tests covering syllable DP, reward guardrails, dataset splits, SFT masking, per-line credit assignment, fitness sharing, DAPO clipping, entropy bonus, and zero-copy KL swapping.
