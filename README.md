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
- **Track B — Grounded `<think>` Model (End-of-Line Sum):** Detailed protocol in [docs/think_grpo_plan.md](docs/think_grpo_plan.md).
- **Track C — Grounded Cumulative `<think>` Model (`word(c)=cum`):** Clean 1:1 A/B upgrade replacing end-of-line mental sums with explicit running totals (`word(c)=cum`) + mild arithmetic consistency check (`-0.20`), trained from untouched base weights on the exact same 498 verified haikus.

### Held-Out Results Comparison (100 Unseen Topics)

| Track / Checkpoint | Exact 5-7-5 | Line 1 / 2 / 3 | Valid Fmt | Relevance | OOV / Echo | Unique | Distinct-2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (Direct, Step 0)** | `3.0%` | `29 / 32 / 39%` | `100%` | `0.704` | `6% / 5%` | `100%` | `0.762` |
| **Track A R1 — Direct SFT** *(all 28L, `r=16`, 229 base haikus)* | `12.0%` | `57 / 38 / 55%` | `100%` | `0.692` | `3% / 0%` | `100%` | `0.719` |
| **Track A R1 — Direct GRPO** *(500 steps, `kl=0.05`, unigram cap on)* | `31.0%` | `74 / 59 / 75%` | `100%` | `0.658` | `2% / 0%` | `100%` | `0.714` |
| **Track A R2 — Direct SFT** *(fresh base LoRA, 547 R1-sampled haikus)* | `50.0%` | `77 / 72 / 81%` | `100%` | `0.665` | `3% / 0%` | `100%` | `0.728` |
| **Track A R2 — Direct GRPO (`@200`, Best Direct)** *(`kl=0.02`, no unigram cap)* | `74.0%` | `85 / 89 / 94%` | `100%` | `0.638` | `0% / 0%` | `99%` | `0.684` |
| **Track A R2 — Direct GRPO (`@300`)** | `73.0%` | `87 / 90 / 91%` | `100%` | `0.640` | `1% / 0%` | `100%` | `0.671` |
| **Track A R2 — Direct GRPO (`@500` final)** | `70.0%` | `86 / 87 / 94%` | `100%` | `0.631` | `4% / 0%` | `100%` | `0.666` |
| **Track B — Grounded `<think>` SFT** *(all 28L, `r=16`, 498 sum traces)* | `61.0%` | `87 / 75 / 87%` | `100%` | `0.690` | `0% / 0%` | `100%` | `0.706` |
| **Track B — Grounded `<think>` GRPO (`@50`)** | `82.0%` | `90 / 87 / 98%` | `99%` | `0.650` | `0% / 0%` | `99%` | `0.628` |
| **Track B — Grounded `<think>` GRPO (`@250`, Best Sum)** | `86.0%` | `97 / 90 / 96%` | `99%` | `0.654` | `0% / 0%` | `100%` | `0.637` |
| **Track B — Grounded `<think>` GRPO (`@300` final)** | `84.0%` | `99 / 85 / 99%` | `100%` | `0.664` | `0% / 0%` | `100%` | `0.644` |
| **Track C — Cumulative `<think>` SFT** *(same 498 haikus, `word(c)=cum`)* | `77.0%` | `93 / 89 / 92%` | `100%` | **`0.704`** | **`0% / 0%`** | `99%` | **`0.702`** |
| **Track C — Cumulative `<think>` GRPO (`@150`, Best Diversity)** | `96.0%` | `99 / 99 / 97%` | `99%` | `0.685` | **`0% / 0%`** | `98%` | **`0.690`** |
| **Track C — Cumulative `<think>` GRPO (`@300` final, Overall Best)** | **`99.0%`** | **`100 / 99 / 100%`** | **`100%`** | **`0.686`** | **`0% / 0%`** | `98%` | `0.673` |

#### Track A Findings (Full-Depth LoRA, Two-Round Direct SFT → GRPO)

- **Round 1 solved template collapse completely (`3% → 12% → 31%`, `distinct_2 = 0.714`):** Full-depth LoRA (`all 28 layers, rank=16`), clean base SFT, permanently locked KL, fitness sharing, cliché decay, DAPO Clip-Higher, and entropy bonus lifted exact 5-7-5 monotonically across 500 steps while keeping `distinct_2 = 0.714` (`100%` unique, peak `2.71 GB` Metal RAM). However, the rolling unigram cliché cap (`min_count = 8` per 64 rollouts) fired on **74%–88%** of steps on normal poetic words (*wind*, *light*, *night*), while `kl_beta = 0.05` pinned the policy near the 12%-accuracy R1 SFT anchor.
- **Round 2 SFT jumped to `50.0%` exact while *increasing* diversity (`distinct_2 = 0.728`):** Rejection-sampling `7,200` candidates from the uncollapsed Round 1 GRPO model with strict deduplication (`max_line_repeats = 2`, `max_bigram_repeats = 3`) yielded **547** balanced, verified 5-7-5 haikus across all 10 domains (no starter > `4.4%`). Training a fresh full-depth LoRA from base on those 547 haikus alone jumped exact 5-7-5 from `31% → 50%` (Line 2: `59% → 72%`) and lifted `distinct_2` to **`0.728`** — providing a far stronger, uncollapsed KL anchor.
- **Round 2 GRPO (`kl_beta = 0.02`, unigram cap disabled) hit `74.0%` exact 5-7-5 at `distinct_2 = 0.684`:** Keeping starter/bigram cliché decay + fitness sharing while removing the unigram cap dropped cliché trigger rate from `~84%` to a healthy `~55%`, driving held-out accuracy to **`74.0%`** (**+71.0%** over base) with **`85% / 89% / 94%`** per-line accuracy, **`0.0%` OOV**, **`0.0%` echo**, **`99%` unique haikus**, **`94%` within ±1 total syllable**, and **no Line-1 starter word over `10%`** — nearly **double** the lexical diversity of Phase 1 `v7` (`0.684` vs. `0.357`) at essentially the same accuracy.

#### Track B Findings (Full-Depth LoRA, Grounded `<think>` SFT → GRPO)

- **Programmatic CMUdict cold-start SFT beat its own donor by `+11%` before a single RL step (`50% → 61%` exact):** Sampling `7,200` candidates from `direct_r2_sft` with strict relevance and starter caps (`min_sim >= 0.60`, `max_starter_repeats = 8`, `max_line_repeats = 2`, `max_bigram_repeats = 2`) yielded **498** balanced exact 5-7-5 haikus (`mean_sim = 0.698`), programmatically wrapped in exact CMUdict word-count traces with **40.8%** (`203/498`) real off-by-1/2 self-revisions (`draft = 6 -> final = 7`). Training a fresh full-depth LoRA from untouched base weights hit **`61.0%` exact 5-7-5**, **`0.690` relevance**, **`0.706` distinct-2**, **`0%` OOV/echo**, and **`99.0%` grounding** (`avg 84.5` `<think>` tokens) out of the box.
- **Grounded Think-GRPO surged to `86.0%` exact 5-7-5 (`97 / 90 / 96%` per line) with genuine phonetic self-correction:** Applying per-line credit assignment directly inside `<think>` (`1:/2:/3:` drafting lines) as well as on the final haiku — alongside `grounding >= 0.70` jackpot gating, `exact_min_sim = 0.48`, `kl_beta = 0.02` locked to `think_sft_v1`, and the Track A R2 anti-collapse stack — lifted exact 5-7-5 from **`61% → 86.0%`** (**+80.0%** over base `<think>`, **+12.0%** over Track A Direct) at just **`2.71 GB`** peak Metal RAM (`~13s/step`).
- **Honest counting & diversity recovery:** Without rewarding scratchpad arithmetic explicitly, **honest CMUdict 5-7-5 count accuracy inside `<think>` climbed from `58.0%` (SFT) → `84.0%` (`@250`)**, grounding stayed at **`99.7%–100.0%`**, `0%` OOV/echo, `100%` unique haikus, `96–97%` within ±1 total syllable, no Line-1 starter exceeded `7–8%`, and `distinct_2` + `relevance` both *climbed* from step 50 (`0.628 / 0.650`) to step 300 (`0.644 / 0.664`).

#### Track C Findings (Clean 1:1 A/B — Cumulative Running-Total `<think>` SFT → GRPO)

- **Error analysis pinpointed Track B's bottleneck:** Across all 14 held-out misses in Track B `@250`, **9 of 14 (64%)** had 100% accurate CMUdict syllable counts per word and failed purely on **end-of-line multi-term mental addition** (e.g., summing 6 word counts to `= 7` when the true sum was `8`).
- **Format alone jumped SFT by `+16.0%` (`61.0% → 77.0%`) on the exact same 498 haikus:** Re-wrapping the exact 498 Track B SFT haikus and 203 real near-miss drafts into running cumulative totals (`Word(c)=cum`) — adding just `+5.3` `<think>` tokens (`84.5 → 89.8`) — lifted SFT exact 5-7-5 from **`61.0% → 77.0%`** (Line 2: `75% → 89%`), honest 5-7-5 counts from **`58.0% → 80.0%`**, arithmetic consistency to **`100.0%`**, relevance to **`0.704`** (matching untouched base), and `distinct_2 = 0.702` — already Pareto-dominating the 2-round Direct GRPO champion (`74%`, `d2 = 0.684`, `rel = 0.638`) before a single step of RL.
- **Cumulative Think-GRPO reached `99.0%` exact 5-7-5 (`100 / 99 / 100%` per line) while *expanding* the Pareto frontier:** Adding a mild pure-integer arithmetic penalty (`-0.20` whenever `cum_i != cum_{i-1} + c_i`) with KL locked to `think_cum_sft_v1` drove held-out accuracy to **`96.0%` at `distinct_2 = 0.690`** (`@150`) and **`99.0%` at `distinct_2 = 0.673`** (`@300` final, **`+4.408`** mean reward out of `+4.50` max, **`0.686`** relevance, **`0.0%`** OOV/echo, **`100.0%`** grounding, **`100.0%`** arithmetic consistency, **`100.0%`** honest CMUdict counts across all 100 prompts, no Line-1 starter > `7%`, peak `2.82 GB` Metal RAM). Even the single held-out miss out of 100 had a 100% accurate `<think>` trace (it simply dropped *"the"* when copying Line 2 out). Because reliable word-by-word counting removes the need to rely on safe syllable templates, Track C simultaneously beat Track B on **accuracy (`+13.0%`), diversity (`+0.036`–`+0.053`), and relevance (`+0.033`)**.

### Selected Held-Out Samples

#### Track A R2 Direct (`step_0200`)

- **a sudden gratitude at sunset** *(5-7-5, relevance `0.81`)*
  > Sunset breaks the sky,
  > Gold and sorrow in the air,
  > Gratitude takes flight.
- **a coastal ferry cutting silver wake** *(5-7-5, relevance `0.78`)*
  > Wave cuts through the mist,
  > Silver wake a fleeting light,
  > Ferry glides in night.
- **a quiet autumn rainstorm on glass** *(5-7-5, relevance `0.74`)*
  > Rain taps on the pane,
  > Silver threads in trembling air,
  > Darkness drinks the light.

#### Track B Grounded `<think>` (`step_0250`, with live self-revisions)

- **a misty autumn rainstorm on glass** *(5-7-5, relevance `0.75`, Line 3 self-revised `6 -> 5`)*
  ```text
  <think>
  1: Rain(1) lashes(2) the(1) glass(1) = 5
  2: Silver(2) trembling(2) in(1) the(1) air(1) = 7
  3: Darkness(2) holds(1) the(1) glass(1) fast(1) = 6 -> Silent(2) cold(1) and(1) still(1) = 5
  </think>
  Rain lashes the glass,
  Silver trembling in the air,
  Silent, cold, and still.
  ```
- **a crowded corner bookstore window** *(5-7-5, relevance `0.65`, Line 2 self-revised `6 -> 7`)*
  ```text
  <think>
  1: Sunlit(2) shelves(1) press(1) tight(1) = 5
  2: Books(1) hum(1) in(1) silent(2) fight(1) = 6 -> Pages(2) flutter(2) in(1) the(1) breeze(1) = 7
  3: Shadowed(2) silence(2) tight(1) = 5
  </think>
  Sunlit shelves press tight,
  Pages flutter in the breeze,
  Shadowed silence, tight.
  ```
- **a sudden gratitude at sunset** *(5-7-5, relevance `0.87`)*
  > Sunset paints the sky,
  > A moment of quiet joy,
  > Gratitude takes flight.
- **a windblown cedar cliff overlooking sea** *(5-7-5, relevance `0.77`)*
  > Cedar sways in air,
  > Sea's blue breath on trembling grass,
  > Shadowed cliff remains.

#### Track C Cumulative `<think>` (`step_0300` final — 99/100 exact, live cumulative self-revisions)

- **a patient heron beside still reeds** *(5-7-5, relevance `0.79`, Line 2 self-revised `8 -> 7`)*
  ```text
  <think>
  1: Fleecy(2)=2 reeds(1)=3 hold(1)=4 still(1)=5
  2: Sunlight(2)=2 bends(1)=3 through(1)=4 feathered(3)=7 stems(1)=8 -> A(1)=1 quiet(2)=3 breath(1)=4 in(1)=5 the(1)=6 air(1)=7
  3: Heron(2)=2 still(1)=3 remains(2)=5
  </think>
  Fleecy reeds hold still,
  A quiet breath in the air,
  Heron still remains.
  ```
- **a moonlit canyon river at twilight** *(5-7-5, relevance `0.77`, Line 2 self-revised `6 -> 7`)*
  ```text
  <think>
  1: Moonlit(2)=2 canyon(2)=4 gleams(1)=5
  2: Whispers(2)=2 of(1)=3 dusk(1)=4 in(1)=5 stone(1)=6 -> Whispers(2)=2 of(1)=3 the(1)=4 river's(2)=6 song(1)=7
  3: Shadowed(2)=2 dawn(1)=3 takes(1)=4 hold(1)=5
  </think>
  Moonlit canyon gleams,
  Whispers of the river's song,
  Shadowed dawn takes hold.
  ```
- **a night sailboat leaning into wind** *(5-7-5, relevance `0.77`, Line 3 self-revised `6 -> 5`)*
  ```text
  <think>
  1: Sea(1)=1 wind(1)=2 bites(1)=3 the(1)=4 mast(1)=5
  2: Shadow(2)=2 leans(1)=3 into(2)=5 the(1)=6 dark(1)=7
  3: Stars(1)=1 catch(1)=2 the(1)=3 wind's(1)=4 last(1)=5 breath(1)=6 -> Night's(1)=1 soft(1)=2 breath(1)=3 in(1)=4 sail(1)=5
  </think>
  Sea wind bites the mast,
  Shadow leans into the dark,
  Night's soft breath in sail.
  ```
- **a moonlit granite ridge above clouds** *(5-7-5, relevance `0.81`)*
  > Moonlit ridge pierces,
  > Clouds weave silver threads and light,
  > Silent vast above.

---

## Repository Structure

- `src/grpo_haiku/` — Phonetic CMUdict verifier (`syllables.py`), local MLX semantic relevance & prompt-echo detector (`relevance.py`), shaped rewards with fitness sharing & cliché tracking (`rewards.py`), dataset builder (`data.py`), rejection-sampling LoRA SFT (`sft.py`), memory-safe MLX GRPO trainer (`grpo.py`), held-out evaluator (`eval.py`), and interactive verifier (`generate.py`).
- `docs/` — Phase 2 design plans for [direct_grpo_plan.md](docs/direct_grpo_plan.md) and [think_grpo_plan.md](docs/think_grpo_plan.md).
- `data/` — Deterministic stratified 900 train / 100 held-out eval JSONL prompt sets.
- `tests/` — Unit tests covering syllable DP, reward guardrails, dataset splits, SFT masking, per-line credit assignment, fitness sharing, DAPO clipping, entropy bonus, and zero-copy KL swapping.
