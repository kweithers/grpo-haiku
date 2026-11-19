"""Deterministic synthetic haiku prompt dataset generator (900 train / 100 eval)."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_EMPTY_THINK_SUFFIX_RE = re.compile(r"<think>\s*</think>\s*$")

SYSTEM_PROMPT = (
    "You are a skilled haiku poet. Write a vivid, evocative English haiku "
    "with strictly 3 lines following the 5-7-5 syllable structure.\n\n"
    "Always think briefly inside <think>...</think> to draft your lines, count "
    "each word's syllables, and fix any line that is not 5-7-5. Use standard "
    "English words and avoid repeating words or copying the prompt verbatim.\n"
    "Then output ONLY the final 3-line haiku inside <haiku>...</haiku> tags, "
    "with nothing after </haiku>."
)

FEWSHOT_USER = "Write a haiku about falling autumn leaves."
FEWSHOT_ASSISTANT = (
    "<think>\n"
    "L1: Crisp(1) red(1) leaves(1) drift(1) down(1) = 5\n"
    "L2: Cold(1) wind(1) sweeps(1) the(1) dark(1) lane(1) = 6 -> swap 'dark' for 'qui-et'(2) -> Cold(1) wind(1) sweeps(1) the(1) qui-et(2) lane(1) = 7\n"
    "L3: Dusk(1) cools(1) the(1) damp(1) earth(1) = 5\n"
    "</think>\n"
    "<haiku>\n"
    "Crisp red leaves drift down\n"
    "Cold wind sweeps the quiet lane\n"
    "Dusk cools the damp earth\n"
    "</haiku>"
)

PROMPT_TEMPLATES = (
    "Write a haiku about {topic}.",
    "Compose a 5-7-5 haiku capturing {topic}.",
    "Write a vivid haiku inspired by {topic}.",
    "Craft a haiku about {topic}.",
    "Write a poetic haiku describing {topic}.",
)

# 10 rich domains x (10 modifiers x 10 concrete subjects = 100 unique topics per domain -> 1,000 total)
DOMAIN_SPECS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "seasons_weather": (
        (
            "a quiet",
            "an early",
            "a crisp",
            "a golden",
            "a sudden",
            "a misty",
            "a bright",
            "a fading",
            "a gentle",
            "a bitter",
        ),
        (
            "winter snowy morning",
            "autumn rainstorm on glass",
            "spring cherry blossom breeze",
            "summer heatwave at noon",
            "first frost on green grass",
            "melting icicle drip",
            "thunderstorm rolling east",
            "foggy October dawn",
            "April hail on rooftops",
            "late November sunset",
        ),
    ),
    "nature_wildlife": (
        (
            "a solitary",
            "a watchful",
            "a restless",
            "a hidden",
            "a patient",
            "a silver",
            "a wandering",
            "a silent",
            "a darting",
            "a drowsy",
        ),
        (
            "heron beside still reeds",
            "red fox crossing fresh snow",
            "koi swimming under lilies",
            "owl perched in old pines",
            "hummingbird near wild sage",
            "spider spinning dawn silk",
            "salmon leaping cold rapids",
            "deer stepping through ferns",
            "cicada singing at dusk",
            "whale surfacing in mist",
        ),
    ),
    "ocean_mountains": (
        (
            "a distant",
            "a windblown",
            "a sunlit",
            "a shadowed",
            "a rugged",
            "a tranquil",
            "a salt-sprayed",
            "a cloud-wrapped",
            "a tide-carved",
            "a moonlit",
        ),
        (
            "lighthouse above dark waves",
            "granite ridge above clouds",
            "tidepool teeming with crabs",
            "alpine meadow in July",
            "driftwood beach at low tide",
            "glacier valley in silence",
            "canyon river at twilight",
            "fishing boat entering harbor",
            "cedar cliff overlooking sea",
            "waterfall hidden in moss",
        ),
    ),
    "urban_moments": (
        (
            "a rainy",
            "a neon",
            "a crowded",
            "a hushed",
            "a sunlit",
            "a midnight",
            "a sleepy",
            "a bustling",
            "a steam-filled",
            "a windy",
        ),
        (
            "subway platform at dusk",
            "corner bookstore window",
            "rooftop garden in summer",
            "alleyway after rainfall",
            "crosswalk under amber lights",
            "laundromat spinning late",
            "baker opening wooden shutters",
            "bus stop in gray drizzle",
            "jazz note drifting outdoors",
            "pigeon strutting on brick",
        ),
    ),
    "food_kitchen": (
        (
            "a steaming",
            "a fragrant",
            "a freshly baked",
            "a simmering",
            "a quiet",
            "a bitter",
            "a sweet",
            "a warm",
            "a rustic",
            "a late-night",
        ),
        (
            "bowl of miso soup",
            "kettle whistling at dawn",
            "loaf of sourdough cooling",
            "cup of green tea",
            "skillet of roasted chestnuts",
            "slice of ripe summer peach",
            "pot of spiced apple cider",
            "espresso poured in silence",
            "honeycomb dripping on toast",
            "ramen broth in winter",
        ),
    ),
    "home_objects": (
        (
            "an antique",
            "a dusty",
            "a sun-warmed",
            "a forgotten",
            "a handmade",
            "a flickering",
            "a worn",
            "a cracked",
            "a cherished",
            "a ticking",
        ),
        (
            "brass pocket watch",
            "ceramic mug by a window",
            "quilted blanket on oak",
            "fountain pen on linen paper",
            "record spinning softly",
            "paper lantern in breeze",
            "wooden rocking chair",
            "cast-iron teapot on embers",
            "key hanging by the door",
            "glass marble catching sunlight",
        ),
    ),
    "emotions_memory": (
        (
            "a lingering",
            "a quiet",
            "a bittersweet",
            "a sudden",
            "a tender",
            "a fragile",
            "a peaceful",
            "a restless",
            "a hopeful",
            "a distant",
        ),
        (
            "farewell at the gate",
            "childhood summer afternoon",
            "letter left unopened",
            "moment of quiet forgiveness",
            "longing for familiar hills",
            "relief after a hard week",
            "laughter heard through walls",
            "courage before starting over",
            "gratitude at sunset",
            "solitude without loneliness",
        ),
    ),
    "coding_tech": (
        (
            "a quiet",
            "a glowing",
            "a stubborn",
            "a midnight",
            "a clean",
            "a tangled",
            "a sudden",
            "a patient",
            "a humming",
            "a minimalist",
        ),
        (
            "cursor blinking on black",
            "bug solved before sunrise",
            "server rack in cold air",
            "green test suite passing",
            "mechanical keyboard clicking",
            "git merge without conflict",
            "laptop screen in dark room",
            "neural network training slowly",
            "old floppy disk in drawer",
            "coffee cooling beside code",
        ),
    ),
    "travel_transit": (
        (
            "a rattling",
            "a night",
            "a coastal",
            "a lonely",
            "a sunlit",
            "a wooden",
            "a slow",
            "a winding",
            "a foggy",
            "a departing",
        ),
        (
            "train crossing rice fields",
            "ferry cutting silver wake",
            "bicycle coasting downhill",
            "gravel road through orchards",
            "footbridge over a creek",
            "window seat above clouds",
            "sailboat leaning into wind",
            "stone stair climbing a hill",
            "empty highway at daybreak",
            "suitcase waiting by stairs",
        ),
    ),
    "night_cosmos": (
        (
            "a pale",
            "a silver",
            "a sprawling",
            "a silent",
            "a drifting",
            "a radiant",
            "a crescent",
            "a cold",
            "a timeless",
            "a shimmering",
        ),
        (
            "full moon over pine trees",
            "meteor streaking past Orion",
            "aurora rippling above snow",
            "eclipse darkening midday sky",
            "comet tail in deep black",
            "firefly meadow under stars",
            "constellation above desert sand",
            "satellite gliding through dusk",
            "milky way over dark lake",
            "dawn star fading in blue",
        ),
    ),
}


@dataclass(frozen=True)
class HaikuPromptExample:
    """Single prompt example for GRPO training or held-out evaluation."""

    id: str
    domain: str
    topic: str
    prompt: str


def generate_dataset(
    seed: int = 42,
    eval_per_domain: int = 10,
) -> tuple[list[HaikuPromptExample], list[HaikuPromptExample]]:
    """Build deterministic stratified train (900) and held-out eval (100) splits."""
    rng = random.Random(seed)
    train_examples: list[HaikuPromptExample] = []
    eval_examples: list[HaikuPromptExample] = []

    for domain in sorted(DOMAIN_SPECS.keys()):
        modifiers, subjects = DOMAIN_SPECS[domain]
        domain_items: list[HaikuPromptExample] = []
        idx = 0
        for mod in modifiers:
            for subj in subjects:
                topic = f"{mod} {subj}"
                template = PROMPT_TEMPLATES[(idx + len(domain)) % len(PROMPT_TEMPLATES)]
                prompt = template.format(topic=topic)
                domain_items.append(
                    HaikuPromptExample(
                        id=f"{domain}_{idx:03d}",
                        domain=domain,
                        topic=topic,
                        prompt=prompt,
                    )
                )
                idx += 1

        rng.shuffle(domain_items)
        eval_examples.extend(domain_items[:eval_per_domain])
        train_examples.extend(domain_items[eval_per_domain:])

    rng.shuffle(train_examples)
    rng.shuffle(eval_examples)
    return train_examples, eval_examples


def save_jsonl(examples: list[HaikuPromptExample], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[HaikuPromptExample]:
    examples: list[HaikuPromptExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                HaikuPromptExample(
                    id=row["id"],
                    domain=row["domain"],
                    topic=row["topic"],
                    prompt=row["prompt"],
                )
            )
    return examples


def format_chat_prompt(tokenizer, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Format prompt with a concise 1-shot <think>/<haiku> demonstration.

    Uses `enable_thinking=False` when supported (Qwen3) so the chat template preserves
    our concise 1-shot <think> scratchpad instead of triggering Qwen3's 1,000-token
    verbose math-CoT mode, while remaining compatible with Gemma and other models.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": FEWSHOT_USER},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            # Qwen3's `enable_thinking=False` appends an empty `<think>\n\n</think>\n\n`
            # suffix to disable thinking; strip it and prefill `<think>\nL1:` so both
            # Qwen3 and Gemma jump straight into concise syllable counting instead of
            # rambling past the token budget.
            prompt = _EMPTY_THINK_SUFFIX_RE.sub("", prompt)
            return prompt + "<think>\nL1:"
        except TypeError:
            # Gemma and older templates do not accept enable_thinking
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt + "<think>\nL1:"
            except Exception:
                # Some templates (e.g. Gemma) merge system into first user message
                merged = [
                    {"role": "user", "content": f"{system_prompt}\n\n{FEWSHOT_USER}"},
                    {"role": "assistant", "content": FEWSHOT_ASSISTANT},
                    {"role": "user", "content": user_prompt},
                ]
                prompt = tokenizer.apply_chat_template(
                    merged,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt + "<think>\nL1:"
    return (
        f"{system_prompt}\n\n"
        f"User: {FEWSHOT_USER}\nAssistant:\n{FEWSHOT_ASSISTANT}\n\n"
        f"User: {user_prompt}\nAssistant:\n<think>\nL1:"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic haiku train/eval JSONL datasets.")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory for JSONL files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    parser.add_argument("--eval-per-domain", type=int, default=10, help="Held-out examples per domain.")
    args = parser.parse_args()

    train, eval_set = generate_dataset(seed=args.seed, eval_per_domain=args.eval_per_domain)
    out_dir = Path(args.out_dir)
    save_jsonl(train, out_dir / "train.jsonl")
    save_jsonl(eval_set, out_dir / "eval.jsonl")
    print(f"Saved {len(train)} train prompts -> {out_dir / 'train.jsonl'}")
    print(f"Saved {len(eval_set)} eval prompts  -> {out_dir / 'eval.jsonl'}")


if __name__ == "__main__":
    main()
