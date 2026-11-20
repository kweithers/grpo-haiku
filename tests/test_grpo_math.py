"""Unit tests for GRPO math, zero-copy LoRA reference toggle, and gradient checkpointing."""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.lora import LoRALinear

from grpo_haiku.config import ModelConfig
from grpo_haiku.grpo import (
    clip_grad_norm,
    compute_group_advantages,
    compute_per_line_token_advantages,
    gradient_checkpointing,
    grpo_token_loss,
    kl_k3,
    load_lora_weights_into_model,
    map_tokens_to_lines,
    no_lora,
    normalize_completion_text,
    reference_policy,
    save_adapter,
    snapshot_lora_weights,
)
from grpo_haiku.rewards import score_completion


def test_group_advantages_zero_mean_and_zero_variance_skip():
    adv, has_var = compute_group_advantages([1.0, 2.0, 3.0, 4.0])
    assert has_var
    assert abs(sum(adv)) < 1e-5
    assert adv[0] < adv[1] < adv[2] < adv[3]

    zero_adv, has_var_zero = compute_group_advantages([1.5, 1.5, 1.5, 1.5])
    assert not has_var_zero
    assert zero_adv == [0.0, 0.0, 0.0, 0.0]


def test_kl_k3_non_negative_and_zero_when_equal():
    lp = mx.array([-1.2, -0.5, -2.3, -0.1])
    same = kl_k3(lp, lp)
    mx.eval(same)
    assert mx.allclose(same, mx.zeros_like(lp), atol=1e-6).item()

    shifted = mx.array([-1.5, -0.2, -2.8, -0.4])
    diff = kl_k3(lp, shifted)
    mx.eval(diff)
    assert bool(mx.all(diff >= -1e-6).item())
    assert float(mx.mean(diff).item()) > 0.0


def test_grpo_token_loss_direction_and_clipping():
    old_lp = mx.array([-1.0, -1.0, -1.0])
    ref_lp = mx.array([-1.0, -1.0, -1.0])

    # Positive advantage: higher new_lp -> lower loss
    loss_better, _ = grpo_token_loss(mx.array([-0.8, -0.8, -0.8]), old_lp, ref_lp, advantage=1.0, kl_beta=0.0)
    loss_worse, _ = grpo_token_loss(mx.array([-1.2, -1.2, -1.2]), old_lp, ref_lp, advantage=1.0, kl_beta=0.0)
    mx.eval(loss_better, loss_worse)
    assert float(loss_better.item()) < float(loss_worse.item())

    # Clipping caps improvement once ratio exceeds 1 + clip_eps
    loss_far, _ = grpo_token_loss(mx.array([0.0, 0.0, 0.0]), old_lp, ref_lp, advantage=1.0, kl_beta=0.0, clip_eps=0.2)
    mx.eval(loss_far)
    assert abs(float(loss_far.item()) - (-1.2)) < 1e-5


def test_no_lora_zero_copy_toggle_restores_scales():
    base = nn.Linear(8, 8, bias=False)
    lora = LoRALinear.from_base(base, r=2, dropout=0.0, scale=4.0)
    # Give lora_b non-zero weights so LoRA changes output
    lora.lora_b = mx.ones_like(lora.lora_b) * 0.5

    x = mx.ones((1, 8))
    y_with_lora = lora(x)
    y_base_direct = base(x)

    with no_lora(lora):
        y_ref = lora(x)
        assert lora.scale == 0.0

    assert lora.scale == 4.0
    y_after = lora(x)
    mx.eval(y_with_lora, y_base_direct, y_ref, y_after)

    assert bool(mx.allclose(y_ref, y_base_direct, atol=1e-5).item())
    assert bool(mx.allclose(y_after, y_with_lora, atol=1e-5).item())
    assert not bool(mx.allclose(y_with_lora, y_base_direct, atol=1e-3).item())


class _TinyBlock(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.proj = LoRALinear.from_base(nn.Linear(dim, dim, bias=False), r=2, scale=2.0)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.proj(x))


class _TinyModel(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.layers = [_TinyBlock(dim), _TinyBlock(dim)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


def test_scoped_gradient_checkpointing_and_save_adapter(tmp_path):
    model = _TinyModel(8)
    model.freeze()
    for layer in model.layers:
        layer.proj.unfreeze(keys=["lora_a", "lora_b"])

    orig_call = _TinyBlock.__call__
    x = mx.ones((2, 8))

    def loss_fn(m, inp):
        return mx.mean(m(inp) ** 2)

    with gradient_checkpointing(model, enabled=True):
        val, grads = nn.value_and_grad(model, loss_fn)(model, x)
        mx.eval(val, grads)

    # Restored cleanly after context exit
    assert _TinyBlock.__call__ is orig_call

    clipped, norm = clip_grad_norm(grads, max_norm=0.5)
    mx.eval(clipped)
    assert norm >= 0.0

    saved_dir = save_adapter(model, ModelConfig(lora_rank=2, lora_scale=2.0, lora_layers=2), tmp_path / "ckpt")
    assert (saved_dir / "adapters.safetensors").exists()
    cfg = json.loads((saved_dir / "adapter_config.json").read_text())
    assert cfg["fine_tune_type"] == "lora"
    assert cfg["lora_parameters"]["rank"] == 2


def test_normalize_completion_text_strips_whitespace():
    assert normalize_completion_text("prompt", "  Line one\nLine two\nLine three  ") == "Line one\nLine two\nLine three"


class _MockPieceTokenizer:
    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces

    def decode(self, ids: list[int]) -> str:
        return "".join(self._pieces[i] for i in ids)


def test_per_line_token_credit_assignment_rewards_good_lines_independently():
    # Tokens: [0:"Silver", 1:" frosted", 2:" branches\n", 3:"Quiet", 4:" snow\n", 5:"Dawn", 6:" breaks"]
    tok = _MockPieceTokenizer(["Silver", " frosted", " branches\n", "Quiet", " snow\n", "Dawn", " breaks"])
    line_map = map_tokens_to_lines([0, 1, 2, 3, 4, 5, 6], tok)
    assert line_map == [0, 0, 0, 1, 1, 2, 2]

    # Completion A: (6, 7, 5) -> Line 1 wrong, Line 2 exact, Line 3 exact
    # Completion B: (5, 5, 4) -> Line 1 exact, Line 2 wrong, Line 3 wrong
    bd_a = score_completion(
        "winter snow",
        "Silver frosted branches\nQuiet snowflakes drift below\nDawn breaks pale and still",
    )
    bd_b = score_completion(
        "winter snow",
        "Frosted cedar branch\nCold snow falls below\nDawn breaks still",
    )

    tids = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]
    advs, has_var = compute_per_line_token_advantages(
        [bd_a, bd_b], tids, tok, line_credit_weight=0.5
    )
    mx.eval(advs)
    assert has_var
    a_vals = advs[0].tolist()
    b_vals = advs[1].tolist()

    # In A: Line 2 (idx 3,4) and Line 3 (idx 5,6) were exact while Line 1 (idx 0..2) missed -> higher adv on L2/L3 than L1!
    assert a_vals[3] > a_vals[0]
    assert a_vals[5] > a_vals[0]
    # In B: Line 1 was exact while Line 2 & 3 missed -> higher adv on L1 than L2/L3!
    assert b_vals[0] > b_vals[3]
    assert b_vals[0] > b_vals[5]

    # Vector advantage passes cleanly through grpo_token_loss
    lp = mx.array([-1.0] * 7)
    loss, kl = grpo_token_loss(lp, lp, lp, advantage=advs[0], kl_beta=0.05)
    mx.eval(loss, kl)
    assert abs(float(kl.item())) < 1e-6


def test_reference_policy_swaps_and_restores_sft_lora_snapshot(tmp_path):
    model = _TinyModel(8)
    model.freeze()
    for layer in model.layers:
        layer.proj.unfreeze(keys=["lora_a", "lora_b"])
        layer.proj.lora_b = mx.ones_like(layer.proj.lora_b) * 0.25

    saved = save_adapter(model, ModelConfig(lora_rank=2, lora_scale=2.0, lora_layers=2), tmp_path / "sft")

    # Reload into a fresh model, snapshot SFT weights, then mutate training weights
    model2 = _TinyModel(8)
    model2.freeze()
    for layer in model2.layers:
        layer.proj.unfreeze(keys=["lora_a", "lora_b"])
    load_lora_weights_into_model(model2, saved)
    sft_snap = snapshot_lora_weights(model2)

    x = mx.ones((1, 8))
    y_sft = model2(x)

    # Simulate GRPO drift away from SFT
    for layer in model2.layers:
        layer.proj.lora_b = mx.ones_like(layer.proj.lora_b) * 0.9
    y_grpo = model2(x)

    with reference_policy(model2, sft_snap):
        y_ref = model2(x)

    y_after = model2(x)
    mx.eval(y_sft, y_grpo, y_ref, y_after)

    assert bool(mx.allclose(y_ref, y_sft, atol=1e-5).item())
    assert bool(mx.allclose(y_after, y_grpo, atol=1e-5).item())
    assert not bool(mx.allclose(y_grpo, y_sft, atol=1e-3).item())
