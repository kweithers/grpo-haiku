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
    gradient_checkpointing,
    grpo_token_loss,
    kl_k3,
    no_lora,
    normalize_completion_text,
    save_adapter,
)


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


def test_normalize_completion_text_restores_prefilled_think():
    assert normalize_completion_text("...<|im_start|>assistant\n<think>\n", "count syls</think><haiku>a\nb\nc</haiku>").startswith("<think>\n")
    assert normalize_completion_text("...<|im_start|>assistant\n<think>\nL1:", " Soft snow\n</think><haiku>a\nb\nc</haiku>").startswith("<think>\nL1: Soft snow")
    assert normalize_completion_text("...<|im_start|>assistant\n", "<think>hi</think><haiku>a\nb\nc</haiku>").startswith("<think>hi")
