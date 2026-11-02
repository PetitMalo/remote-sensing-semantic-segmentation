"""Unit tests for the LoRA wrapper and the helper that installs it.

The timm backbone is built with ``pretrained=False`` on the small DINOv3
ViT-S/16 -- the same code path as the ViT-L every real run uses -- so the
tests stay offline and fast.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import timm
import torch
from torch import nn

from rsss.models.lora import (
    DEFAULT_TARGETS,
    LoRALinear,
    apply_lora,
    lora_modules,
    lora_parameters,
)

#: A DINOv3 ViT-S/16: same architecture as the ViT-L, 12 blocks, 21 M params.
SMALL_MODEL = "vit_small_patch16_dinov3.lvd1689m"
SMALL_DEPTH = 12
IN_FEATURES = 6
OUT_FEATURES = 10
RANK = 4
ALPHA = 8.0
#: What the adapters cost on the small ViT: 12 blocks of qkv (384->1152) and
#: proj (384->384), each r * (in + out).
SMALL_BUDGET = SMALL_DEPTH * RANK * (384 + 1152 + 384 + 384)


@pytest.fixture(scope="module")
def small_vit() -> Any:
    """A frozen-size DINOv3 ViT-S, rebuilt per test by deep copy."""
    return timm.create_model(SMALL_MODEL, pretrained=False, num_classes=0)


@pytest.fixture
def linear() -> nn.Linear:
    """A small pretrained-style linear layer."""
    return nn.Linear(IN_FEATURES, OUT_FEATURES)


def test_a_fresh_adapter_is_exactly_the_frozen_layer(linear: nn.Linear) -> None:
    """B starts at zero, so the wrapper computes what the pretrained layer did."""
    wrapped = LoRALinear(linear)
    x = torch.randn(3, IN_FEATURES)

    assert torch.equal(wrapped(x), linear(x))


def test_the_base_layer_is_frozen_and_only_the_adapters_train(
    linear: nn.Linear,
) -> None:
    wrapped = LoRALinear(linear)

    assert not any(parameter.requires_grad for parameter in wrapped.base.parameters())
    trainable = [
        name
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable == ["lora_A", "lora_B"]


def test_the_adapters_have_the_low_rank_shapes(linear: nn.Linear) -> None:
    wrapped = LoRALinear(linear, rank=RANK)
    out = wrapped(torch.randn(2, IN_FEATURES))

    assert wrapped.lora_A.shape == (RANK, IN_FEATURES)
    assert wrapped.lora_B.shape == (OUT_FEATURES, RANK)
    assert wrapped.in_features == IN_FEATURES
    assert wrapped.out_features == OUT_FEATURES
    assert out.shape == (2, OUT_FEATURES)


def test_the_correction_is_scaled_by_alpha_over_rank(linear: nn.Linear) -> None:
    wrapped = LoRALinear(linear, rank=RANK, alpha=ALPHA)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, IN_FEATURES, generator=generator)
    with torch.no_grad():
        wrapped.lora_A.copy_(torch.randn(RANK, IN_FEATURES, generator=generator))
        wrapped.lora_B.copy_(torch.randn(OUT_FEATURES, RANK, generator=generator))

    expected = linear(x) + (x @ wrapped.lora_A.T @ wrapped.lora_B.T) * (ALPHA / RANK)

    assert torch.allclose(wrapped(x), expected)


def test_dropout_only_regularises_the_correction_in_training(linear: nn.Linear) -> None:
    wrapped = LoRALinear(linear, dropout=0.5)
    with torch.no_grad():
        wrapped.lora_B.fill_(1.0)
    x = torch.randn(64, IN_FEATURES)

    wrapped.eval()
    assert torch.equal(wrapped(x), wrapped(x))

    wrapped.train()
    torch.manual_seed(0)
    first = wrapped(x)
    torch.manual_seed(1)
    second = wrapped(x)
    assert not torch.equal(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rank": 0}, "rank must be a positive integer"),
        ({"rank": -2}, "rank must be a positive integer"),
        ({"alpha": 0.0}, "alpha must be positive"),
        ({"dropout": 1.0}, r"dropout must be in \[0, 1\)"),
        ({"dropout": -0.1}, r"dropout must be in \[0, 1\)"),
    ],
)
def test_invalid_adapter_settings_raise(
    linear: nn.Linear, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LoRALinear(linear, **kwargs)


def test_applying_lora_wraps_attention_in_every_block(small_vit: Any) -> None:
    model = copy.deepcopy(small_vit)
    apply_lora(model, rank=RANK, alpha=ALPHA)

    wrapped = {
        name for name, module in model.named_modules() if isinstance(module, LoRALinear)
    }

    assert wrapped == {
        f"blocks.{index}.{suffix}"
        for index in range(SMALL_DEPTH)
        for suffix in ("attn.qkv", "attn.proj")
    }
    assert type(model.blocks[0].mlp.fc1) is nn.Linear
    assert type(model.blocks[0].mlp.fc2) is nn.Linear


def test_the_patch_embedding_is_not_a_target(small_vit: Any) -> None:
    """``attn.proj`` must not select ``patch_embed.proj``: suffixes, not substrings."""
    model = copy.deepcopy(small_vit)
    apply_lora(model, rank=RANK, alpha=ALPHA)

    assert isinstance(model.patch_embed.proj, nn.Conv2d)


def test_applying_lora_does_not_change_the_forward_output(
    small_vit: Any,
) -> None:
    model = copy.deepcopy(small_vit)
    model.eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        before = model.forward_intermediates(
            x, indices=1, norm=True, output_fmt="NCHW", intermediates_only=True
        )[0]
        apply_lora(model, rank=RANK, alpha=ALPHA)
        after = model.forward_intermediates(
            x, indices=1, norm=True, output_fmt="NCHW", intermediates_only=True
        )[0]

    assert torch.equal(before, after)


def test_applying_lora_returns_the_adapter_parameters(small_vit: Any) -> None:
    model = copy.deepcopy(small_vit)
    returned = apply_lora(model, rank=RANK, alpha=ALPHA)

    assert [id(parameter) for parameter in returned] == [
        id(parameter) for parameter in lora_parameters(model)
    ]
    assert all(isinstance(parameter, nn.Parameter) for parameter in returned)
    assert all(parameter.requires_grad for parameter in returned)


def test_the_adapter_budget_follows_the_rank_and_the_layer_widths(
    small_vit: Any,
) -> None:
    model = copy.deepcopy(small_vit)
    apply_lora(model, rank=RANK, alpha=ALPHA)

    size = sum(parameter.numel() for parameter in lora_parameters(model))

    assert size == SMALL_BUDGET


def test_gradients_reach_the_adapters_but_not_the_base(small_vit: Any) -> None:
    model = copy.deepcopy(small_vit)
    apply_lora(model, rank=RANK, alpha=ALPHA)
    modules = lora_modules(model)

    model(torch.randn(1, 3, 64, 64)).sum().backward()

    assert all(module.lora_A.grad is not None for module in modules)
    assert all(module.lora_B.grad is not None for module in modules)
    assert all(
        parameter.grad is None
        for module in modules
        for parameter in module.base.parameters()
    )


def test_an_unknown_target_raises(small_vit: Any) -> None:
    model = copy.deepcopy(small_vit)

    with pytest.raises(ValueError, match=r"No module matches LoRA target\(s\)"):
        apply_lora(model, targets=("attn.quv",), rank=RANK, alpha=ALPHA)

    assert not lora_modules(model)


def test_a_partially_matched_target_list_raises_before_adapting(
    small_vit: Any,
) -> None:
    """A typo next to a valid target must not leave the model half-adapted."""
    model = copy.deepcopy(small_vit)

    with pytest.raises(ValueError, match="attn.quv"):
        apply_lora(
            model, targets=(*DEFAULT_TARGETS, "attn.quv"), rank=RANK, alpha=ALPHA
        )

    assert not lora_modules(model)


def test_a_non_linear_target_raises() -> None:
    model = nn.Module()
    attention = nn.Module()
    attention.add_module("qkv", nn.ReLU())
    model.add_module("attn", attention)

    with pytest.raises(TypeError, match="not nn.Linear"):
        apply_lora(model, targets=("attn.qkv",), rank=RANK, alpha=ALPHA)


def test_lora_modules_are_reported_in_module_order(small_vit: Any) -> None:
    model = copy.deepcopy(small_vit)
    apply_lora(model, rank=RANK, alpha=ALPHA)

    names = [
        name for name, module in model.named_modules() if isinstance(module, LoRALinear)
    ]

    assert names[:2] == ["blocks.0.attn.qkv", "blocks.0.attn.proj"]
    assert names[-2:] == [
        f"blocks.{SMALL_DEPTH - 1}.attn.qkv",
        f"blocks.{SMALL_DEPTH - 1}.attn.proj",
    ]


def test_the_adapter_dtype_and_device_follow_the_base() -> None:
    base = nn.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.float64)
    wrapped = LoRALinear(base, rank=RANK, alpha=ALPHA)
    x = torch.randn(2, IN_FEATURES, dtype=torch.float64)

    out = wrapped(x)

    assert out.dtype == torch.float64
    assert wrapped.lora_A.device == base.weight.device


def test_a_batch_of_one_input_still_produces_the_shapes() -> None:
    """The correction is a matmul, so a single row must not be squeezed."""
    wrapped = LoRALinear(nn.Linear(IN_FEATURES, OUT_FEATURES), rank=2)

    out = wrapped(torch.randn(1, IN_FEATURES))

    assert out.shape == (1, OUT_FEATURES)
