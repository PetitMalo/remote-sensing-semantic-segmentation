"""Unit tests for the DINOv3 linear-probe model and its shipped recipe.

Every backbone is built with ``pretrained=False``, so the tests stay offline and
fast; fetching the SAT-493M checkpoint is timm's business (and
``rsss.models.weights``'s job on a cluster). The fine-grained mechanics run on
the small web-pretrained variant, which follows the same code path; one test
builds the real 303 M SAT architecture.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import pytest
import timm
import torch
from torch.nn import functional as F

from rsss.models import MODEL_REGISTRY, build_model
from rsss.models.dinov3 import (
    DEFAULT_MODEL,
    SAT_MEAN,
    SAT_STD,
    DinoLinearProbe,
    bitfit_parameters,
    build_dinov3_bitfit,
    build_dinov3_linear_probe,
    dinov3_normalization,
    expand_patch_embed,
    unfreeze_bitfit,
)
from rsss.training.config import load_train_config
from rsss.training.module import SegmentationModule

# A DINOv3 ViT-S/16: same architecture and code path, 21 M parameters.
SMALL_MODEL = "vit_small_patch16_dinov3.lvd1689m"
SIZE = 64
CLASSES = 3
#: Depth of the stand-in above; the ViT-L every real run uses has 24 blocks.
SMALL_DEPTH = 12
#: The tensors BitFit+ trains inside a block: both LayerScale vectors, both
#: LayerNorm affine pairs and the biases that exist (``qkv`` has none).
BITFIT_BLOCK_SUFFIXES = (
    "gamma_1",
    "gamma_2",
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "attn.proj.bias",
    "mlp.fc1.bias",
    "mlp.fc2.bias",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_CONFIG = REPO_ROOT / "configs" / "train_dinov3.yaml"
SHIPPED_BITFIT_CONFIG = REPO_ROOT / "configs" / "train_dinov3_bitfit.yaml"


def small_backbone() -> Any:
    """Return an untrained ViT-S/16, the cheap stand-in for the SAT ViT-L."""
    return timm.create_model(SMALL_MODEL, pretrained=False, num_classes=0)


def small_probe(**overrides) -> DinoLinearProbe:
    """Return the probe on the small backbone, with test-friendly defaults."""
    kwargs = {
        "model_name": SMALL_MODEL,
        "pretrained": False,
        "in_channels": 5,
        "classes": CLASSES,
        "freeze_backbone": True,
        "train_stem": False,
    }
    kwargs.update(overrides)
    # The registry decorator widens the builder's return type to ``nn.Module``.
    return cast(DinoLinearProbe, build_dinov3_linear_probe(**kwargs))


def small_bitfit(**overrides) -> DinoLinearProbe:
    """Return the BitFit+ model on the small backbone, offline and fast."""
    kwargs = {
        "model_name": SMALL_MODEL,
        "pretrained": False,
        "in_channels": 5,
        "classes": CLASSES,
        "grad_checkpointing": False,
    }
    kwargs.update(overrides)
    return cast(DinoLinearProbe, build_dinov3_bitfit(**kwargs))


def expected_bitfit_names(depth: int) -> set[str]:
    """Return the block tensors BitFit+ trains, besides the stem."""
    blocks = {
        f"blocks.{index}.{suffix}"
        for index in range(depth)
        for suffix in BITFIT_BLOCK_SUFFIXES
    }
    return {"norm.weight", "norm.bias"} | blocks


# --------------------------------------------------------------------------- #
# Registry and pretraining statistics                                         #
# --------------------------------------------------------------------------- #


def test_registry_exposes_the_linear_probe():
    assert "dinov3_linear_probe" in MODEL_REGISTRY


def test_the_default_backbone_is_the_satellite_vit():
    assert DEFAULT_MODEL == "vit_large_patch16_dinov3.sat493m"


def test_the_constants_are_the_pretrained_statistics():
    config = timm.get_pretrained_cfg(DEFAULT_MODEL)
    assert config is not None
    assert tuple(config.mean) == SAT_MEAN
    assert tuple(config.std) == SAT_STD


def test_normalization_is_the_sat_statistics_at_uint8_scale():
    means, stds = dinov3_normalization()

    assert means == pytest.approx([109.65, 104.805, 75.48])
    assert stds == pytest.approx([54.315, 39.78, 36.465])


# --------------------------------------------------------------------------- #
# Patch-embedding expansion                                                   #
# --------------------------------------------------------------------------- #


def test_expanding_the_stem_copies_rgb_filters_and_zeroes_the_extra_bands():
    backbone = small_backbone()
    rgb_weight = backbone.patch_embed.proj.weight.detach().clone()
    rgb_bias = backbone.patch_embed.proj.bias.detach().clone()

    expand_patch_embed(backbone, 5)

    conv = backbone.patch_embed.proj
    assert conv.in_channels == 5
    assert torch.equal(conv.weight[:, :3], rgb_weight)
    assert torch.count_nonzero(conv.weight[:, 3:]) == 0
    assert torch.equal(conv.bias, rgb_bias)


def test_expanding_the_stem_rejects_fewer_bands_than_pretrained():
    with pytest.raises(ValueError, match="would drop pretrained filters"):
        expand_patch_embed(small_backbone(), 2)


def test_extra_bands_do_not_change_the_initial_network():
    """At initialization the widened stem must be the RGB model, pixel for pixel."""
    rgb = small_backbone().eval()
    widened = copy.deepcopy(rgb).eval()
    expand_patch_embed(widened, 5)

    with torch.no_grad():
        image = torch.randn(1, 3, SIZE, SIZE)
        padded = torch.cat([image, torch.zeros(1, 2, SIZE, SIZE)], dim=1)
        expected = rgb.forward_features(image)
        actual = widened.forward_features(padded)

    assert torch.allclose(actual, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# Forward, freezing and gradients                                             #
# --------------------------------------------------------------------------- #


def test_the_probe_returns_full_resolution_logits():
    net = small_probe()
    assert net(torch.randn(1, 5, SIZE, SIZE)).shape == (1, CLASSES, SIZE, SIZE)


def test_the_default_backbone_builds_offline():
    net = build_model({"name": "dinov3_linear_probe", "pretrained": False})

    assert isinstance(net, DinoLinearProbe)
    assert net(torch.randn(1, 5, SIZE, SIZE)).shape == (1, 13, SIZE, SIZE)


def test_a_frozen_backbone_leaves_only_the_head_trainable():
    net = small_probe()

    assert all(not p.requires_grad for p in net.backbone.parameters())
    assert all(p.requires_grad for p in net.head.parameters())


def test_train_stem_unfreezes_exactly_the_patch_embedding():
    net = small_probe(train_stem=True)

    assert all(p.requires_grad for p in net.backbone.patch_embed.proj.parameters())
    others = [
        parameter
        for name, parameter in net.backbone.named_parameters()
        if not name.startswith("patch_embed.proj")
    ]
    assert others and all(not p.requires_grad for p in others)


def test_the_extra_band_filters_receive_nonzero_gradients():
    """The NIR/nDSM filters start at zero: only a nonzero gradient can move them."""
    net = small_probe(train_stem=True)
    net.train()

    logits = net(torch.randn(2, 5, SIZE, SIZE))
    labels = torch.randint(0, CLASSES, (2, SIZE, SIZE))
    F.cross_entropy(logits, labels).backward()

    gradient = net.backbone.patch_embed.proj.weight.grad
    assert gradient is not None
    assert gradient[:, :3].abs().sum() > 0  # the pretrained RGB filters
    assert gradient[:, 3:].abs().sum() > 0  # the appended NIR and nDSM filters
    others = [
        parameter.grad
        for name, parameter in net.backbone.named_parameters()
        if not name.startswith("patch_embed.proj")
    ]
    assert all(g is None for g in others)


def test_a_frozen_stem_gets_no_gradient():
    net = small_probe(train_stem=False)

    net(torch.randn(2, 5, SIZE, SIZE)).sum().backward()

    assert net.backbone.patch_embed.proj.weight.grad is None
    assert all(p.grad is not None for p in net.head.parameters())


# --------------------------------------------------------------------------- #
# Parameter groups                                                            #
# --------------------------------------------------------------------------- #


def test_parameter_groups_partition_the_network():
    net = small_probe()

    groups = net.parameter_groups()

    assert set(groups) == {"stem", "backbone", "head"}
    assert {id(p) for p in groups["stem"]} == {
        id(p) for p in net.backbone.patch_embed.proj.parameters()
    }
    assert {id(p) for p in groups["head"]} == {id(p) for p in net.head.parameters()}
    listed = [id(p) for parameters in groups.values() for p in parameters]
    assert len(listed) == len(set(listed))
    assert set(listed) == {id(p) for p in net.parameters()}


# --------------------------------------------------------------------------- #
# BitFit+                                                                     #
# --------------------------------------------------------------------------- #


def test_bitfit_is_registered():
    assert "dinov3_bitfit" in MODEL_REGISTRY


def test_bitfit_parameters_excludes_the_stem_and_lists_each_parameter_once():
    backbone = small_backbone()

    selected = bitfit_parameters(backbone)

    stem = {id(p) for p in backbone.patch_embed.proj.parameters()}
    assert stem.isdisjoint(id(parameter) for parameter in selected)
    assert len(selected) == len({id(parameter) for parameter in selected})
    assert [id(p) for p in selected] == [id(p) for p in bitfit_parameters(backbone)]


def test_unfreeze_bitfit_switches_on_exactly_the_selected_parameters():
    net = small_probe()

    enabled = unfreeze_bitfit(net.backbone)

    assert {id(p) for p in enabled} == {id(p) for p in bitfit_parameters(net.backbone)}
    assert all(parameter.requires_grad for parameter in enabled)


def test_bitfit_plus_trains_the_biases_norms_and_layerscales_of_every_block():
    net = small_bitfit()

    trainable = {name for name, p in net.backbone.named_parameters() if p.requires_grad}

    stem = {"patch_embed.proj.weight", "patch_embed.proj.bias"}
    assert trainable == stem | expected_bitfit_names(SMALL_DEPTH)


def test_bitfit_plus_leaves_every_weight_that_mixes_tokens_frozen():
    net = small_bitfit()

    frozen = {
        name for name, p in net.backbone.named_parameters() if not p.requires_grad
    }

    for name in (
        "blocks.0.attn.qkv.weight",
        "blocks.0.attn.proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.mlp.fc2.weight",
        "blocks.11.attn.qkv.weight",
        "cls_token",
        "reg_token",
    ):
        assert name in frozen


def test_bitfit_plus_gradients_reach_the_first_and_last_block():
    net = small_bitfit()
    net.train()

    logits = net(torch.randn(2, 5, SIZE, SIZE))
    labels = torch.randint(0, CLASSES, (2, SIZE, SIZE))
    F.cross_entropy(logits, labels).backward()

    gradients = {name: p.grad for name, p in net.backbone.named_parameters()}
    for name in (
        "blocks.0.gamma_1",
        "blocks.0.norm1.weight",
        "blocks.11.mlp.fc2.bias",
        "norm.weight",
        "patch_embed.proj.weight",
    ):
        gradient = gradients[name]
        assert gradient is not None and gradient.abs().sum() > 0, name
    assert gradients["blocks.0.attn.qkv.weight"] is None


def test_the_satellite_bitfit_parameter_budget_is_pinned():
    """The ViT-L budget the recipe documents: 296,960 BitFit+ of 1,622,029."""
    net = cast(DinoLinearProbe, build_dinov3_bitfit(pretrained=False))

    groups = net.parameter_groups()

    assert sum(p.numel() for p in groups["bitfit"]) == 296_960
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) == 1_622_029


def test_bitfit_plus_adds_its_own_parameter_group():
    net = small_bitfit()

    groups = net.parameter_groups()

    assert set(groups) == {"bitfit", "stem", "backbone", "head"}
    assert {id(p) for p in groups["bitfit"]} == {
        id(p) for p in bitfit_parameters(net.backbone)
    }
    assert not any(p.requires_grad for p in groups["backbone"])
    listed = [id(p) for parameters in groups.values() for p in parameters]
    assert len(listed) == len(set(listed))
    assert set(listed) == {id(p) for p in net.parameters()}


def test_the_bitfit_group_is_optimized_at_its_own_rate():
    module = SegmentationModule(
        model={
            "name": "dinov3_bitfit",
            "model_name": SMALL_MODEL,
            "pretrained": False,
            "grad_checkpointing": False,
        },
        num_classes=CLASSES,
        optimizer={"lr": 1e-3, "bitfit_lr": 2e-3, "stem_lr": 1e-4},
        scheduler={"warmup_epochs": 0, "t_max": 2},
    )

    optimizer = cast(Any, module.configure_optimizers())["optimizer"]

    # Group order follows parameter_groups(): bitfit, stem, then the head.
    assert [group["lr"] for group in optimizer.param_groups] == [2e-3, 1e-4, 1e-3]


def test_bitfit_refuses_a_backbone_that_is_not_frozen():
    with pytest.raises(ValueError, match="freeze_backbone=False"):
        build_dinov3_bitfit(pretrained=False, freeze_backbone=False)


def test_bitfit_enables_gradient_checkpointing_by_default():
    net = cast(
        DinoLinearProbe, build_dinov3_bitfit(model_name=SMALL_MODEL, pretrained=False)
    )

    assert net.backbone.grad_checkpointing is True
    assert small_bitfit(grad_checkpointing=False).backbone.grad_checkpointing is False


def test_checkpointed_blocks_still_return_full_resolution_logits():
    net = small_bitfit(grad_checkpointing=True)

    assert net(torch.randn(1, 5, SIZE, SIZE)).shape == (1, CLASSES, SIZE, SIZE)


# --------------------------------------------------------------------------- #
# Shipped recipe                                                              #
# --------------------------------------------------------------------------- #


def test_the_shipped_recipe_uses_the_satellite_statistics():
    config = load_train_config(SHIPPED_CONFIG)
    means, stds = dinov3_normalization()

    assert config.model["name"] == "dinov3_linear_probe"
    assert config.model["train_stem"] is True
    assert config.datamodule["means"][:3] == pytest.approx(means)
    assert config.datamodule["stds"][:3] == pytest.approx(stds)
    assert config.optimizer is not None
    assert config.optimizer["stem_lr"] < config.optimizer["lr"]


def test_the_bitfit_recipe_only_differs_in_what_it_trains():
    """Every other knob is shared with the probe, or the comparison is void."""
    probe = load_train_config(SHIPPED_CONFIG)
    bitfit = load_train_config(SHIPPED_BITFIT_CONFIG)

    assert bitfit.model["name"] == "dinov3_bitfit"
    assert bitfit.model["model_name"] == probe.model["model_name"]
    assert bitfit.model["in_channels"] == probe.model["in_channels"]
    assert bitfit.model["train_stem"] is True
    assert bitfit.model["grad_checkpointing"] is True
    assert bitfit.num_classes == probe.num_classes
    assert bitfit.loss == probe.loss
    assert bitfit.scheduler == probe.scheduler
    assert bitfit.trainer == probe.trainer
    assert bitfit.datamodule == probe.datamodule

    assert bitfit.optimizer is not None
    assert probe.optimizer is not None
    assert bitfit.optimizer["bitfit_lr"] == bitfit.optimizer["lr"]
    for key in ("lr", "stem_lr", "weight_decay"):
        assert bitfit.optimizer[key] == probe.optimizer[key]


def test_the_bitfit_recipe_builds_and_segments():
    config = load_train_config(SHIPPED_BITFIT_CONFIG)

    net = build_model({**config.model, "pretrained": False, "classes": CLASSES})

    assert isinstance(net, DinoLinearProbe)
    assert net(torch.randn(1, 5, SIZE, SIZE)).shape == (1, CLASSES, SIZE, SIZE)
