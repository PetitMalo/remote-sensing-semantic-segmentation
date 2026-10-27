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
    build_dinov3_linear_probe,
    dinov3_normalization,
    expand_patch_embed,
)
from rsss.training.config import load_train_config

# A DINOv3 ViT-S/16: same architecture and code path, 21 M parameters.
SMALL_MODEL = "vit_small_patch16_dinov3.lvd1689m"
SIZE = 64
CLASSES = 3

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_CONFIG = REPO_ROOT / "configs" / "train_dinov3.yaml"


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
