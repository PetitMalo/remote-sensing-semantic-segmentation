"""Unit tests for the model registry and the UNet builder.

The encoder is built with ``encoder_weights=None`` throughout: it keeps the
tests offline and fast, and the pretrained-weights path is smp's business.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rsss.models import MODEL_REGISTRY, build_model, register_model


def test_registry_exposes_unet():
    assert "unet" in MODEL_REGISTRY


def test_build_model_uses_config_name():
    net = build_model({"name": "unet", "encoder_weights": None})
    assert isinstance(net, nn.Module)


def test_unet_defaults_to_five_bands_and_thirteen_classes():
    net = build_model({"name": "unet", "encoder_weights": None})
    assert net(torch.randn(1, 5, 64, 64)).shape == (1, 13, 64, 64)


def test_unet_logits_keep_the_input_resolution():
    net = build_model(
        {"name": "unet", "encoder_weights": None, "in_channels": 3, "classes": 2}
    )
    assert net(torch.randn(1, 3, 64, 64)).shape == (1, 2, 64, 64)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown model 'dino'"):
        build_model({"name": "dino"})


def test_build_model_requires_a_name():
    with pytest.raises(ValueError, match="needs a 'name' key"):
        build_model({"encoder_name": "resnet34"})


def test_registering_the_same_name_twice_raises():
    name = "test-only-model"
    register_model(name)(lambda: nn.Identity())
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_model(name)(lambda: nn.Identity())
    finally:
        MODEL_REGISTRY.pop(name)
