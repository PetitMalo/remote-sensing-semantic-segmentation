"""Unit tests for the pretrained-weight prefetch helpers.

The fetch itself is smp's or timm's business and needs the network, so these
tests stub ``get_encoder`` and ``timm.create_model`` and assert the prefetch
drives them with the requested weights.
"""

from __future__ import annotations

import pytest

from rsss.models import weights
from rsss.models.dinov3 import DEFAULT_MODEL


@pytest.fixture
def get_encoder_calls(monkeypatch) -> list[tuple[str, dict]]:
    """Patch ``get_encoder`` in the module and record its calls."""
    calls: list[tuple[str, dict]] = []

    def _get_encoder(name: str, **kwargs) -> object:
        calls.append((name, kwargs))
        return object()

    monkeypatch.setattr(weights, "get_encoder", _get_encoder)
    return calls


@pytest.fixture
def create_model_calls(monkeypatch) -> list[tuple[str, dict]]:
    """Patch ``timm.create_model`` and record its calls."""
    calls: list[tuple[str, dict]] = []

    def _create_model(name: str, **kwargs) -> object:
        calls.append((name, kwargs))
        return object()

    monkeypatch.setattr(weights.timm, "create_model", _create_model)
    return calls


def test_prefetch_requests_the_default_weights(get_encoder_calls):
    weights.prefetch_encoder_weights("resnet34")

    assert get_encoder_calls == [
        ("resnet34", {"in_channels": 3, "weights": "imagenet"})
    ]


def test_prefetch_builds_on_the_requested_channel_count(get_encoder_calls):
    weights.prefetch_encoder_weights("resnet34", in_channels=5)

    assert get_encoder_calls[0][1]["in_channels"] == 5


def test_prefetch_without_weights_is_a_noop(get_encoder_calls):
    weights.prefetch_encoder_weights("resnet34", weights=None)

    assert get_encoder_calls == []


def test_main_fetches_the_requested_encoder(get_encoder_calls):
    assert weights.main(["--encoder", "resnet50", "--weights", "imagenet"]) == 0

    assert get_encoder_calls == [
        ("resnet50", {"in_channels": 3, "weights": "imagenet"})
    ]


def test_main_uses_documented_defaults(get_encoder_calls):
    assert weights.main([]) == 0

    assert get_encoder_calls[0][0] == weights.DEFAULT_ENCODER


def test_dinov3_prefetch_builds_the_pretrained_rgb_model(create_model_calls):
    weights.prefetch_dinov3_weights()

    assert create_model_calls == [
        (
            DEFAULT_MODEL,
            {"pretrained": True, "num_classes": 0, "in_chans": 3},
        )
    ]


def test_main_bare_dinov3_fetches_the_satellite_backbone(
    create_model_calls, get_encoder_calls
):
    assert weights.main(["--dinov3"]) == 0

    assert create_model_calls[0][0] == DEFAULT_MODEL
    assert get_encoder_calls == []


def test_main_dinov3_takes_a_model_name(create_model_calls):
    assert weights.main(["--dinov3", "vit_small_patch16_dinov3.lvd1689m"]) == 0

    assert create_model_calls[0][0] == "vit_small_patch16_dinov3.lvd1689m"
