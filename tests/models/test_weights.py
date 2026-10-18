"""Unit tests for the encoder-weight prefetch helper.

The fetch itself is smp's business and needs the network, so these tests stub
``get_encoder`` and assert the prefetch drives it with the requested weights.
"""

from __future__ import annotations

import pytest

from rsss.models import weights


@pytest.fixture
def get_encoder_calls(monkeypatch) -> list[tuple[str, dict]]:
    """Patch ``get_encoder`` in the module and record its calls."""
    calls: list[tuple[str, dict]] = []

    def _get_encoder(name: str, **kwargs) -> object:
        calls.append((name, kwargs))
        return object()

    monkeypatch.setattr(weights, "get_encoder", _get_encoder)
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
