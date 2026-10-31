"""Unit tests for ``rsss.models.offline_model_config``.

The helper exists so a predictor can rebuild any registered architecture
without downloading pretrained weights; these tests pin the key it sets for
each builder style and make a new model that accepts neither key fail loudly
instead of silently breaking ``rsss.predict``.
"""

from __future__ import annotations

import pytest
from torch import nn

from rsss.models import offline_model_config, register_model


def test_smp_style_builder_gets_encoder_weights_none():
    assert offline_model_config({"name": "unet"})["encoder_weights"] is None


@pytest.mark.parametrize("name", ["dinov3_linear_probe", "dinov3_bitfit"])
def test_timm_style_builder_gets_pretrained_false(name):
    config = offline_model_config({"name": name, "pretrained": True})

    assert config["pretrained"] is False


def test_a_builder_without_an_offline_key_raises():
    """A model accepting neither key can only be built online: say so instead.

    Registered permanently, like ``tiny_seg`` in ``tests/training``: cleaning
    up would restore the ImportError the module guard prevents.
    """

    @register_model("_offline_config_no_key")
    def _builder(in_channels: int = 5, classes: int = 13) -> nn.Module:
        return nn.Identity()

    with pytest.raises(ValueError, match="neither 'encoder_weights' nor 'pretrained'"):
        offline_model_config({"name": "_offline_config_no_key"})


def test_an_unregistered_name_raises():
    with pytest.raises(ValueError, match="registered 'name'"):
        offline_model_config({"name": "not_a_model"})


def test_a_missing_name_raises():
    with pytest.raises(ValueError, match="registered 'name'"):
        offline_model_config({})
