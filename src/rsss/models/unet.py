"""UNet baseline, built on segmentation-models-pytorch."""

from __future__ import annotations

from collections.abc import Sequence

import segmentation_models_pytorch as smp
from torch import nn

from rsss.models import register_model


@register_model("unet")
def build_unet(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 5,
    classes: int = 13,
    decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
) -> nn.Module:
    """Build a UNet with a pretrained encoder.

    Parameters
    ----------
    encoder_name : str
        Encoder architecture, e.g. ``"resnet34"``.
    encoder_weights : str, optional
        Pretrained weights, ``"imagenet"`` or None for random init. None is
        what tests and smoke runs should use: it skips the download.
    in_channels : int
        Number of input bands. FLAIR imagery has 5 (RGB + NIR + nDSM), which
        smp handles by adapting the encoder's first convolution.
    classes : int
        Number of output classes.
    decoder_channels : Sequence[int]
        Widths of the five decoder stages.

    Returns
    -------
    nn.Module
        The network, returning raw logits (no activation): the loss applies
        the softmax.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        decoder_channels=tuple(decoder_channels),
        in_channels=in_channels,
        classes=classes,
    )
