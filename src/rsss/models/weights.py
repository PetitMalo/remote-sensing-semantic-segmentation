"""Fetch and cache a model's pretrained weights for offline nodes.

Building a model with ``encoder_weights`` makes ``smp`` download the weights
from the Hugging Face Hub into the local cache the first time; a compute node
without outbound network cannot do that. The same holds for timm, which the
DINOv3 backbone downloads through. This module performs just the fetch, so a
batch script can stage the weights on a login node and the job can then build
the model from the same cache.

Run it with ``uv run python -m rsss.models.weights`` (``-h`` for the options).
"""

from __future__ import annotations

import argparse

import timm
from loguru import logger
from segmentation_models_pytorch.encoders import get_encoder

from rsss.models.dinov3 import DEFAULT_MODEL as DEFAULT_DINOV3

DEFAULT_ENCODER = "resnet34"
DEFAULT_WEIGHTS = "imagenet"


def prefetch_encoder_weights(
    encoder_name: str = DEFAULT_ENCODER,
    weights: str | None = DEFAULT_WEIGHTS,
    in_channels: int = 3,
) -> None:
    """Download ``encoder_name``'s ``weights`` into the local cache.

    The encoder is built once so ``smp`` resolves and caches the weights, then
    discarded: only the cache side effect matters. It is built on three input
    channels, matching the ImageNet weights; ``smp`` re-adapts the first
    convolution when the model is later built with its real band count.

    Parameters
    ----------
    encoder_name : str
        smp encoder name, e.g. ``"resnet34"``.
    weights : str, optional
        Weight set to fetch, e.g. ``"imagenet"``; None is a no-op.
    in_channels : int
        Number of input channels to build the encoder with.

    Raises
    ------
    KeyError
        If ``encoder_name`` is unknown to smp, or has no such weight set.
    """
    if weights is None:
        logger.info("No weights requested for {}; nothing to fetch.", encoder_name)
        return
    get_encoder(encoder_name, in_channels=in_channels, weights=weights)
    logger.info("Cached the {} {} weights.", encoder_name, weights)


def prefetch_dinov3_weights(model_name: str = DEFAULT_DINOV3) -> None:
    """Download timm's ``model_name`` backbone into the local cache.

    Building the backbone is what makes timm resolve its checkpoint and cache
    it, so it is built once and discarded. It is built on three input channels,
    the pretrained shape: the segmentation model widens the stem itself.

    Parameters
    ----------
    model_name : str
        timm name of a DINOv3 backbone, e.g. the SAT-493M ViT-L/16.

    Raises
    ------
    RuntimeError
        If timm knows no such model, or it has no pretrained weights.
    """
    timm.create_model(model_name, pretrained=True, num_classes=0, in_chans=3)
    logger.info("Cached the {} backbone.", model_name)


def main(argv: list[str] | None = None) -> int:
    """Fetch the requested model's pretrained weights into the local cache."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        help="smp encoder name (default: %(default)s).",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="Weight set to fetch (default: %(default)s).",
    )
    parser.add_argument(
        "--dinov3",
        nargs="?",
        const=DEFAULT_DINOV3,
        default=None,
        metavar="MODEL",
        help=(
            "fetch a timm DINOv3 backbone instead of an smp encoder; bare, it "
            f"fetches the satellite ViT (default: {DEFAULT_DINOV3})."
        ),
    )
    args = parser.parse_args(argv)
    if args.dinov3 is not None:
        prefetch_dinov3_weights(args.dinov3)
    else:
        prefetch_encoder_weights(args.encoder, weights=args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
