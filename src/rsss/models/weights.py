"""Fetch and cache an encoder's pretrained weights for offline nodes.

Building a model with ``encoder_weights`` makes ``smp`` download the weights
from the Hugging Face Hub into the local cache the first time; a compute node
without outbound network cannot do that. This module performs just the fetch,
so a batch script can stage the weights on a login node and the job can then
build the model from the same cache.

Run it with ``uv run python -m rsss.models.weights`` (``-h`` for the options).
"""

from __future__ import annotations

import argparse

from loguru import logger
from segmentation_models_pytorch.encoders import get_encoder

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


def main(argv: list[str] | None = None) -> int:
    """Fetch the requested encoder weights into the local cache."""
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
    args = parser.parse_args(argv)
    prefetch_encoder_weights(args.encoder, weights=args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
