"""Model registry.

Configs name a model (``"unet"``, ``"dinov3_linear_probe"``) instead of
importing a class, so every architecture is reachable from the same config
schema and the shared training module never changes when one is added.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import nn

ModelBuilder = Callable[..., nn.Module]

MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Return a decorator registering a builder under ``name``.

    Parameters
    ----------
    name : str
        Config key selecting the model, e.g. ``"unet"``.

    Raises
    ------
    ValueError
        If ``name`` is already registered (a silent override would make the
        model actually used depend on import order).
    """

    def _decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model {name!r} is already registered.")
        MODEL_REGISTRY[name] = builder
        return builder

    return _decorator


def build_model(cfg: Mapping[str, Any]) -> nn.Module:
    """Instantiate the model described by ``cfg``.

    Parameters
    ----------
    cfg : Mapping[str, Any]
        Mapping with a ``"name"`` key and the builder's keyword arguments.

    Returns
    -------
    nn.Module
        The network, mapping ``(batch, in_channels, H, W)`` to per-class logits.

    Raises
    ------
    ValueError
        If ``"name"`` is missing or not registered.
    """
    kwargs = dict(cfg)
    if "name" not in kwargs:
        raise ValueError(
            f"Model config needs a 'name' key, one of {sorted(MODEL_REGISTRY)}."
        )
    name = kwargs.pop("name")
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}; available models: {sorted(MODEL_REGISTRY)}."
        )
    return MODEL_REGISTRY[name](**kwargs)


# Imported last: the builders register themselves at import time, and they
# import ``register_model`` back from this partially initialised module.
from rsss.models import dinov3, unet  # noqa: F401
