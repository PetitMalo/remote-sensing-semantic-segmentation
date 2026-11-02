"""Low-rank adaptation (LoRA) of a frozen backbone's linear layers.

LoRA freezes a pretrained weight ``W`` and learns a low-rank correction
``B @ A`` beside it, so a linear layer computes ``W x + (alpha / rank) * B A x``.
``B`` starts at zero, so the adapted network is exactly the pretrained one at
initialisation -- the same property that makes the frozen probe a fair
measurement of the features themselves.

Adapting attention is enough to let token mixing move while the backbone stays
frozen: the MLP and every weight the pretrained model already has keep their
values. That places LoRA between BitFit+ (biases and scales only, no token
mixing) and a whole-block fine-tune (every weight of the last blocks), for a
few million parameters rather than tens of millions.

Only the module mechanics live here. Which parameters a recipe trains, and how
its parameter groups are built, belong to the builder in
:mod:`rsss.models.dinov3`. Merging the adapters back into the base weights for
an export is deliberately not implemented yet: inference through the wrapper
computes the same function.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

import torch
from torch import Tensor, nn

#: Attention's linear maps, the layers DINOv3's LoRA fine-tuning adapts.
DEFAULT_TARGETS = ("attn.qkv", "attn.proj")


class _Module(Protocol):
    """The slice of ``nn.Module`` these helpers rely on.

    Typed structurally so a backbone described by a protocol -- as DINOv3's
    is in :mod:`rsss.models.dinov3` -- can be passed without a cast.
    """

    def named_modules(self) -> Iterator[tuple[str, nn.Module]]: ...

    def modules(self) -> Iterator[nn.Module]: ...

    def get_submodule(self, target: str) -> nn.Module: ...

    def parameters(self) -> Iterator[nn.Parameter]: ...

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]: ...


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank correction.

    Parameters
    ----------
    base : nn.Linear
        The pretrained layer. It is frozen here, so a caller that already
        froze the network does not have to do anything else.
    rank : int
        Rank of the correction, at least one.
    alpha : float
        Numerator of the ``alpha / rank`` scale applied to the correction.
        ``2 * rank`` is the common default: it keeps the initial correction's
        magnitude independent of the rank.
    dropout : float
        Dropout probability applied to the input of the correction only, in
        ``[0, 1)``. It regularizes the adapters without touching the frozen
        path.

    Raises
    ------
    ValueError
        If ``rank``, ``alpha`` or ``dropout`` is out of its range.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(rank, int) or rank < 1:
            raise ValueError(f"rank must be a positive integer, got {rank!r}.")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha!r}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}.")

        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        # No-op at 0.0, which keeps the wrapper deterministic without a branch
        # in the forward pass.
        self.dropout = nn.Dropout(dropout)
        options: dict[str, Any] = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, **options),
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, **options),
        )
        # A in the base layer's own init distribution and B at zero: the
        # correction starts at exactly zero but its gradient does not.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        """Width of the layer's input."""
        return self.base.in_features

    @property
    def out_features(self) -> int:
        """Width of the layer's output."""
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        """Return the frozen layer's output plus the scaled low-rank correction."""
        correction = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + correction * self.scaling

    def extra_repr(self) -> str:
        """Describe the adapter in a module listing."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"dropout={self.dropout.p}"
        )


def _matches(name: str, target: str) -> bool:
    """Return whether a module's dotted ``name`` is ``target`` or ends with it.

    Matching on whole dot-separated suffixes, not on raw substrings, is what
    keeps ``attn.proj`` from also selecting ``patch_embed.proj``.
    """
    return name == target or name.endswith(f".{target}")


def apply_lora(
    model: _Module,
    targets: Sequence[str] = DEFAULT_TARGETS,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
) -> list[nn.Parameter]:
    """Replace the linear layers ``targets`` names with LoRA wrappers.

    Parameters
    ----------
    model : nn.Module
        The network to adapt, typically a timm ViT whose pretrained weights are
        already loaded.
    targets : Sequence[str], optional
        Dotted suffixes of the modules to wrap, e.g. ``"attn.qkv"``. Each
        suffix selects every module whose name ends with it -- all blocks, for
        a ViT.
    rank, alpha, dropout : int, float, float
        Adapter settings, as :class:`LoRALinear` takes them.

    Returns
    -------
    list[nn.Parameter]
        The adapter parameters switched on, in module order; the base layers
        stay frozen.

    Raises
    ------
    ValueError
        If a target matches no module -- a typo would otherwise train a model
        that is silently still frozen -- or an adapter setting is invalid.
    TypeError
        If a matched module is not a linear layer.
    """
    # Collect before wrapping: replacing a module while iterating
    # ``named_modules`` would change the traversal under the generator, and
    # checking every target first is what leaves the model untouched when one
    # of them is a typo.
    found: list[tuple[str, nn.Module]] = []
    matched = {target: False for target in targets}
    for name, module in model.named_modules():
        for target in targets:
            if _matches(name, target):
                found.append((name, module))
                matched[target] = True

    unmatched = sorted(target for target, seen in matched.items() if not seen)
    if unmatched:
        raise ValueError(
            f"No module matches LoRA target(s) {unmatched} in "
            f"{model.__class__.__name__}; the available layers' names should "
            f"end with every one of {sorted(targets)}."
        )

    adapters: list[nn.Parameter] = []
    for name, module in found:
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"LoRA target matched {name!r}, which is "
                f"{module.__class__.__name__}, not nn.Linear."
            )
        parent_path, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        wrapper = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attribute, wrapper)
        adapters.extend((wrapper.lora_A, wrapper.lora_B))
    return adapters


def lora_modules(model: _Module) -> list[LoRALinear]:
    """Return the LoRA wrappers inside ``model``, in module order."""
    return [module for module in model.modules() if isinstance(module, LoRALinear)]


def lora_parameters(model: _Module) -> list[nn.Parameter]:
    """Return the adapter parameters inside ``model``, in module order."""
    return [
        parameter
        for module in lora_modules(model)
        for parameter in (module.lora_A, module.lora_B)
    ]
