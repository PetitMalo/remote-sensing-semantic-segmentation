"""DINOv3 satellite backbone with DINOv3's linear segmentation probe.

The reference implementation (``facebookresearch/dinov3``) needs PyTorch >=
2.7.1, which PANDO's glibc 2.17 stack cannot install. ``timm`` carries a
pure-PyTorch port of the same architecture, plus the ViT-L/16 distilled on
SAT-493M, and runs on torch 2.6. Its weights are public on the Hub, so the
offline nodes need only a staged cache (see :mod:`rsss.models.weights`).

The head is the linear probe of the DINOv3 paper: a 1x1 convolution on the last
block's patch map, bilinearly upsampled to the input resolution. It is what
measures the backbone's frozen features, before any decoder is added.

:func:`build_dinov3_bitfit` keeps that head and unfreezes the backbone's
BitFit+ subset -- every bias, every LayerNorm affine pair and DINOv3's
LayerScale vectors -- on top of the stem the extra bands already need. It is
the lightest adaptation that still lets the features themselves move.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, cast

import timm
import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn import functional as F

from rsss.models import register_model


class _TimmViT(Protocol):
    """The slice of a timm ViT this module relies on.

    Typed structurally rather than imported: timm is untyped, and going through
    ``nn.Module.__getattr__`` would type every attribute as ``Tensor | Module``.
    """

    num_features: int
    patch_embed: Any
    grad_checkpointing: bool

    def parameters(self) -> Iterator[nn.Parameter]: ...

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]: ...

    def modules(self) -> Iterator[nn.Module]: ...

    def set_grad_checkpointing(self, enable: bool = True) -> None: ...

    def forward_intermediates(self, x: Tensor, **kwargs: Any) -> list[Tensor]: ...


#: The ViT-L/16 distilled on SAT-493M; the 7B variant does not fit a V100.
DEFAULT_MODEL = "vit_large_patch16_dinov3.sat493m"
#: SAT-493M's RGB pretraining statistics, in [0, 1] units, as published in the
#: DINOv3 README (SAT-493M is Maxar RGB ortho-imagery, so there are only three).
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)
#: Names timm gives DINOv3's LayerScale vectors. They are bare 1-D parameters
#: attached to the block, not a module, so only their name identifies them.
LAYER_SCALE_NAMES = ("gamma_1", "gamma_2")


def dinov3_normalization(scale: float = 255.0) -> tuple[list[float], list[float]]:
    """Return the SAT-493M RGB statistics in the data pipeline's units.

    ``FitDataset`` standardizes raw uint8 bands with ``norm(..., "custom")``, so
    the published [0, 1] statistics are multiplied by ``scale``. Scaling a mean
    and a standard deviation by the same factor is exactly what dividing the
    image by it first would do.

    Parameters
    ----------
    scale : float
        Value the statistics are expressed in, i.e. the top of the input dtype's
        range; 255 for the uint8 FLAIR rasters.

    Returns
    -------
    tuple[list[float], list[float]]
        Per-RGB-band means and stds.
    """
    return (
        [value * scale for value in SAT_MEAN],
        [value * scale for value in SAT_STD],
    )


def expand_patch_embed(backbone: _TimmViT, in_channels: int) -> None:
    """Widen ``backbone``'s patch-embedding convolution to ``in_channels``.

    The pretrained RGB filters and bias are copied verbatim and the extra input
    channels start at zero, so at initialization the network computes exactly
    what the RGB model computes. The extra bands only contribute once their
    filters are trained, which is why a frozen backbone needs
    ``train_stem=True`` for them to matter at all.

    Parameters
    ----------
    backbone : _TimmViT
        A timm ViT whose ``patch_embed.proj`` is a ``nn.Conv2d``.
    in_channels : int
        Number of input channels wanted, the pretrained RGB count at minimum.

    Raises
    ------
    ValueError
        If ``in_channels`` would drop pretrained filters.
    """
    old = backbone.patch_embed.proj
    if in_channels < old.in_channels:
        raise ValueError(
            f"in_channels={in_channels} would drop pretrained filters: "
            f"{backbone.patch_embed.proj.in_channels} exist."
        )
    if in_channels == old.in_channels:
        return
    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_channels].copy_(old.weight)
        if old.bias is not None:
            assert new.bias is not None, "the pretrained stem has a bias"
            new.bias.copy_(old.bias)
    backbone.patch_embed.proj = new


def bitfit_parameters(backbone: _TimmViT) -> list[nn.Parameter]:
    """Return the parameters BitFit+ trains in ``backbone``.

    They are the ones that re-scale or re-centre a feature instead of mixing
    it: the bias of every convolution and linear layer, every LayerNorm's
    affine pair, and DINOv3's LayerScale vectors. Two details of this
    architecture matter:

    * ``attn.qkv`` is built bias-free and its ``q_norm``/``k_norm`` are
      parameterless, so attention proper contributes nothing here; the
      LayerScale vectors are what reach it.
    * the patch-embedding bias belongs to the ``stem`` group, which trains the
      whole convolution anyway. A parameter must sit in exactly one group, so
      it is dropped from this selection.

    Parameters
    ----------
    backbone : _TimmViT
        A timm ViT whose blocks are LayerNorm-based, as DINOv3's are.

    Returns
    -------
    list[nn.Parameter]
        The selected parameters, each exactly once, in module order.
    """
    selected: dict[int, nn.Parameter] = {}
    for module in backbone.modules():
        if isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                for parameter in (module.weight, module.bias):
                    if isinstance(parameter, nn.Parameter):
                        selected.setdefault(id(parameter), parameter)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            bias = module.bias
            if isinstance(bias, nn.Parameter):
                selected.setdefault(id(bias), bias)
    for name, parameter in backbone.named_parameters():
        if name.rsplit(".", 1)[-1] in LAYER_SCALE_NAMES:
            selected.setdefault(id(parameter), parameter)
    stem = {id(parameter) for parameter in backbone.patch_embed.proj.parameters()}
    return [parameter for key, parameter in selected.items() if key not in stem]


def unfreeze_bitfit(backbone: _TimmViT) -> list[nn.Parameter]:
    """Make the BitFit+ subset of ``backbone`` trainable.

    Parameters
    ----------
    backbone : _TimmViT
        A timm ViT, typically already frozen by the builder.

    Returns
    -------
    list[nn.Parameter]
        The parameters that were switched on, as :func:`bitfit_parameters`
        selects them.
    """
    parameters = bitfit_parameters(backbone)
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


class DinoLinearProbe(nn.Module):
    """A DINOv3 ViT backbone with a linear segmentation head.

    Parameters
    ----------
    backbone : _TimmViT
        A timm ViT with prefix tokens and ``forward_intermediates``; its
        ``num_features`` widths the head.
    classes : int
        Number of output classes.
    """

    def __init__(self, backbone: _TimmViT, classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Conv2d(backbone.num_features, classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        """Return per-class logits at the input resolution."""
        patches = self.backbone.forward_intermediates(
            x,
            indices=1,
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )[0]
        return F.interpolate(
            self.head(patches),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Partition every parameter into the groups the recipe rates separately.

        ``stem`` is the patch-embedding convolution: the only pretrained weights
        a frozen backbone moves for the extra bands, because the NIR/nDSM
        filters appended to it start untrained. ``bitfit`` is the trainable
        BitFit+ subset, present only when the builder unfroze it. ``backbone``
        is the rest of the ViT, ``head`` the linear probe. Each parameter
        belongs to exactly one group, so an optimizer built from these can never
        silently drop one.
        """
        stem = list(self.backbone.patch_embed.proj.parameters())
        bitfit = [
            parameter
            for parameter in bitfit_parameters(self.backbone)
            if parameter.requires_grad
        ]
        owned = {id(parameter) for parameter in (*stem, *bitfit)}
        groups: dict[str, list[nn.Parameter]] = {}
        if bitfit:
            groups["bitfit"] = bitfit
        groups["stem"] = stem
        groups["backbone"] = [
            parameter
            for parameter in self.backbone.parameters()
            if id(parameter) not in owned
        ]
        groups["head"] = list(self.head.parameters())
        return groups


def _build_probe(
    model_name: str,
    pretrained: bool,
    in_channels: int,
    classes: int,
    freeze_backbone: bool,
    train_stem: bool,
) -> DinoLinearProbe:
    """Build the probe and freeze what ``freeze_backbone`` asks for."""
    backbone = cast(
        _TimmViT,
        timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=3),
    )
    expand_patch_embed(backbone, in_channels)
    if freeze_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        if train_stem:
            for parameter in backbone.patch_embed.proj.parameters():
                parameter.requires_grad_(True)
        elif in_channels > 3:
            logger.warning(
                "freeze_backbone without train_stem: the {} extra FLAIR bands "
                "meet zero-initialized, untrainable filters and are ignored.",
                in_channels - 3,
            )
    return DinoLinearProbe(backbone, classes=classes)


@register_model("dinov3_linear_probe")
def build_dinov3_linear_probe(
    model_name: str = DEFAULT_MODEL,
    pretrained: bool = True,
    in_channels: int = 5,
    classes: int = 13,
    freeze_backbone: bool = True,
    train_stem: bool = False,
    grad_checkpointing: bool = False,
) -> DinoLinearProbe:
    """Build a DINOv3 backbone with a linear segmentation head.

    Parameters
    ----------
    model_name : str
        timm model name, e.g. the SAT-493M ViT-L or a web-pretrained variant.
    pretrained : bool
        Load the pretrained weights. Tests and smoke runs pass False: it skips
        the ~1.2 GB download.
    in_channels : int
        Number of input bands. FLAIR has 5 (RGB + NIR + nDSM); the extra bands
        are appended to the RGB stem with zero-initialized filters.
    classes : int
        Number of output classes.
    freeze_backbone : bool
        Freeze every backbone parameter, leaving the head trainable.
    train_stem : bool
        Unfreeze ``patch_embed.proj`` after freezing, so the NIR/nDSM filters
        can learn. Without it, and with more than three bands, those bands meet
        zero filters no gradient can move, and are silently ignored.
    grad_checkpointing : bool
        Trade compute for activation memory by recomputing each block in the
        backward pass. Useless on the frozen probe (no backward through the
        blocks) but harmless, and needed once a block weight trains.

    Returns
    -------
    DinoLinearProbe
        The network, returning raw logits at the input resolution.

    Notes
    -----
    The stem is always trainable when ``freeze_backbone`` is False; the flag
    only matters for the frozen setup it exists for.
    """
    model = _build_probe(
        model_name, pretrained, in_channels, classes, freeze_backbone, train_stem
    )
    if grad_checkpointing:
        model.backbone.set_grad_checkpointing(True)
    return model


@register_model("dinov3_bitfit")
def build_dinov3_bitfit(
    model_name: str = DEFAULT_MODEL,
    pretrained: bool = True,
    in_channels: int = 5,
    classes: int = 13,
    freeze_backbone: bool = True,
    train_stem: bool = True,
    grad_checkpointing: bool = True,
) -> DinoLinearProbe:
    """Build a DINOv3 backbone with a linear head, adapted with BitFit+.

    Same architecture, head and stem handling as
    :func:`build_dinov3_linear_probe`; the difference is that every bias, every
    LayerNorm affine pair and every LayerScale vector also trains. That is
    ~0.1% of the backbone, but it touches all of its blocks, so it is the
    lightest recipe in which the features themselves move. Because the backward
    pass now reaches every block, ``grad_checkpointing`` defaults to True.

    Parameters
    ----------
    model_name : str
        timm model name, e.g. the SAT-493M ViT-L or a web-pretrained variant.
    pretrained : bool
        Load the pretrained weights. Tests and smoke runs pass False.
    in_channels : int
        Number of input bands, appended to the RGB stem as in the probe.
    classes : int
        Number of output classes.
    freeze_backbone : bool
        Must stay True: BitFit+ is defined as an adaptation of a frozen
        backbone, and unfreezing everything is a different recipe.
    train_stem : bool
        Unfreeze ``patch_embed.proj`` so the NIR/nDSM filters can learn. True
        by default because BitFit+ never touches the stem's weights, and those
        filters start at zero.
    grad_checkpointing : bool
        Recompute each block in the backward pass, keeping the probe's batch
        size affordable on a 32 GB V100.

    Returns
    -------
    DinoLinearProbe
        The network, returning raw logits at the input resolution.

    Raises
    ------
    ValueError
        If ``freeze_backbone`` is False.
    """
    if not freeze_backbone:
        raise ValueError(
            "dinov3_bitfit adapts a frozen backbone; freeze_backbone=False "
            "already trains every parameter, so use dinov3_linear_probe to "
            "fine-tune fully."
        )
    model = _build_probe(
        model_name, pretrained, in_channels, classes, freeze_backbone, train_stem
    )
    unfreeze_bitfit(model.backbone)
    if grad_checkpointing:
        model.backbone.set_grad_checkpointing(True)
    return model
