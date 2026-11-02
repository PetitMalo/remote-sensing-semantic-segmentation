"""FLAIR#1 band order, land-cover nomenclature and plotting colours.

The dataset ships 13 land-cover classes encoded as MSK values 1..13, and a
5-band raster ordered Red, Green, Blue, NIR, nDSM. Keeping the names and the
palette here gives the inference figures and the exploration notebook one
source of truth, and keeps the 0-based class indices used by the metrics
(``class_0`` is MSK value 1) aligned with a human-readable label.
"""

from __future__ import annotations

# Band order of the FLAIR#1 aerial rasters (official dataset card).
BAND_NAMES = ("Red", "Green", "Blue", "NIR", "nDSM")

# FLAIR#1 13-class nomenclature (official IGN dataset card), MSK values 1..13.
FLAIR_CLASSES: dict[int, str] = {
    1: "Building",
    2: "Pervious surface",
    3: "Impervious surface",
    4: "Bare soil",
    5: "Water",
    6: "Coniferous",
    7: "Deciduous",
    8: "Brushwood",
    9: "Vineyard",
    10: "Herbaceous vegetation",
    11: "Agricultural land",
    12: "Plowed land",
    13: "Swimming pool",
}

# One colour per class, indexed like the masks: index 0 is MSK value 1.
CLASS_COLORS = (
    "#8c6d31",
    "#b3b3b3",
    "#4d4d4d",
    "#a0783c",
    "#0000d3",
    "#008b45",
    "#006400",
    "#9be303",
    "#c1292a",
    "#00ffe1",
    "#fafe78",
    "#86c9ca",
    "#ff01ff",
)


def class_label(index: int) -> str:
    """Return the legend label of a 0-based class index, e.g. ``0 Building``."""
    return f"{index} {FLAIR_CLASSES[index + 1]}"
