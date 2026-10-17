"""Unit tests for the shared metrics.

``ClassSupport`` is exercised through the ``StageMetric`` protocol, the same
interface the training module uses.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from rsss.training.metrics import ClassSupport, StageMetric


def make_support(num_classes: int) -> StageMetric:
    """Return a support metric behind the interface the module relies on."""
    return ClassSupport(num_classes=num_classes)


def test_support_counts_the_labelled_pixels_of_each_class():
    support = make_support(num_classes=3)
    target = torch.tensor([[[0, 0], [1, 2]]])

    support.update(torch.zeros_like(target), target)

    assert support.compute().tolist() == [2, 1, 1]


def test_support_is_zero_for_a_class_absent_from_the_labels():
    support = make_support(num_classes=3)

    support.update(None, torch.full((1, 2, 2), 1))

    assert support.compute().tolist() == [0, 4, 0]


def test_support_accumulates_across_batches():
    support = make_support(num_classes=2)
    values: Tensor = torch.tensor([[[0, 1]]])

    support.update(None, values)
    support.update(None, values)

    assert support.compute().tolist() == [2, 2]


@pytest.mark.filterwarnings("ignore:.*compute.*called before the.*update")
def test_support_resets_between_epochs():
    support = make_support(num_classes=2)
    support.update(None, torch.zeros((1, 2, 2), dtype=torch.long))

    support.reset()

    assert support.compute().tolist() == [0, 0]
