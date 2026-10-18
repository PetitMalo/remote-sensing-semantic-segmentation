"""Unit tests for ``rsss.data.data_module``.

Most tests stub ``build_flair_splits`` so ``FlairDatamodule`` is exercised
without any file reads; one end-to-end test runs against a synthetic dataset tree
written to ``tmp_path`` via the shared ``write_flair_dataset`` fixture.
"""

from __future__ import annotations

import pytest
from torch.utils.data import RandomSampler, SequentialSampler

from rsss.data.data_loader import FitDataset, PredictDataset
from rsss.data.data_module import FlairDatamodule

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_splits() -> dict[str, dict[str, list[str]]]:
    """A three-split file dict shaped like ``build_flair_splits`` output."""
    return {
        "train": {"IMG": ["train_img.tif"], "MSK": ["train_msk.tif"]},
        "validation": {"IMG": ["val_img.tif"], "MSK": ["val_msk.tif"]},
        "test": {"IMG": ["test_img.tif"], "MSK": ["test_msk.tif"]},
    }


@pytest.fixture
def discovery_calls(monkeypatch, fake_splits) -> list[str]:
    """Patch split discovery with a stub and record the ``data_dir`` arguments."""
    calls: list[str] = []

    def _build(data_dir: str) -> dict[str, dict[str, list[str]]]:
        calls.append(data_dir)
        return fake_splits

    monkeypatch.setattr("rsss.data.data_module.build_flair_splits", _build)
    return calls


@pytest.fixture
def flair_root(tmp_path, write_flair_dataset) -> str:
    """A synthetic FLAIR#1 tree with one tile per split."""
    return str(write_flair_dataset(tmp_path / "flair_1_toy_dataset"))


# --------------------------------------------------------------------------- #
# Construction                                                                #
# --------------------------------------------------------------------------- #


def test_constructor_sets_documented_defaults():
    datamodule = FlairDatamodule()

    assert datamodule.data_dir == "data"
    assert datamodule.channels == [1, 2, 3, 4, 5]
    assert datamodule.batch_size == 2
    assert datamodule.num_classes == 13
    assert datamodule.start_one is True
    assert datamodule.use_metadata is True
    assert datamodule.drop_last is True
    assert datamodule.norm_type == "scaling"


def test_constructor_starts_without_datasets():
    datamodule = FlairDatamodule()

    assert datamodule.train_dataset is None
    assert datamodule.val_dataset is None
    assert datamodule.pred_dataset is None


def test_make_fit_dataset_forwards_configuration():
    def augment(image, mask):
        return {"image": image, "mask": mask}

    datamodule = FlairDatamodule(
        channels=[2],
        num_classes=4,
        start_one=False,
        use_metadata=False,
        norm_type="custom",
        means=[1.0],
        stds=[2.0],
    )
    dict_files = {"IMG": ["img.tif"], "MSK": ["msk.tif"]}

    dataset = datamodule._make_fit_dataset(dict_files, use_augmentations=augment)

    assert isinstance(dataset, FitDataset)
    assert dataset.channels == [2]
    assert dataset.num_classes == 4
    assert dataset.start_one is False
    assert dataset.use_metadata is False
    assert dataset.use_augmentations is augment
    assert dataset.norm_type == "custom"
    assert dataset.means == [1.0]
    assert dataset.stds == [2.0]


# --------------------------------------------------------------------------- #
# Split resolution                                                            #
# --------------------------------------------------------------------------- #


def test_resolve_prefers_provided_dict(discovery_calls):
    datamodule = FlairDatamodule()
    provided = {"IMG": ["img.tif"], "MSK": ["msk.tif"]}

    resolved = datamodule._resolve("train", provided, "dict_train")

    assert resolved is provided
    assert discovery_calls == []


def test_resolve_derives_split_when_not_provided(fake_splits):
    datamodule = FlairDatamodule()
    datamodule._splits = fake_splits

    assert (
        datamodule._resolve("validation", None, "dict_val") is fake_splits["validation"]
    )


def test_resolve_raises_when_split_has_no_images():
    datamodule = FlairDatamodule(data_dir="my_data")
    datamodule._splits = {"test": {"IMG": [], "MSK": []}}

    with pytest.raises(ValueError) as excinfo:
        datamodule._resolve("test", None, "dict_test")

    message = str(excinfo.value)
    assert "'test' split has no images" in message
    assert "dict_test" in message
    assert "my_data" in message


def test_resolve_raises_even_when_split_is_provided_but_empty():
    datamodule = FlairDatamodule()

    with pytest.raises(ValueError, match="dict_train"):
        datamodule._resolve("train", {"IMG": [], "MSK": []}, "dict_train")


def test_flair_splits_discovers_once_and_caches(discovery_calls, fake_splits):
    datamodule = FlairDatamodule(data_dir="my_data")

    first = datamodule._flair_splits()
    second = datamodule._flair_splits()

    assert first == fake_splits
    assert second is first
    assert discovery_calls == ["my_data"]


# --------------------------------------------------------------------------- #
# setup                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stage", [None, "fit"])
def test_setup_fit_builds_train_and_val_datasets(discovery_calls, stage):
    datamodule = FlairDatamodule()

    datamodule.setup(stage)

    assert isinstance(datamodule.train_dataset, FitDataset)
    assert isinstance(datamodule.val_dataset, FitDataset)
    assert datamodule.train_dataset.list_imgs.tolist() == ["train_img.tif"]
    assert datamodule.val_dataset.list_imgs.tolist() == ["val_img.tif"]
    assert discovery_calls == ["data"]


def test_setup_fit_reuses_one_discovery_for_train_and_val(discovery_calls, fake_splits):
    datamodule = FlairDatamodule(dict_train=fake_splits["train"], dict_val=None)

    datamodule.setup("fit")

    # Only the validation split had to be derived.
    assert discovery_calls == ["data"]


def test_setup_fit_enables_augmentations_on_train_only(discovery_calls):
    def augment(image, mask):
        return {"image": image, "mask": mask}

    datamodule = FlairDatamodule(use_augmentations=augment)

    datamodule.setup("fit")

    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    assert isinstance(train_dataset, FitDataset)
    assert isinstance(val_dataset, FitDataset)
    assert train_dataset.use_augmentations is augment
    assert val_dataset.use_augmentations is None


def test_setup_validate_builds_only_val_dataset(discovery_calls):
    datamodule = FlairDatamodule()

    datamodule.setup("validate")

    assert datamodule.train_dataset is None
    assert isinstance(datamodule.val_dataset, FitDataset)
    assert datamodule.pred_dataset is None


def test_setup_predict_builds_predict_dataset(discovery_calls):
    datamodule = FlairDatamodule(channels=[1])

    datamodule.setup("predict")

    assert isinstance(datamodule.pred_dataset, PredictDataset)
    assert datamodule.pred_dataset.channels == [1]
    assert datamodule.pred_dataset.list_imgs.tolist() == ["test_img.tif"]
    assert datamodule.train_dataset is None


def test_setup_predict_prefers_explicit_test_dict(discovery_calls):
    datamodule = FlairDatamodule(dict_test={"IMG": ["explicit.tif"], "MSK": ["m.tif"]})

    datamodule.setup("predict")

    pred_dataset = datamodule.pred_dataset
    assert isinstance(pred_dataset, PredictDataset)
    assert pred_dataset.list_imgs.tolist() == ["explicit.tif"]
    assert discovery_calls == []


# --------------------------------------------------------------------------- #
# Dataloaders                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("train_dataloader", "setup\\('fit'\\)"),
        ("val_dataloader", "setup\\('fit'\\)"),
        ("predict_dataloader", "setup\\('predict'\\)"),
    ],
)
def test_dataloaders_raise_before_setup(method, message):
    datamodule = FlairDatamodule()

    with pytest.raises(RuntimeError, match=message):
        getattr(datamodule, method)()


def test_train_dataloader_shuffles_and_honours_batch_settings(discovery_calls):
    datamodule = FlairDatamodule(batch_size=3, drop_last=False)
    datamodule.setup("fit")

    loader = datamodule.train_dataloader()

    assert loader.batch_size == 3
    assert loader.drop_last is False
    assert isinstance(loader.sampler, RandomSampler)


def test_val_dataloader_preserves_order(discovery_calls):
    datamodule = FlairDatamodule()
    datamodule.setup("fit")

    loader = datamodule.val_dataloader()

    assert isinstance(loader.sampler, SequentialSampler)


def test_predict_dataloader_uses_single_item_batches(discovery_calls):
    datamodule = FlairDatamodule(batch_size=8, drop_last=True)
    datamodule.setup("predict")

    loader = datamodule.predict_dataloader()

    assert loader.batch_size == 1
    assert loader.drop_last is False
    assert isinstance(loader.sampler, SequentialSampler)


# --------------------------------------------------------------------------- #
# End to end (real synthetic files)                                           #
# --------------------------------------------------------------------------- #


def test_setup_fit_and_train_dataloader_on_synthetic_tree(flair_root):
    datamodule = FlairDatamodule(data_dir=flair_root, batch_size=1, drop_last=False)

    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    assert batch["img"].shape == (1, 5, 4, 4)
    assert batch["msk"].shape == (1, 13, 4, 4)
