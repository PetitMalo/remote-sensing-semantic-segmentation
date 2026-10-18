from __future__ import annotations

from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from rsss.data.config import DataConfig, load_data_config
from rsss.data.data_loader import (
    Augmentation,
    FitDataset,
    PredictDataset,
)
from rsss.data.download import DATASETS, ensure_dataset
from rsss.data.splits import build_flair_splits


class FlairDatamodule(LightningDataModule):
    """Lightning data module for the FLAIR aerial/label pairs.

    When ``dict_train``, ``dict_val`` or ``dict_test`` is omitted, the matching
    split is read from the dataset's path CSVs (``<data_dir>/CSVs``) via
    :func:`rsss.data.splits.build_flair_splits`. A split that ends up empty
    raises, rather than silently reusing another split's data.

    Point ``config`` at a ``configs/data.yaml`` (or pass a
    :class:`~rsss.data.config.DataConfig`) to take ``data_dir``, ``dataset`` and
    ``download`` from it instead of the constructor. When ``download`` is set,
    ``prepare_data()`` fetches the named dataset before ``setup()`` reads it.

    Parameters
    ----------
    config : DataConfig | str | Path, optional
        Data configuration to read. Mutually exclusive with ``data_dir``,
        ``dataset`` and ``download``.
    dataset : str, optional
        Key into :data:`rsss.data.download.DATASETS`. Required to download.
    download : bool
        Fetch ``dataset`` in ``prepare_data()`` when it is missing.
    """

    def __init__(
        self,
        data_dir: str = "data",
        dict_train: dict | None = None,
        dict_val: dict | None = None,
        dict_test: dict | None = None,
        num_workers: int = 1,
        batch_size: int = 2,
        drop_last: bool = True,
        num_classes: int = 13,
        start_one: bool = True,
        channels: list | None = None,
        use_metadata: bool = True,
        use_augmentations: Augmentation | None = None,
        norm_type: str = "scaling",
        means: list | None = None,
        stds: list | None = None,
        *,
        config: DataConfig | str | Path | None = None,
        dataset: str | None = None,
        download: bool = False,
    ):
        super().__init__()
        if config is not None:
            if data_dir != "data" or dataset is not None or download:
                raise ValueError(
                    "Pass either `config` or `data_dir`/`dataset`/`download`, not both."
                )
            resolved = (
                config if isinstance(config, DataConfig) else load_data_config(config)
            )
            data_dir = str(resolved.dataset_dir)
            dataset = resolved.dataset
            download = resolved.download
        if channels is None:
            channels = [1, 2, 3, 4, 5]
        self.data_dir = data_dir
        self.dataset = dataset
        self.download = download
        self.dict_train = dict_train
        self.dict_val = dict_val
        self.dict_test = dict_test
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.start_one = start_one
        self.channels = channels
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pred_dataset = None
        self.drop_last = drop_last
        self.use_metadata = use_metadata
        self.use_augmentations = use_augmentations
        self.norm_type = norm_type
        self.means = means
        self.stds = stds
        # FLAIR split, read lazily from ``<data_dir>/CSVs`` and reused across
        # stages so the CSVs are parsed once.
        self._splits: dict[str, dict[str, list[str]]] | None = None

    def prepare_data(self) -> None:
        """Fetch the configured dataset when it is missing, before ``setup()``."""
        if not self.download:
            return
        if self.dataset is None:
            raise ValueError(
                "download=True requires a dataset; pass `dataset` or a `config` "
                "that names one."
            )
        remote = DATASETS[self.dataset]
        root = Path(self.data_dir)
        if root.name != remote.name:
            raise ValueError(
                f"data_dir {self.data_dir!r} must end with {remote.name!r} to "
                f"download the {self.dataset!r} dataset."
            )
        ensure_dataset(remote, root.parent)

    def _flair_splits(self) -> dict[str, dict[str, list[str]]]:
        """Return the FLAIR#1 split, reading the CSVs once."""
        if self._splits is None:
            self._splits = build_flair_splits(self.data_dir)
        return self._splits

    def _resolve(self, split: str, provided: dict | None, arg_name: str) -> dict:
        """Return the file dict for a split, deriving it when not provided.

        Parameters
        ----------
        split : str
            Split name, one of "train", "validation", "test".
        provided : dict, optional
            Explicit file dict for the split, or None to derive it.
        arg_name : str
            Constructor argument name, used in the error message.

        Raises
        ------
        ValueError
            If the split has no images, instead of silently falling back to
            another split (a val set identical to train makes metrics useless).
        """
        files = provided if provided is not None else self._flair_splits()[split]
        if not files.get("IMG"):
            raise ValueError(
                f"The '{split}' split has no images. Pass an explicit "
                f"{arg_name} to FlairDatamodule, or point data_dir "
                f"({self.data_dir!r}) at an extracted FLAIR#1 dataset."
            )
        return files

    def _make_fit_dataset(
        self, dict_files: dict, use_augmentations: Augmentation | None = None
    ) -> FitDataset:
        return FitDataset(
            dict_files=dict_files,
            channels=self.channels,
            num_classes=self.num_classes,
            start_one=self.start_one,
            use_metadata=self.use_metadata,
            use_augmentations=use_augmentations,
            norm_type=self.norm_type,
            means=self.means,
            stds=self.stds,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._make_fit_dataset(
                self._resolve("train", self.dict_train, "dict_train"),
                use_augmentations=self.use_augmentations,
            )
            self.val_dataset = self._make_fit_dataset(
                self._resolve("validation", self.dict_val, "dict_val"),
            )
        elif stage == "validate":
            self.val_dataset = self._make_fit_dataset(
                self._resolve("validation", self.dict_val, "dict_val"),
            )
        elif stage == "test":
            self.test_dataset = self._make_fit_dataset(
                self._resolve("test", self.dict_test, "dict_test"),
            )
        elif stage == "predict":
            self.pred_dataset = PredictDataset(
                dict_files=self._resolve("test", self.dict_test, "dict_test"),
                channels=self.channels,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata,
                norm_type=self.norm_type,
                means=self.means,
                stds=self.stds,
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader().")
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader().")
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader().")
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def predict_dataloader(self):
        if self.pred_dataset is None:
            raise RuntimeError("Call setup('predict') before predict_dataloader().")
        return DataLoader(
            dataset=self.pred_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
