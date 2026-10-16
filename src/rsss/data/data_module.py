from __future__ import annotations

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from rsss.data.data_loader import (
    FitDataset,
    PredictDataset,
)
from rsss.data.splits import build_flair_splits


class FlairDatamodule(LightningDataModule):
    """Lightning data module for the FLAIR aerial/label pairs.

    When ``dict_train``, ``dict_val`` or ``dict_test`` is omitted, the matching
    split is derived from the official FLAIR#1 domain split via
    :func:`rsss.data.splits.build_flair_splits`. A split that ends up empty
    raises, rather than silently reusing another split's data.
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
        use_augmentations: bool = False,
        norm_type: str = "scaling",
        means: list | None = None,
        stds: list | None = None,
    ):
        super().__init__()
        if channels is None:
            channels = [1, 2, 3, 4, 5]
        self.data_dir = data_dir
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
        self.pred_dataset = None
        self.drop_last = drop_last
        self.use_metadata = use_metadata
        self.use_augmentations = use_augmentations
        self.norm_type = norm_type
        self.means = means
        self.stds = stds
        # Official FLAIR#1 split, built lazily from data_dir and reused across
        # stages so a single discovery pass serves every omitted dict.
        self._splits: dict[str, dict[str, list[str]]] | None = None

    def prepare_data(self) -> None:
        pass

    def _flair_splits(self) -> dict[str, dict[str, list[str]]]:
        """Return the official FLAIR#1 split, discovering it once."""
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
                f"{arg_name} to FlairDatamodule, or make the matching FLAIR "
                f"domain available under {self.data_dir!r}."
            )
        return files

    def _make_fit_dataset(self, dict_files: dict, use_augmentations=None) -> FitDataset:
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
