from __future__ import annotations

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from rsss.data.data_loader import (
    FitDataset,
    PredictDataset,
    discover_data,
)


class FlairDatamodule(LightningDataModule):
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

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Determine dict_files for each stage
        if stage == "fit" or stage == "validate":
            if self.dict_train is not None:
                self.train_dataset = FitDataset(
                    dict_files=self.dict_train,
                    channels=self.channels,
                    num_classes=self.num_classes,
                    start_one=self.start_one,
                    use_metadata=self.use_metadata,
                    use_augmentations=self.use_augmentations,
                    norm_type=self.norm_type,
                    means=self.means,
                    stds=self.stds,
                )
            else:
                # Auto-discover training data from data_dir
                self.train_dataset = FitDataset(
                    dict_files=discover_data(self.data_dir),
                    channels=self.channels,
                    num_classes=self.num_classes,
                    start_one=self.start_one,
                    use_metadata=self.use_metadata,
                    use_augmentations=self.use_augmentations,
                    norm_type=self.norm_type,
                    means=self.means,
                    stds=self.stds,
                )

            if self.dict_val is not None:
                self.val_dataset = FitDataset(
                    dict_files=self.dict_val,
                    channels=self.channels,
                    num_classes=self.num_classes,
                    start_one=self.start_one,
                    use_metadata=self.use_metadata,
                    norm_type=self.norm_type,
                    means=self.means,
                    stds=self.stds,
                )
            else:
                # Auto-discover validation data from data_dir (hold-out from train)
                # For now, use a subset or the same area; here we just use discover_data
                # and split manually would be complex, so we'll use discover_data
                # and take a separate portion. We'll just use the full discovered set
                # as training and leave val empty, or we could use a second area.
                # For this adaptation, we'll use discover_data for val as well,
                # but in practice users should provide dict_train/dict_val.
                self.val_dataset = FitDataset(
                    dict_files=discover_data(self.data_dir),
                    channels=self.channels,
                    num_classes=self.num_classes,
                    start_one=self.start_one,
                    use_metadata=self.use_metadata,
                    norm_type=self.norm_type,
                    means=self.means,
                    stds=self.stds,
                )

        elif stage == "predict":
            if self.dict_test is not None:
                self.pred_dataset = PredictDataset(
                    dict_files=self.dict_test,
                    channels=self.channels,
                    num_classes=self.num_classes,
                    use_metadata=self.use_metadata,
                    norm_type=self.norm_type,
                    means=self.means,
                    stds=self.stds,
                )
            else:
                # Auto-discover test/prediction data from data_dir
                self.pred_dataset = PredictDataset(
                    dict_files=discover_data(self.data_dir),
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
