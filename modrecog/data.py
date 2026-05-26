from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


class RadioMLDataset(Dataset):
    def __init__(self, hdf5_path: Path) -> None:
        super().__init__()
        self.hdf5_path = hdf5_path

        self._file: h5py.File | None = None
        with h5py.File(hdf5_path, "r") as f:
            self.length = f["X"].shape[0]
            self.labels: np.ndarray = np.argmax(f["Y"][:], axis=1).astype(np.int64)
            self.snr: np.ndarray = f["Z"][:].squeeze().astype(np.float32)

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._open()["X"][idx]).T.contiguous()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class RadioMLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        raw_path: str,
        batch_size: int = 512,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        snr_min: int | None = None,
        snr_max: int | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.raw_path = Path(raw_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.max_samples = max_samples
        self.seed = seed

        self._train: Subset | None = None
        self._val: Subset | None = None
        self._test: Subset | None = None

    def setup(self, stage: str | None = None) -> None:
        dataset = RadioMLDataset(self.raw_path)
        indices = np.arange(len(dataset))

        snr = dataset.snr[indices]
        mask = np.ones(len(indices), dtype=bool)
        if self.snr_min is not None:
            mask &= snr >= self.snr_min
        if self.snr_max is not None:
            mask &= snr <= self.snr_max
        indices = indices[mask]

        if self.max_samples is not None and len(indices) > self.max_samples:
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(indices, self.max_samples, replace=False)

        n_kept = len(indices)
        print(f"Dataset: {n_kept:,} samples after SNR/max_samples filter.")

        strat_labels = dataset.labels[indices]

        train_idx, tmp_idx = train_test_split(
            indices,
            test_size=self.val_split + self.test_split,
            stratify=strat_labels,
            random_state=self.seed,
        )
        relative_test = self.test_split / (self.val_split + self.test_split)
        val_idx, test_idx = train_test_split(
            tmp_idx,
            test_size=relative_test,
            stratify=dataset.labels[tmp_idx],
            random_state=self.seed,
        )

        self._train = Subset(dataset, train_idx.tolist())
        self._val = Subset(dataset, val_idx.tolist())
        self._test = Subset(dataset, test_idx.tolist())

    def _dataloader(self, subset: Subset, shuffle: bool) -> DataLoader:
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self._test, shuffle=False)
