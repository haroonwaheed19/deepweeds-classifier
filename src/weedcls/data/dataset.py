from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from weedcls.logging_utils import get_logger

log = get_logger("dataset")

CLASS_NAMES = [
    "Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia",
    "Rubber vine", "Siam weed", "Snake weed", "Negative",
]


class WeedDataset(Dataset):
    def __init__(self, csv_path: str | Path, img_dir: str | Path, transform=None):
        df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        exists = df["Filename"].map(lambda f: (self.img_dir / f).is_file())
        if not exists.all():
            log.warning("%s: %d listed images are missing on disk, skipping them",
                        csv_path, int((~exists).sum()))
        self.df = df[exists].reset_index(drop=True)
        if self.df.empty:
            raise FileNotFoundError(f"no images from {csv_path} found under {self.img_dir}")
        self.labels = self.df["Label"].to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        fn = self.df.loc[i, "Filename"]
        try:
            with Image.open(self.img_dir / fn) as im:
                img = im.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"could not decode {fn}: {e}") from e
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])


def class_weights(labels: np.ndarray, num_classes: int = 9, power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency^power weights, scaled so the average weight over samples is 1.

    power=0.5 (inverse sqrt) is a compromise: full inverse frequency gives Negative a weight
    ~9x smaller than a weed and tends to over-predict weeds.
    """
    freq = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = 1.0 / np.maximum(freq, 1) ** power
    w *= len(labels) / (w * freq).sum()
    return torch.tensor(w, dtype=torch.float32)


def make_sampler(labels: np.ndarray, num_classes: int = 9, power: float = 0.5) -> WeightedRandomSampler:
    w = class_weights(labels, num_classes, power).numpy()
    return WeightedRandomSampler(w[labels], num_samples=len(labels), replacement=True)


def make_loader(ds: Dataset, batch_size: int, train: bool, num_workers: int = 2,
                sampler=None) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train and sampler is None, sampler=sampler,
        num_workers=num_workers, persistent_workers=num_workers > 0, drop_last=train,
    )
