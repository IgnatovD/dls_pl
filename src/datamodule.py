import os
from typing import Optional

import torch
import zipfile
import requests
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule

from src.augmentations import get_transforms
from src.config import DataConfig


class ImageDM(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()

        self._config = config
        self.path_dataset_zip = os.path.join(self._config.dir_save, 'dataset.zip')

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):

        if not os.path.exists(self._config.dir_save):

            os.makedirs(self._config.dir_save, exist_ok=True)

            print(f'Download dataset to "{self.path_dataset_zip}"')
            download_dataset(self._config.dataset_url, self.path_dataset_zip)

            with zipfile.ZipFile(self.path_dataset_zip, 'r') as zip_ref:
                zip_ref.extractall(self._config.dir_save)

        if stage == 'fit':
            path_ds_train = os.path.join(self._config.dir_save, 'Classification_data', 'train')
            train_transform = get_transforms(stage, self._config.resize)

            train_dataset = datasets.ImageFolder(path_ds_train,
                                                 transform=train_transform)

            n_samples = len(train_dataset.samples)
            n_train_samples = int(n_samples*self._config.train_size)
            n_val_samples = n_samples - n_train_samples

            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(train_dataset, [n_train_samples, n_val_samples])

        elif stage == 'test':
            path_ds_test = os.path.join(self._config.dir_save, 'Classification_data', 'train')
            test_transform = get_transforms(stage, self._config.resize)
            self.test_dataset = datasets.ImageFolder(path_ds_test,
                                                transform=test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._config.n_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self._config.n_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self._config.n_workers,
        )


def download_dataset(dataset_url: str, path_dataset_zip: str):
    response = requests.get(dataset_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(path_dataset_zip, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))