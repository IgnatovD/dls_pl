from typing import List

from omegaconf import OmegaConf
from pydantic import BaseModel


class DataConfig(BaseModel):
    dataset_url: str
    dir_save: str
    resize: int
    batch_size: int
    n_workers: int
    train_size: float


class Config(BaseModel):
    project_name: str
    experiment_name: str
    dir_save_experiment: str
    data_config: DataConfig
    n_epochs: int
    accelerator: str
    device: int
    monitor_metric: str
    monitor_mode: str
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    loss_fn: str

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
