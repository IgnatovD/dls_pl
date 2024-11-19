import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.config import Config
from src.datamodule import ImageDM
from src.lightning_module import Сlassifier


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()

def train(config: Config):
    datamodule = ImageDM(config.data_config)
    datamodule.setup()
    model = Сlassifier(config)

    experiment_save_path = os.path.join(config.dir_save_experiment, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}',
    )

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device] if config.accelerator == 'gpu' else config.device,
        log_every_n_steps=20,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config.monitor_metric, patience=4, mode=config.monitor_mode),
        ]
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(
        ckpt_path=checkpoint_callback.best_model_path,
        datamodule=datamodule,
    )


if __name__ == '__main__':
    args = arg_parse()
    pl.seed_everything(42, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config)
