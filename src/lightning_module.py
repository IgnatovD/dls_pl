import torch
import pytorch_lightning as pl

from src.model import CNN
from src.config import Config
from src.train_utils import load_object


class Сlassifier(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config

        self._model = CNN()
        self.loss_fn = load_object(self._config.loss_fn)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):
        images, gt_labels = batch
        pr_logits = self(images)
        loss = self.loss_fn(pr_logits, gt_labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):

        images, gt_labels = batch
        pr_logits = self(images)
        loss = self.loss_fn(pr_logits, gt_labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, gt_labels = batch
        pr_logits = self(images)
        loss = self.loss_fn(pr_logits, gt_labels)
        return loss


