from typing import List, Optional
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb

from .network_model import NetworkModel
from ..utils.pruning import compute_eigs_to_keep, bema_scheduler

class NetworkModule(pl.LightningModule):
    def __init__(
        self,
        dims: List[int],
        use_relu: List[bool],
        learning_rate: float = 0.02,
        momentum: float = 0.9,
        alpha: float = 0.25,
        beta: float = 0.9,
        goodness_of_fit_cutoff: List[float] = [1.0],
        l1_lambda: float = 0.000005,
        l2_lambda: float = 0.000005,
        split_frequency: int = 1,
        pruning_enabled: bool = True,
        show_eignspectra: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = NetworkModel(
            without_rel=use_relu,
            dims=dims,
            alpha=alpha,
            beta=beta,
            goodness_of_fit_cutoff=goodness_of_fit_cutoff
        )

        self.initial_param_count = self.model.get_parameter_count()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum
        )
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer,
            lr_lambda=lambda epoch: 0.96
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.nll_loss(output, y)

        # Add L1 and L2 regularization
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        l2_norm = sum((p ** 2).sum() for p in self.parameters())
        loss += self.hparams.l1_lambda * l1_norm + self.hparams.l2_lambda * l2_norm

        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).float().mean()

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.nll_loss(output, y)

        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).float().mean()

        self.log('val_loss', loss)
        self.log('val_acc', acc)

        # print(f'Validation loss: {loss}, Validation accuracy: {acc}')

        return loss

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.hparams.split_frequency == 0 and self.hparams.pruning_enabled:
            self.apply_pruning()

    def apply_pruning(self):
        for i, layer in enumerate(self.model.fc):
            lp_transformed, eigs_to_keep, good_fit = compute_eigs_to_keep(
                self.model,
                self.model.get_layer_matrix(i),
                self.model.dims,
                self.current_epoch,
                self.hparams.goodness_of_fit_cutoff,
                self.hparams.show_eignspectra
            )

            result = layer.split(
                bema_scheduler(self.current_epoch),
                save_name=f'layer_{i}_epoch_{self.current_epoch}',
            )

            # Log pruning metrics
            self.log(f'layer_{i}_eigs_kept', eigs_to_keep)
            # self.log(f'layer_{i}_good_fit', good_fit)

        current_params = self.model.get_parameter_count()
        param_reduction = 1 - (current_params / self.initial_param_count)
        self.log('parameter_reduction', param_reduction)