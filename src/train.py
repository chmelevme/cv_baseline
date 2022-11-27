import click

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from model import SegmentationModel
import dataset_utils


@click.command()
@click.argument('max_epochs', type=click.INT)
@click.argument('accelerator', type=click.STRING)
@click.argument('num_workers', type=click.INT)
@click.argument('feature_scale', type=click.INT, default=16)
@click.argument('n_classes', type=click.INT)
@click.argument('is_deconv', type=click.BOOL, default=True)
@click.argument('in_channels', type=click.INT, default=1)
@click.argument('is_batchnorm', type=click.BOOL, default=True)
def train(max_epochs, accelerator, num_workers, feature_scale, n_classes, is_deconv, in_channels, is_batchnorm):
    train_loader, valid_loader = dataset_utils.make_train_val_dataloader()
    mlf_logger = MLFlowLogger(experiment_name="Baseline",
                              tracking_uri="./outs")

    trainer = Trainer(logger=mlf_logger, max_epochs=max_epochs, accelerator=accelerator, log_every_n_steps=1,
                      callbacks=[TQDMProgressBar(), ModelCheckpoint(dirpath="./outs/model", save_top_k=1,
                                                                    monitor="validation_loss", mode='min',
                                                                    filename="best_model")],
                      num_processes=num_workers)
    model = SegmentationModel(feature_scale=feature_scale, n_classes=n_classes, is_deconv=is_deconv,
                              in_channels=in_channels, is_batchnorm=is_batchnorm)
    trainer.fit(model, train_loader, valid_loader)
    return model


if __name__ == '__main__':
    train()
