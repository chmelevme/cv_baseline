from model import SegmentationModel
from pytorch_lightning import Trainer
from dataset_utils import CustomDataset
import torch.utils.data as data
from torchmetrics import Dice
import click


@click.command()
@click.argument('split', type=click.STRING, default='train')
@click.argument('accelerator', type=click.STRING)
@click.argument('feature_scale', type=click.INT, default=16)
@click.argument('n_classes', type=click.INT)
@click.argument('is_deconv', type=click.BOOL, default=True)
@click.argument('in_channels', type=click.INT, default=1)
@click.argument('is_batchnorm', type=click.BOOL, default=True)
def calc_dice_metric(split, accelerator, feature_scale, n_classes, is_deconv, in_channels, is_batchnorm):
    model = SegmentationModel.load_from_checkpoint("outs/model/best_model.ckpt", feature_scale=feature_scale,
                                                   n_classes=n_classes,
                                                   in_channels=in_channels, is_deconv=is_deconv,
                                                   is_batchnorm=is_batchnorm).base_model
    model.to(accelerator)
    df = CustomDataset(split)
    dl = data.DataLoader(df, 1)
    dice = Dice().to(accelerator)
    for batch in dl:
        batch[0] = batch[0].to(accelerator)
        batch[1] = batch[1].to(accelerator)
        preds = model(batch[0])
        dice(preds, batch[1])
        del batch, preds
    print(f"Dice similarity coefficient on all data: {dice.compute()}")


if __name__ == "__main__":
    calc_dice_metric()
