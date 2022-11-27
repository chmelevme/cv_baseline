import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
from torchmetrics import Dice
import pytorch_lightning as pl


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 1), padding_size=(1, 1, 0),
                 init_stride=(1, 1, 1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True), )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True), )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4, 4, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        else:
            self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset1 = outputs2.size()[2] - inputs1.size()[2]
        offset2 = outputs2.size()[3] - inputs1.size()[3]
        offset3 = outputs2.size()[4] - inputs1.size()[4]
        padding = [offset3, 0, offset2, 0, offset1, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class unet_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.center = UnetConv3(filters[1], filters[2], self.is_batchnorm)

        # upsampling
        self.up_concat2 = UnetUp3(filters[2], filters[1], self.is_deconv, is_batchnorm)
        self.up_concat1 = UnetUp3(filters[1], filters[0], self.is_deconv, is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        center = self.center(maxpool2)

        up2 = self.up_concat2(conv2, center)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        offset = inputs.size()[2] - final.size()[2]
        padding = [0, 0, 0, 0, 0, offset]
        final = F.pad(final, padding)
        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class SegmentationModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.base_model = unet_3D(*args, **kwargs)
        self.trainDice = Dice()
        self.validDice = Dice()
        self.save_hyperparameters(
            {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)})

    def training_step(self, batch, batch_idx):
        out = self.base_model(batch[0])
        loss = nn.functional.cross_entropy(out, batch[1])
        self.log("train_loss", loss)
        self.trainDice(out, batch[1])
        self.log('train_dice', self.trainDice, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.base_model(batch[0])
        loss = nn.functional.cross_entropy(out, batch[1])
        self.validDice(out, batch[1])
        self.log('valid_dice', self.validDice, on_step=True, on_epoch=True)
        self.log("validation_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        out = self.base_model(batch[0])
        return out

    def configure_optimizers(self):
        return torch.optim.AdamW(self.base_model.parameters(), lr=0.001)
