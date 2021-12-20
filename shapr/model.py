import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from data_generator import SHAPRDataset
from torch.utils.data import DataLoader, random_split
#from .metrics import *

class EncoderBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.encoderblock = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(1, 3, 3), padding='same', bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(1, 3, 3), padding='same', bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.encoderblock(x)

class DecoderBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.decoderblock = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(1, 3, 3), padding='same', bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(1, 3, 3), padding='same', bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.decoderblock(x)

class Down122(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)),
        )
    def forward(self, x):
        return self.maxpool(x)

class Down222(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool3d((2, 2, 2)),
        )
    def forward(self, x):
        return self.maxpool(x)

class Up211(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(2,1,1), stride=(2, 1, 1))
    def forward(self, x):
        return self.up(x)

class Up222(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2,2,2), stride=(2, 2, 2))
    def forward(self, x):
        return self.up(x)

class EncoderOut(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc_out = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,  kernel_size=(1, 3, 3), padding='same', bias=False),
            nn.Sigmoid())
    def forward(self, x):
        return self.enc_out(x)

class DecoderOut(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dec_out = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding='same', bias=False),
            nn.Sigmoid())
    def forward(self, x):
        return self.dec_out(x)

class DiscriminatorOut(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.disc_out = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding='same', bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, 1, kernel_size=(3, 3, 3), padding='same', bias=False),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        return self.disc_out(x)

class SHAPR(nn.Module):
    def __init__(self):
        super(SHAPR, self).__init__()
        n_filters = 10
        self.conv1 = EncoderBlock(2, n_filters)
        self.down1 = Down122()
        self.conv2 = EncoderBlock(n_filters, n_filters*2)
        self.down2 = Down122()
        self.conv3 = EncoderBlock(n_filters*2, n_filters*4)
        self.encout = EncoderOut(n_filters*4, n_filters*8)

        self.conv4 = EncoderBlock(n_filters*8, n_filters*8)
        self.up4 = Up211(n_filters*8, n_filters*8)
        self.conv5 = EncoderBlock(n_filters*8, n_filters*8)
        self.up5 = Up211(n_filters*8, n_filters*8)
        self.conv6 = EncoderBlock(n_filters*8, n_filters*4)
        self.up6 = Up222(n_filters*4, n_filters*4)
        self.conv7 = EncoderBlock(n_filters*4, n_filters*4)
        self.up7 = Up211(n_filters*4, n_filters*4)
        self.conv8 = EncoderBlock(n_filters*4, n_filters*2)
        self.up8 = Up211(n_filters*2, n_filters*2)
        self.conv9 = EncoderBlock(n_filters*2, n_filters)
        self.up9 = Up222(n_filters, n_filters)
        self.conv10 = EncoderBlock(n_filters, n_filters)
        self.decout = DecoderOut(n_filters, 1)


    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.down1(x)
        x = self.conv2(x)
        x = self.down2(x)
        x = self.conv3(x)
        x_enc = self.encout(x)
        x = self.conv4(x_enc)
        x = self.up4(x)
        x = self.conv5(x)
        x = self.up5(x)
        x = self.conv6(x)
        x = self.up6(x)
        x = self.conv7(x)
        x = self.up7(x)
        x = self.conv8(x)
        x = self.up8(x)
        x = self.conv9(x)
        x = self.up9(x)
        x_dec = self.decout(x)
        return x_dec

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        n_filters = 10
        self.conv1 = EncoderBlock(1, n_filters)
        self.down1 = Down222()
        self.conv2 = EncoderBlock(n_filters, n_filters*2)
        self.down2 = Down222()
        self.conv3 = EncoderBlock(n_filters*2, n_filters*4)
        self.down3 = Down222()
        self.conv4 = EncoderBlock(n_filters*4, n_filters*8)
        self.down4 = Down222()
        self.conv5 = EncoderBlock(n_filters*8, n_filters*16)
        self.discout = DiscriminatorOut(n_filters, 1)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.down1(x)
        x = self.conv2(x)
        x = self.down2(x)
        x = self.conv3(x)
        x = self.down3(x)
        x = self.conv4(x)
        x = self.down4(x)
        x = self.conv5(x)
        x_dis = self.discout(x)
        return x_dis

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

netSHAPR = SHAPR().to(device)
netSHAPR.apply(weights_init)
print(netSHAPR)

netDiscriminator = Discriminator().to(device)
netDiscriminator.apply(weights_init)
print(netDiscriminator)


class LightningSHAPRoptimiziation(pl.LightningModule):
    def __init__(self):
        super.__init__()
        n_filters = 10
        self.conv1 = EncoderBlock(2, n_filters)
        self.down1 = Down122()
        self.conv2 = EncoderBlock(n_filters, n_filters*2)
        self.down2 = Down122()
        self.conv3 = EncoderBlock(n_filters*2, n_filters*4)
        self.encout = EncoderOut(n_filters*4, n_filters*8)
        self.conv4 = EncoderBlock(n_filters*8, n_filters*8)
        self.up4 = Up211(n_filters*8, n_filters*8)
        self.conv5 = EncoderBlock(n_filters*8, n_filters*8)
        self.up5 = Up211(n_filters*8, n_filters*8)
        self.conv6 = EncoderBlock(n_filters*8, n_filters*4)
        self.up6 = Up222(n_filters*4, n_filters*4)
        self.conv7 = EncoderBlock(n_filters*4, n_filters*4)
        self.up7 = Up211(n_filters*4, n_filters*4)
        self.conv8 = EncoderBlock(n_filters*4, n_filters*2)
        self.up8 = Up211(n_filters*2, n_filters*2)
        self.conv9 = EncoderBlock(n_filters*2, n_filters)
        self.up9 = Up222(n_filters, n_filters)
        self.conv10 = EncoderBlock(n_filters, n_filters)
        self.decout = DecoderOut(n_filters, 1)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.down1(x)
        x = self.conv2(x)
        x = self.down2(x)
        x = self.conv3(x)
        x_enc = self.encout(x)
        x = self.conv4(x_enc)
        x = self.up4(x)
        x = self.conv5(x)
        x = self.up5(x)
        x = self.conv6(x)
        x = self.up6(x)
        x = self.conv7(x)
        x = self.up7(x)
        x = self.conv8(x)
        x = self.up8(x)
        x = self.conv9(x)
        x = self.up9(x)
        x_dec = self.decout(x)
        return x_dec

    def configure_optimizers(self):
        params = [SHAPR.parameters(), Discriminator.parameters()]
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return optimizer

    def MSEloss(self, y_true, y_pred):
        MSE = torch.nn.MSELoss()
        return MSE(y_true, y_pred)

    def training_step(self, images, true_obj):
        pred = self.forward(images)
        loss = self.MSEloss(true_obj, pred)
        self.log("train loss", loss)
        return loss

    def validation_step(self, images, true_obj):
        pred = self.forward(images)
        loss = self.MSEloss(true_obj, pred)
        self.log("validation loss", loss)

