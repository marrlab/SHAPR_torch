import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from data_generator import SHAPRDataset
from torch.utils.data import DataLoader, random_split
from metrics import dice_loss as dice_loss
import torchvision
from collections import OrderedDict
import os

from torch_topological.nn import CubicalComplex
from torch_topological.nn import WassersteinDistance

from torch_topological.nn.data import batch_iter


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
            nn.Conv3d(in_channels, mid_channels, kernel_size=(3, 3, 3), padding='same', bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(3, 3, 3), padding='same', bias=False),
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
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.disc_out(x)


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
        self.discout = DiscriminatorOut(n_filters*16, 1)

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


class SHAPR(nn.Module):
    def __init__(self):
        super().__init__()
        n_filters = 10
        self.conv1 = EncoderBlock(2, n_filters)
        self.down1 = Down122()
        self.conv2 = EncoderBlock(n_filters, n_filters * 2)
        self.down2 = Down122()
        self.conv3 = EncoderBlock(n_filters * 2, n_filters * 4)
        self.encout = EncoderOut(n_filters * 4, n_filters * 8)
        self.conv4 = EncoderBlock(n_filters * 8, n_filters * 8)
        self.up4 = Up211(n_filters * 8, n_filters * 8)
        self.conv5 = EncoderBlock(n_filters * 8, n_filters * 8)
        self.up5 = Up211(n_filters * 8, n_filters * 8)
        self.conv6 = EncoderBlock(n_filters * 8, n_filters * 4)
        self.up6 = Up222(n_filters * 4, n_filters * 4)
        self.conv7 = EncoderBlock(n_filters * 4, n_filters * 4)
        self.up7 = Up211(n_filters * 4, n_filters * 4)
        self.conv8 = EncoderBlock(n_filters * 4, n_filters * 2)
        self.up8 = Up211(n_filters * 2, n_filters * 2)
        self.conv9 = EncoderBlock(n_filters * 2, n_filters)
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


class LightningSHAPRoptimization(pl.LightningModule):
    def __init__(self, settings, cv_train_filenames, cv_val_filenames):
        super(LightningSHAPRoptimization, self).__init__()

        self.random_seed = settings.random_seed
        self.path = settings.path
        self.cv_train_filenames = cv_train_filenames
        self.cv_val_filenames = cv_val_filenames
        self.batch_size = settings.batch_size
        # Define model
        self.shapr = SHAPR()

        # Define learning rate
        self.lr = 0.01

        # Defining loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = dice_loss

        # Required for topological feature calculation. We want cubical
        # complexes because they handle images intrinsically.
        #
        #  TODO: Consider superlevel set filtrations?
        #  TODO: Consider different weighting schemes for Wasserstein?
        self.cubical_complex = CubicalComplex(dim=3)
        self.topo_loss = WassersteinDistance(q=2)
        self.topo_lambda = settings.topo_lambda
        self.topo_interp = settings.topo_interp
        self.topo_feat_d = settings.topo_feat_d

    def forward(self, x):
        return self.shapr(x)

    def configure_optimizers(self):
        lr = 0.01
        b1 = 0.5
        b2 = 0.999
        return torch.optim.Adam(self.shapr.parameters(), lr=lr)

    def MSEloss(self, y_true, y_pred):
        MSE = torch.nn.MSELoss()
        return MSE(y_true, y_pred)

    def binary_crossentropy_Dice(self, y_pred, y_true):
        #return self.dice_loss(y_pred, y_true)
        return (self.dice_loss(y_pred, y_true) + F.binary_cross_entropy(y_pred, y_true)) / 2
        #return (self.MSEloss(y_pred, y_true) + F.binary_cross_entropy(y_pred, y_true)) / 2

    def topological_step(self, pred_obj, true_obj):
        """Calculate topological features and adjust loss."""
        # Check whether there's anything to do here. This makes it
        # possible to disable the calculation of topological features
        # altogether.
        if self.topo_lambda == 0.0:
            return 0.0

        if self.topo_interp != 0:
            size = (self.topo_interp, ) * 3
            pred_obj_ = nn.functional.interpolate(input=pred_obj, size=size)
            true_obj_ = nn.functional.interpolate(input=true_obj, size=size)

        # No interpolation desired by client; use the original data set,
        # thus making everything slower.
        else:
            pred_obj_ = pred_obj
            true_obj_ = true_obj

        # Calculate topological features of predicted 3D tensor and true
        # 3D tensor. The `squeeze()` ensures that we are ignoring single
        # dimensions such as channels.
        pers_info_pred = self.cubical_complex(pred_obj_.squeeze())
        pers_info_true = self.cubical_complex(true_obj_.squeeze())

        pers_info_pred = [
            x for x in batch_iter(pers_info_pred, dim=self.topo_feat_d)
        ]

        pers_info_true = [
            x for x in batch_iter(pers_info_true, dim=self.topo_feat_d)
        ]

        topo_loss = torch.stack([
            self.topo_loss(pred_batch, true_batch)
            for pred_batch, true_batch in zip(pers_info_pred, pers_info_true)
        ])

        self.log("topo_loss", topo_loss.mean()),
        return self.topo_lambda * topo_loss.mean()


    def training_step(self, train_batch, batch_idx):
        images, true_obj = train_batch
        pred = self(images)
        loss = self.binary_crossentropy_Dice(pred, true_obj)

        loss += self.topological_step(pred, true_obj)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        images, true_obj = val_batch
        pred = self(images)
        loss = self.binary_crossentropy_Dice(pred, true_obj)

        loss += self.topological_step(pred, true_obj)

        self.log("val_loss", loss)

    def train_dataloader(self):
        dataset = SHAPRDataset(self.path, self.cv_train_filenames, self.random_seed)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        return train_loader

    def val_dataloader(self):
        dataset = SHAPRDataset(self.path, self.cv_val_filenames, self.random_seed)
        val_loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        return val_loader

    def test_dataloader(self):
        dataset = SHAPRDataset(self.path, self.cv_test_filenames, self.random_seed)
        test_loader = DataLoader(dataset)
        return test_loader

# Define GAN
class LightningSHAPR_GANoptimization(pl.LightningModule):
    def __init__(self, settings, cv_train_filenames, cv_val_filenames):
        super(LightningSHAPR_GANoptimization, self).__init__()

        self.random_seed = settings.random_seed
        self.path = settings.path
        self.cv_train_filenames = cv_train_filenames
        self.cv_val_filenames = cv_val_filenames
        self.batch_size = settings.batch_size
        # Define model
        self.shapr = SHAPR()
        if settings.epochs_SHAPR > 0:
            list_of_weights = os.listdir(settings.path + "logs/")
            list_of_weights = [settings.path + "logs/" + wp for wp in list_of_weights]
            latest_weights = max(list_of_weights, key=os.path.getctime)
            checkpoint = torch.load(latest_weights, map_location=lambda storage, loc: storage)
            new_checkpoint = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if 'shapr' in k:
                    name = k[6:]  # remove `shapr.`
                else:
                    name = k
                new_checkpoint[name] = v

            self.shapr.load_state_dict(new_checkpoint)

        self.discriminator = Discriminator()
        self.lr = 0.0001
        self.loss = nn.CrossEntropyLoss()
        self.dice_loss = dice_loss

    def forward(self, z):
        return self.shapr(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def MSEloss(self, y_true, y_pred):
        MSE = torch.nn.MSELoss()
        return MSE(y_true, y_pred)

    def binary_crossentropy_Dice(self, y_pred, y_true):
        return (self.dice_loss(y_pred, y_true) + F.binary_cross_entropy(y_pred, y_true)) / 2
        #return (self.MSEloss(y_pred, y_true) + F.binary_cross_entropy(y_pred, y_true))/2

    def train_dataloader(self):
        dataset = SHAPRDataset(self.path, self.cv_train_filenames, self.random_seed)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        return train_loader

    def val_dataloader(self):
        dataset = SHAPRDataset(self.path, self.cv_val_filenames, self.random_seed)
        val_loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        return val_loader

    def test_dataloader(self):
        dataset = SHAPRDataset(self.path, self.cv_test_filenames, self.random_seed)
        test_loader = DataLoader(dataset)
        return test_loader

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        images, true_obj = train_batch
        valid = torch.ones(images.size(0), 1)
        valid = valid.type_as(images)

        fake = torch.zeros(images.size(0), 1)
        fake = fake.type_as(images)

        if optimizer_idx == 0:
            supervised_loss = self.binary_crossentropy_Dice(self(images), true_obj)
            g_loss = self.adversarial_loss(self.discriminator(self(images)), valid)
            print("supervised loss:", supervised_loss.item(), "gan loss:", g_loss.item())
            loss = (10*supervised_loss + g_loss) / 11
            tqdm_dict = {'g_loss': loss}
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # test discriminator on real images
            real_loss = self.adversarial_loss(self.discriminator(true_obj), valid)

            # how well can it label as fake?
            fake_loss = self.adversarial_loss(self.discriminator(self(images).detach()), fake)

            # test discriminator on fake images
            loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': loss}
            print("discriminator loss:", loss.item())
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def validation_step(self, val_batch, batch_idx):
        images, true_obj = val_batch
        loss = self.binary_crossentropy_Dice(true_obj, self(images))
        self.log("val_loss", loss)

    def configure_optimizers(self):
        lr_1 = 0.001
        b1_1 = 0.5
        b2_1 = 0.999
        lr_2 = 0.0001
        b1_2 = 0.5
        b2_2 = 0.999

        opt_s = torch.optim.Adam(self.shapr.parameters())#, lr=0.001)
        opt_d = torch.optim.Adam(self.discriminator.parameters(),lr = 0.00005)# lr=0.00000005)
        #opt_g = torch.optim.Adam(self.shapr.parameters(), lr=lr_1, betas=(b1_1, b2_1))
        #opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_2, betas=(b1_2, b2_2))
        return [opt_s, opt_d], []
