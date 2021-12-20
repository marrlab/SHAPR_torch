import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from data_generator import SHAPRDataset
from torch.utils.data import DataLoader, random_split
#from .metrics import *
import torchvision
from collections import OrderedDict

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

        self.path = settings.path
        self.cv_train_filenames = cv_train_filenames
        self.cv_val_filenames = cv_val_filenames
        self.batch_size = settings.batch_size
        # Define model
        self.shapr = SHAPR()

        # Define learning rate
        self.lr = 0.01

        # Defining loss
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.shapr(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def MSEloss(self, y_true, y_pred):
        MSE = torch.nn.MSELoss()
        return MSE(y_true, y_pred)

    def training_step(self, train_batch, batch_idx):
        images, true_obj = train_batch
        pred = self.forward(images)
        loss = self.MSEloss(true_obj, pred)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        images, true_obj = val_batch
        pred = self.forward(images)
        loss = self.MSEloss(true_obj, pred)
        self.log("val_loss", loss)

    def train_dataloader(self):
        dataset = SHAPRDataset(self.path, self.cv_train_filenames)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        return train_loader

    def val_dataloader(self):
        dataset = SHAPRDataset(self.path, self.cv_val_filenames)
        val_loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        return val_loader

# Define GAN
class LightningSHAPR_GANoptimization(pl.LightningModule):
    def __init__(self, settings, cv_train_filenames, cv_val_filenames):
        super(LightningSHAPR_GANoptimization, self).__init__()

        self.path = settings.path
        self.cv_train_filenames = cv_train_filenames
        self.cv_val_filenames = cv_val_filenames
        self.batch_size = settings.batch_size
        # Define model
        self.shapr = SHAPR()
        list_of_files = [f for f in os.listdir(settings.path + "/logs/") if "SHAPR_training" in f]
        max(list_of_files, key=os.path.getctime)
        self.shapr.load_from_checkpoint(settings.path + "/logs/")
        self.discriminator = Discriminator()
        # Define learning rate
        self.lr = 0.01

        # Defining loss
        self.loss = nn.CrossEntropyLoss()

    def forward(self, z):
        return self.shapr(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def train_dataloader(self):
        dataset = SHAPRDataset(self.path, self.cv_train_filenames)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        return train_loader

    def val_dataloader(self):
        dataset = SHAPRDataset(self.path, self.cv_val_filenames)
        val_loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        return val_loader

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, obj_true = batch

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_obj = self(imgs)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self.generated_obj), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(obj_true), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self.generated_obj.detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lr_1 = 0.01
        b1_1 = 0.5
        b2_1 = 0.999
        lr_2 = 0.01
        b1_2 = 0.5
        b2_2 = 0.999

        opt_g = torch.optim.Adam(self.parameters(), lr=lr_1, betas=(b1_1, b2_1))
        opt_d = torch.optim.Adam(self.parameters(), lr=lr_2, betas=(b1_2, b2_2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.shapr.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)