import numpy as np
from shapr.utils import *
from shapr.metrics import *
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

def augmentation(obj, img, random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    if random.choice([True, True, False]) == True:
        obj = np.flip(obj, len(np.shape(obj)) - 1).copy()
        img = np.flip(img, len(np.shape(img)) - 1).copy()
    if random.choice([True, True, False]) == True:
        obj = np.flip(obj, len(np.shape(obj)) - 2).copy()
        img = np.flip(img, len(np.shape(img)) - 2).copy()

    if random.choice([True, True, False]) == True:
        angle = np.random.choice(int(360 * 100)) / 100
        img = np.nan_to_num(rotate(img, angle, resize=False, preserve_range=True))
        for i in range(0, np.shape(obj)[0]):
            obj[i, :, :] = np.nan_to_num(rotate(obj[i, :, :], angle, resize=False, preserve_range=True))

    if random.choice([True, True, False]) == True:
        from skimage.util import random_noise
        img = random_noise(img, mode='gaussian', var= 0.02)

    if random.choice([True, True, False]) == True:
        obj_shape = np.shape(obj)
        img_shape = np.shape(img)
        x_shift = np.random.choice(int(40))
        y_shift = np.random.choice(int(40))
        x_shift2 = np.random.choice(int(40))
        y_shift2 = np.random.choice(int(40))
        z_shift = np.random.choice(int(10))
        z_shift2 = np.random.choice(int(10))
        obj = obj[z_shift:-(z_shift2+1), x_shift:-(x_shift2+1), y_shift:-(y_shift2+1)]
        img = img[int(x_shift/4):-int(x_shift2/4+1), int(y_shift/4):-int(y_shift2/4+1),:]
        obj = resize(obj, obj_shape, preserve_range=True)
        img = resize(img, img_shape, preserve_range=True)
    return obj, img



"""
The data generator will open the 3D segmentation, 2D masks and 2D images for each fold from the directory given the filenames and return a tensor
The 2D mask and the 2D image will be multiplied pixel-wise to remove the background
"""

class SHAPRDataset(Dataset):
    def __init__(self, path, filenames, random_seed):
        self.path = path
        self.filenames = filenames
        self.random_seed = random_seed

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        obj = import_image(os.path.join(self.path, "obj", self.filenames[idx])) / 255.
        img = import_image(os.path.join(self.path, "mask", self.filenames[idx])) / 255.
        bf = import_image(os.path.join(self.path, "image", self.filenames[idx])) / 255.
        msk_bf = np.zeros((2, int(np.shape(img)[0]), int(np.shape(img)[1])))
        msk_bf[0, :, :] = img
        msk_bf[1, :, :] = bf * img
        obj, msk_bf = augmentation(obj, msk_bf, self.random_seed)
        mask_bf = msk_bf[:, np.newaxis, ...]
        obj = obj[np.newaxis,:,:,:]
        return torch.from_numpy(mask_bf).float(), torch.from_numpy(obj).float()


def get_test_image(self, filename):
    img = import_image(os.path.join(self.path, "mask", filename)) / 255.
    bf = import_image(os.path.join(self.path, "image", filename)) / 255.
    obj = import_image(os.path.join(self.path, "obj", filename)) / 255.
    mask_bf = np.zeros((2, 1, int(np.shape(img)[0]), int(np.shape(img)[1])))
    mask_bf[0, 0, :, :] = img
    mask_bf[1, 0, :, :] = bf * img
    mask_bf = mask_bf[np.newaxis,...]
    return mask_bf, obj


