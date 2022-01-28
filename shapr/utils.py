from skimage.io import imread
import numpy as np
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize, rotate
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time
import torch


# Mohammad: Does scikit-image imread take care of all type of image?
def import_image(path_name):
    '''
    This function loads the image from the specified path
    NOTE: The alpha channel is removed (if existing) for consistency
    Args:
        path_name (str): path to image file

    return:
        image_data: numpy array containing the image data in at the given path.
    '''
    if path_name.endswith('.npy'):
        image_data = np.array(np.load(path_name))
    else:
        image_data = imread(path_name)
        # If has an alpha channel, remove it for consistency
    if np.array(np.shape(image_data))[-1] == 4:
        image_data = image_data[: ,: ,0:3]
    return image_data