import torch
import torch.nn as nn
from shapr.utils import *
import torch.nn.functional as F
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # adapted from: https://github.com/milesial/Pytorch-UNet/
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def dice_crossentropy_loss(y_true, y_pred):
    """
    Adding the dice loss and the binary crossentropy as well as a penalty for the volume
    """
    binary_crossentropy = torch.nn.BCELoss()
    return DiceLoss(y_true, y_pred) #+ binary_crossentropy(y_true, y_pred)

def mse(y_true, y_pred):
    MSE = torch.nn.MSELoss()
    return MSE(y_true, y_pred)

def IoU(y_true,y_pred):
    intersection = y_true + y_pred
    intersection = np.count_nonzero(intersection > 1.5)
    union = y_true + y_pred
    union = np.count_nonzero(union > 0.5)
    return intersection / union
