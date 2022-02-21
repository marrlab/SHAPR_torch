import torch
import torch.nn as nn
import torch.nn.functional as F

class Dice_loss(torch.nn.Module):
    def __init__(self):
        super(Dice_loss, self).__init__()

    def forward(self, y_pred, y_true):
        ones = torch.ones_like(y_true)

        mask = y_true != 2

        p0 = y_pred
        p1 = ones - y_pred
        g0 = y_true
        g1 = ones - y_true

        tp_num = torch.sum((p0 * g0)[mask])
        fp_num = torch.sum((p0 * g1)[mask])
        fn_num = torch.sum((p1 * g0)[mask])

        prec = (tp_num + 1) / (tp_num + fp_num + 1)
        rec = (tp_num + 1) / (tp_num + fn_num + 1)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        return 1 - f1

class Volume_error(torch.nn.Module):
    def __init__(self):
        super(Volume_error, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred_binary = (y_pred > 0.5).float()
        y_true_binary = (y_true > 0.5).float()
        return torch.abs(torch.count_nonzero(y_pred_binary)-torch.count_nonzero(y_true_binary))/torch.count_nonzero(y_true_binary)

class IoU_error(torch.nn.Module):
    def __init__(self):
        super(IoU_error, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred_binary = (y_pred > 0.5).float()
        y_true_binary = (y_true > 0.5).float()
        intersection = y_pred_binary + y_true_binary
        intersection = torch.count_nonzero(intersection > 1.5)
        union = y_pred_binary + y_true_binary
        union = torch.count_nonzero(union > 0.5)
        return 1 - intersection / union
