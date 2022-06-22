import torch.nn.functional as F
import torch
import torch.nn as nn
import sys

t = torch.zeros((1, 1, 9, 9))

t.fill_(10000000)
t[0][0][4][4] = 0

pred_obj_ = nn.functional.interpolate(
            input=t,
            size=(3, 3),
            mode='area',
        )
print(pred_obj_)
