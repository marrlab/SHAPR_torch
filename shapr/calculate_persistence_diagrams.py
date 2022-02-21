"""Calculate persistence diagrams of imput images."""

import argparse
import collections
import itertools
import os
import torch
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapr.utils import import_image

from torch_topological.nn import CubicalComplex
from torch_topological.nn import WassersteinDistance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'FILE',
        nargs='+',
        type=str,
        help='Input file(s)'
    )

    args = parser.parse_args()

    cubical_complex = CubicalComplex(dim=3)

    all_dfs = []
    index = 0

    for filename in args.FILE:
        img = import_image(filename) / 255.0
        img = torch.tensor(img.squeeze())
        img = img.unsqueeze(dim=0)

        pers_info = cubical_complex(img)[0]

        for dim, info in enumerate(pers_info):
            print(info.diagram)
