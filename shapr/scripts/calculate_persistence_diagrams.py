"""Calculate persistence diagrams of imput images."""

import argparse
import os
import torch

import numpy as np
import torch.nn as nn

from shapr.utils import import_image

from torch_topological.nn import CubicalComplex
from torch_topological.nn import WassersteinDistance

from torch_topological.utils import total_persistence

import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'FILE',
        nargs='+',
        type=str,
        help='Input file(s)'
    )
    parser.add_argument(
        '-s', '--size',
        type=int,
        help='Specifies size for interpolation'
    )

    args = parser.parse_args()

    cubical_complex = CubicalComplex(dim=3)

    # Collects all persistence information in order to calculate
    # distances afterwards.
    pers_info_all = []

    for filename in args.FILE:
        img = import_image(filename) / 255.0
        img = torch.tensor(img.squeeze())

        if args.size is not None:
            img = img.unsqueeze(dim=0).unsqueeze(dim=0)
            size = (args.size,) * 3
            img = nn.functional.interpolate(
                input=img,
                size=size,

            ).squeeze()

        img = img.unsqueeze(dim=0)

        pers_info = cubical_complex(img)[0]
        pers_info_all.append(pers_info)

        fig, axes = plt.subplots(ncols=len(pers_info), squeeze=True)

        for dim, info in enumerate(pers_info):
            diagram = info.diagram

            axes[dim].set_aspect('equal')
            axes[dim].set_xlim(-0.1, 1.1)
            axes[dim].set_ylim(-0.1, 1.1)
            axes[dim].scatter(diagram[:, 0], diagram[:, 1])
            axes[dim].plot([-0.1, 1.1], [-0.1, 1.1], 'k')

            print(info.diagram)

            name = os.path.basename(filename)
            name = os.path.splitext(name)[0]
            name = f'{name}_d{dim}.txt'

            np.savetxt(
                os.path.join('/tmp', name),
                info.diagram.numpy(),
                fmt='%.4f',
                header='creation destruction',
                comments='',
            )

    wasserstein = WassersteinDistance(q=2)

    for x, y in zip(pers_info_all, pers_info_all[1:]):
        for dim, (x_, y_) in enumerate(zip(x, y)):
            dist = wasserstein(x_, y_)

            print('Dimension:', dim)
            print(
                'Total persistence:',
                total_persistence(x_.diagram, p=1),
                total_persistence(y_.diagram, p=1)
            )
            print('Distance:', dist)

    plt.show()
