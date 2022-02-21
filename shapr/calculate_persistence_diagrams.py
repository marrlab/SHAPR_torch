"""Calculate persistence diagrams of imput images."""

import argparse
import torch

from shapr.utils import import_image

from torch_topological.nn import CubicalComplex
from torch_topological.nn import WassersteinDistance

import matplotlib.pyplot as plt


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

    # Collects all persistence information in order to calculate
    # distances afterwards.
    pers_info_all = []

    for filename in args.FILE:
        img = import_image(filename) / 255.0
        img = torch.tensor(img.squeeze())
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

    plt.show()
