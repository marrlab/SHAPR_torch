"""Analyse interpolation errors of input data set."""

import argparse
import os
import torch

from shapr._settings import SHAPRConfig
from shapr.data_generator import SHAPRDataset

from torch.nn import MSELoss

from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--params',
        type=str,
        default=None,
        help='Config file'
    )

    args = parser.parse_args()

    settings = SHAPRConfig(params=args.params)

    all_files = os.listdir(
        os.path.join(settings.path, "obj/")
    )

    data_set = SHAPRDataset(settings.path, all_files, settings.random_seed)
    loader = DataLoader(data_set, batch_size=8, shuffle=True)

    for _, objects in loader:
        size = settings.topo_interp 
        objects_interp = torch.nn.functional.interpolate(
            input=objects, size=(size, ) * 3,
            mode='trilinear',
            align_corners=True,
        )

        objects_recon = torch.nn.functional.interpolate(
            input=objects_interp, size=(64, ) * 3,
            mode='trilinear',
            align_corners=True,
        )

        diff = objects - objects_recon
        diff = diff.squeeze().view(diff.shape[0], -1)

        loss = torch.linalg.vector_norm(diff, ord=torch.inf, dim=1)
        print(loss.mean())
