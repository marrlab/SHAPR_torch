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
    parser.add_argument(
        '-s', '--size',
        type=int,
        help='If set, overrides size specified in configuration'
    )

    args = parser.parse_args()

    settings = SHAPRConfig(params=args.params)

    all_files = os.listdir(
        os.path.join(settings.path, "obj/")
    )

    data_set = SHAPRDataset(settings.path, all_files, settings.random_seed)
    loader = DataLoader(data_set, batch_size=8, shuffle=True)

    size = settings.topo_interp

    if args.size is not None:
        size = args.size

    loss_fn = MSELoss()

    for _, objects in loader:
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

        loss = loss_fn(objects, objects_recon)
        print(size, loss.mean().numpy())
