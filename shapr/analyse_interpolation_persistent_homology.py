"""Analyse interpolation errors of input data set (in terms of TDA)."""

import argparse
import os
import torch

from shapr._settings import SHAPRConfig
from shapr.data_generator import SHAPRDataset

from torch.utils.data import DataLoader

from torch_topological.nn import CubicalComplex
from torch_topological.nn import WassersteinDistance


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

    loss_fn = WassersteinDistance(q=1)

    cubical_complex = CubicalComplex(dim=3)

    for _, objects in loader:
        objects_interp = torch.nn.functional.interpolate(
            input=objects, size=(size, ) * 3,
            mode='trilinear',
            align_corners=True,
        )

        objects = objects.squeeze()
        objects_interp = objects_interp.squeeze()

        pers_info_source = cubical_complex(objects)
        pers_info_interp = cubical_complex(objects_interp)

        loss = torch.stack([
            loss_fn(pred_batch, true_batch)
            for pred_batch, true_batch in
            zip(pers_info_source, pers_info_interp)
        ])

        print(loss.mean())
