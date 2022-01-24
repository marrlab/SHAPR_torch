"""Analyse topology of input data set."""

import itertools
import os
import torch

from shapr import settings
from shapr.data_generator import SHAPRDataset

from torch.utils.data import DataLoader

from torch_topological.nn import CubicalComplex

from torch_topological.utils import total_persistence
from torch_topological.utils import persistent_entropy


if __name__ == '__main__':
    all_files = os.listdir(
        os.path.join(settings.path, "obj/")
    )

    data_set = SHAPRDataset(settings.path, all_files)
    loader = DataLoader(data_set, batch_size=8, shuffle=True)

    cubical_complex = CubicalComplex(dim=3)

    for _, objects in loader:
        # TODO: Make this configurable or read it from settings?
        size = 16
        objects = torch.nn.functional.interpolate(
            input=objects, size=(size, ) * 3,
        )

        pers_info = cubical_complex(objects.squeeze())

        for dim in range(3):
            pers_info_ = [
                [x__ for x__ in x_ if x__.dimension == dim]
                for x_ in pers_info
            ]

            diagrams = [
                [x__.diagram for x__ in x_]
                for x_ in pers_info_
            ]

            diagrams = list(itertools.chain.from_iterable(diagrams))
