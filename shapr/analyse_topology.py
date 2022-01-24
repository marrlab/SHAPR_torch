"""Analyse topology of input data set."""

import collections
import itertools
import os
import torch
import sys

import numpy as np
import pandas as pd

from shapr import settings
from shapr.data_generator import SHAPRDataset

from torch.utils.data import DataLoader

from torch_topological.nn import CubicalComplex

from torch_topological.utils import total_persistence
from torch_topological.utils import persistent_entropy


def calculate_statistics(diagrams):
    """Calculate statistics of batch of persistence diagrams."""
    result = collections.defaultdict(list)

    stat_fns = {
        'total_persistence': total_persistence,
        'persistent_entropy': persistent_entropy,
    }

    for name, fn in stat_fns.items():
        for diagram in diagrams:
            result[name].append(fn(diagram).numpy())

    return pd.DataFrame.from_dict(result)


if __name__ == '__main__':
    all_files = os.listdir(
        os.path.join(settings.path, "obj/")
    )

    data_set = SHAPRDataset(settings.path, all_files)
    loader = DataLoader(data_set, batch_size=8, shuffle=True)

    cubical_complex = CubicalComplex(dim=3)

    all_dfs = []
    index = 0

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
            df = calculate_statistics(diagrams)
            df['dimension'] = dim
            df['index'] = np.arange(len(objects)) + index

            all_dfs.append(df)

        index += len(objects)

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(sys.stdout, index=False)

    # Create feature vector representation: we just group by index, then
    # ravel all measurements over all dimensions.
    X = []

    for name, group in df.groupby('index'):
        cols = group.columns
        cols = [c for c in cols if c != 'dimension' and c != 'index']

        F = group[cols].to_numpy().ravel()
        X.append(F)

    X = np.asarray(X)
