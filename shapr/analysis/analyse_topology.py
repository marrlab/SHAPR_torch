"""Analyse topology of input data set."""

import argparse
import collections
import itertools
import os
import torch
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import MDS

from shapr._settings import SHAPRConfig
from shapr.data_generator import SHAPRDataset

from torch.utils.data import DataLoader

from torch_topological.nn import CubicalComplex
from torch_topological.nn import WassersteinDistance

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

    return pd.DataFrame.from_dict(result).astype(float)


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

    cubical_complex = CubicalComplex(dim=3)

    all_dfs = []
    index = 0

    for _, objects in loader:
        size = settings.topo_interp
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

            # Experiment a little bit with the Wasserstein distance
            # here, at least with the first batch.
            if index == 0:
                wasserstein = WassersteinDistance(q=2)
                pers_info_ = itertools.chain.from_iterable(pers_info_)

                for X, Y in zip(pers_info_, pers_info_):
                    dist = wasserstein([X], [Y])
                    tp1 = total_persistence(X.diagram)
                    tp2 = total_persistence(Y.diagram)

                    if tp1 + tp2 < dist:
                        print('Ooops...')
                    else:
                        print(dist.numpy(), (tp1 + tp2).numpy())

            diagrams = list(itertools.chain.from_iterable(diagrams))
            df = calculate_statistics(diagrams)
            df['dimension'] = dim
            df['index'] = np.arange(len(objects)) + index

            all_dfs.append(df)

        index += len(objects)

    df = pd.concat(all_dfs, ignore_index=True).fillna(value=4711)
    df.to_csv(sys.stdout, index=False)

    print(df.describe())

    # Create feature vector representation: we just group by index, then
    # ravel all measurements over all dimensions.
    X = []

    for name, group in df.groupby('index'):
        cols = group.columns
        cols = [c for c in cols if c != 'dimension' and c != 'index']

        F = group[cols].to_numpy().ravel()
        X.append(F)

    X = np.asarray(X)

    mds = MDS()
    Z = mds.fit_transform(X)

    plt.scatter(Z[:, 0], Z[:, 1])
    plt.show()
