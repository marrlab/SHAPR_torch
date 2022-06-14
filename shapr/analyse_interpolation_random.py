"""Analyse interpolation errors of input data set."""

import torch

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def run_experiment(n, S, s, d, infinity_norm=False):
    losses = []
    for i in range(n):
        X = torch.as_tensor(rng.uniform(0, 1, size=(S,) * d)) \
            .unsqueeze(0)                                     \
            .unsqueeze(0)

        x = torch.nn.functional.interpolate(
            input=X, size=(s, ) * d,
            mode='nearest'
        )

        x = torch.nn.functional.interpolate(
            input=x, size=(S, ) * d,
            mode='nearest'
        )

        X = X.squeeze()
        x = x.squeeze()

        if infinity_norm:
            losses.append(torch.linalg.norm(
                (x - X).flatten(), ord=torch.inf, dim=None).numpy()
            )
        else:
            losses.append(torch.linalg.norm(
                (x - X).flatten(), ord=None, dim=None).numpy()
            )

    losses = np.asarray(losses)

    L = np.max(losses)
    U = np.sqrt(S ** d)

    print(U, L, U - L)

    return pd.DataFrame.from_dict({s: losses.tolist()}, orient='index')


if __name__ == '__main__':

    n = 100
    S = 64
    d = 3

    rng = np.random.default_rng(42)
    data = []

    for s in [2, 4, 8, 16, 32]:
        data.append(run_experiment(n, S, s, d, False))

    data = pd.concat(data).T

    sns.boxplot(data=pd.melt(data), x='variable', y='value')
    plt.show()
