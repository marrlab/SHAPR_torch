"""Analyse `wandb` output files."""

import argparse

import numpy as np
import pandas as pd


def show_best_performance(df):
    """Extract best performance over different scenarios."""
    # Only use finished runs; plus, remove columns that consist of only
    # NaN values, as they might interfere with grouping.
    df = df.query('State == "finished"')
    df = df.dropna(axis=1, how='all')

    metric_columns = [
        col for col in df.columns if col.startswith('test')
    ]

    topo_params = [
        col for col in df.columns if col.startswith('topo_')
    ]

    # Specify aggregation functions for the `groupby` operation below.
    agg = {
        metric: [np.mean, np.std] for metric in metric_columns
    }

    df_grouped = df.groupby(topo_params).agg(agg)
    print(df_grouped)

    means = [
        (c, h) for (c, h) in df_grouped.columns if h == 'mean'
    ]

    n_levels = len(df_grouped.index.names)

    print('Minima are achieved by the following groups:\n')
    for level in range(n_levels):
        print(df_grouped[means].groupby(level=level).idxmin())

    # Aggregate over all scenarios and find the best compromise
    # solution.
    print('\nMean performance over all scores:\n')
    print(df_grouped[means].mean(axis=1))

    print(
        '\nOverall minimum performance:',
        df_grouped[means].mean(axis=1).idxmin(axis=0)
    )

    print('\nRestricting to scenarios with small regularisation strength:\n')

    # Filter scenario that we converged on.
    df = df.query('topo_lambda < 1.0 and topo_interp < 24')

    if len(df['topo_lambda'].unique()) <= 1:
        topo_params.remove('topo_lambda')
        df = df.drop('topo_lambda', axis='columns')

    df_grouped = df.groupby(topo_params).agg(agg)
    print(df_grouped)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILENAME', type=str, help='Input filename')

    args = parser.parse_args()

    df = pd.read_csv(args.FILENAME)
    show_best_performance(df)
