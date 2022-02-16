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

    print(df.groupby(topo_params).agg(agg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILENAME', type=str, help='Input filename')

    args = parser.parse_args()

    df = pd.read_csv(args.FILENAME)
    show_best_performance(df)

