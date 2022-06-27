"""Evaluate data with respect to ground truth and plot it.

This script generates plots for evaluating runs of SHAPR. Different
scenarios or models can be tested against a "ground truth" data set
and the results are stored in PDF files.

This script requres a configuration file to work.
"""

import argparse
import collections
import copy
import json
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_dilation

from skimage.filters import threshold_otsu
from skimage.io import imread


plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': False,
    'svg.fonttype': 'none',
})


def parse_config(filename):
    """Parse JSON configuration file.

    Parameters
    ----------
    filename : str
        Input filename; must be a JSON configuration file.

    Returns
    -------
    dict
        The parsed configuration.
    """
    with open(filename) as f:
        config = json.load(f)

    assert 'source' in config
    assert 'targets' in config

    return config


def norm_thres(data):
    """Perform Otsu's method for thresholding."""
    maxd = np.max(data)
    data = np.nan_to_num(data / maxd)
    if np.max(data) > 0:
        thresh = threshold_otsu(data)
        binary = data > thresh
    else:
        binary = data

    return binary * 1.0


def get_surface(obj):
    """Calculate surface area."""
    k = np.ones((3,3,3), dtype=int)
    obj = obj.astype(int)
    boundary = binary_dilation(obj==0, k) & obj
    return np.sum(boundary)


def IoU(y_true, y_pred):
    """Calculate IoU between ground truth and prediction."""
    intersection = y_true + y_pred
    intersection = np.count_nonzero(intersection > 1.5)
    union = y_true + y_pred
    union = np.count_nonzero(union > 0.5)
    return intersection / union


def get_roughness(obj):
    """Calculate surface roughness."""
    smoothed_obj = copy.deepcopy(gaussian_filter(obj, 2))
    smoothed_obj = smoothed_obj > 0.5
    return np.sum(np.abs(smoothed_obj - obj))


def evaluate(source, target, quick=False):
    """Evaluate source and target image along different dimensions.

    Parameters
    ----------
    quick : bool
        If set, only calculates metrics that do not require extensive
        surface meshing operations.

    Returns
    -------
    dict
        A dictionary with the keys being abbreviations of the error
        metrics, and the values being the result of the comparison.
        Together, these dictionaries will be combined into a larger
        data frame that contains the raw values.
    """
    data = {
        'iou_error': (1 - IoU(source, target)),
        'volume_error': (
            np.abs(np.sum(target) - np.sum(source)) / np.sum(source)
        )
    }

    if not quick:
        source_surface = get_surface(source)
        data['surface_error'] = (
            np.abs(get_surface(target) - source_surface)
            / source_surface
        )

        source_roughness = get_roughness(source)
        data['roughness_error'] = (
            np.abs(get_roughness(target) - source_roughness)
            / source_roughness
        )

    return data


def make_df(data, metric):
    """Create data frame for all targets, using a given metric.

    Parameters
    ----------
    data : dict of `pd.DataFrame`
        Data frames; the keys are supposed to belong to different
        targets, while the values store all evaluated metrics.

    metric : str
        Metric to extract; must refer to a valid column of all the data
        frames.

    Returns
    -------
    pd.DataFrame
        Long-form data set storing the results of `metric`.
    """
    result = {}

    for target in data:
        result[target] = data[target][metric]

    return pd.DataFrame.from_dict(result)


def swarmplot(data, label, ax):
    """Create swarmplot with specific label."""
    ax = sns.violinplot(
        data=data,
        showfliers=False,
        color='lightgray',
        boxprops={'facecolor': 'None'},
        orient='h',
        ax=ax
    )

    ax = sns.swarmplot(
        data=data,
        color='.25',
        size=1.0,
        orient='h',
        ax=ax
    )

    ax.set_xlabel(label, size=15)
    ax.set_xlim(-0.01, 1.01)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('CONFIG', type=str, help='Configuration file')

    parser.add_argument(
        '-q', '--quick',
        action='store_true',
        help='If set, only calculates statistics that can be obtained '
             'efficiently.'
    )

    args = parser.parse_args()
    config = parse_config(args.CONFIG)

    # Stores all metrics calculated during the script run. The key
    # refers to one of the targets that is configured by the user,
    # while the values are the resulting metrics.
    data = collections.defaultdict(list)

    source_path = config['source']['path']
    filenames = sorted(os.listdir(source_path))
    processed = []

    for filename in tqdm(filenames, desc='File'):
        source = norm_thres(
            imread(os.path.join(source_path, filename))
        )

        # First check whether all files exist; else, we skip processing
        # in order to ensure consistent lists.
        skip = False
        for target_ in config['targets']:
            target_path = os.path.join(target_['path'], filename)

            if not os.path.exists(target_path):
                skip = True
                break

        if skip:
            continue
        else:
            processed.append(os.path.basename(filename))

        for target_ in config['targets']:
            target_path = os.path.join(target_['path'], filename)

            target = np.squeeze(
                norm_thres(np.nan_to_num(
                        imread(target_path)
                    )
                )
            )

            label = target_['label']
            data[label].append(
                evaluate(source, target, args.quick)
            )

    for target in data.keys():
        data[target] = pd.DataFrame.from_records(data[target])
        data[target]['filename'] = processed

    fig, axes = plt.subplots(
        ncols=4 - 2 * args.quick,
        squeeze=True,
        sharey=True,
        figsize=(10, 6)
    )

    filenames = list(map(os.path.basename, filenames))

    if len(filenames) > len(processed):
        skipped = sorted(set(filenames) - set(processed))
        print(f'Skipped some files: {skipped}')

    swarmplot(make_df(data, 'iou_error'), '1 - IoU', axes[0])
    swarmplot(make_df(data, 'volume_error'), 'Rel. volume error', axes[1])

    if not args.quick:
        swarmplot(
            make_df(data, 'surface_error'),
            'Rel. surface area error',
            axes[2],
        )

        swarmplot(
            make_df(data, 'roughness_error'),
            'Rel. surface roughness error',
            axes[3],
        )

    plt.tight_layout()
    plt.savefig(config['output'], backend='svg')
    plt.show()
