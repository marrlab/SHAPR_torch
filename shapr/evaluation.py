"""Evaluate data with respect to ground truth."""

import argparse
import collections
import os
import trimesh

from tqdm import tqdm

import numpy as np
import pandas as pd

from skimage import measure
from skimage.io import imread


import matplotlib.pyplot as plt
import seaborn as sns

from skimage.filters import threshold_otsu


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
    """Calculate surface area using a mesh."""
    verts_pred, faces_pred, _, _ = measure.marching_cubes(
        obj * 255.,
        method='lewiner'
    )
    surface_pred = measure.mesh_surface_area(verts_pred, faces_pred)
    return surface_pred


def IoU(y_true, y_pred):
    """Calculate IoU between ground truth and prediction."""
    intersection = y_true + y_pred
    intersection = np.count_nonzero(intersection > 1.5)
    union = y_true + y_pred
    union = np.count_nonzero(union > 0.5)
    return intersection / union


def get_roughness(obj):
    """Calculate surface roughness using mesh."""
    verts, faces, _, _ = measure.marching_cubes(
        obj * 255.,
        method='lewiner',
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    smesh = trimesh.smoothing.filter_humphrey(mesh)

    # Additional conversion requird since we are getting
    # a `TrackedArray` that does not play nice with SNS.
    roughness = np.mean(np.sqrt((np.sum((verts-smesh.vertices)**2))))
    roughness = np.asarray(roughness)

    return roughness


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
        size=1.5,
        orient='h',
        ax=ax
    )

    ax.set_xlabel(label, size=15)
    ax.set_xlim(-0.01, 1.01)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('SOURCE', type=str, help='Source directory')
    parser.add_argument('TARGET', type=str, nargs='+', help='Target directory')

    args = parser.parse_args()

    iou_inv = collections.defaultdict(list)
    volume = collections.defaultdict(list)
    surface = collections.defaultdict(list)
    roughness = collections.defaultdict(list)

    filenames = sorted(os.listdir(args.SOURCE))

    for filename in tqdm(filenames[:5], desc='File'):
        source = imread(os.path.join(args.SOURCE, filename)) / 255.0

        for target_ in args.TARGET:
            target = np.squeeze(
                norm_thres(np.nan_to_num(
                        imread(os.path.join(target_, filename))
                    )
                )
            )

            name = os.path.basename(target_)

            if np.mean(source) > 0.1:
                iou_inv[name].append(1 - IoU(source, target))
                volume[name].append(
                    np.abs(np.sum(target) - np.sum(source)) / np.sum(source)
                )

                source_surface = get_surface(source)
                surface[name].append(
                    np.abs(get_surface(target) - source_surface) / source_surface
                )

                source_roughness = get_roughness(source)
                roughness[name].append(
                    np.abs(get_roughness(target) - source_roughness)
                    / source_roughness
                )

    fig, axes = plt.subplots(nrows=4, squeeze=True, figsize=(5, 6))

    swarmplot(pd.DataFrame.from_dict(iou_inv), '1 - IoU', axes[0])
    swarmplot(pd.DataFrame.from_dict(volume), 'Volume error', axes[1])
    swarmplot(pd.DataFrame.from_dict(surface), 'Surface error', axes[2])
    swarmplot(pd.DataFrame.from_dict(roughness), 'Roughness error', axes[3])

    plt.tight_layout()
    plt.show()

########################################################################
# HIC SVNT DRACONES
########################################################################

mask_path = "/media/dominik/LaCie/SHAPR_pytorch/Organoid/mask/"

volume = []
mask_area = []

files = os.listdir(pytorch_path)
print(len(files))

fname = []
for index, file in enumerate(files): 
    pytorchdata = np.squeeze(norm_thres(np.nan_to_num(imread(pytorch_path + file))))
    mask = np.squeeze(norm_thres(np.nan_to_num(imread(mask_path + file))))
    volume.append(np.sum(pytorchdata))

    mask_area.append(np.sum(mask))

plt.scatter(mask_area, volume, s = 1)
plt.show()
