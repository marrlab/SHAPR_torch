"""Evaluate data with respect to ground truth."""

import argparse
import os
import trimesh

from tqdm import tqdm

import numpy as np

from skimage import measure
from skimage.io import imread


import matplotlib.pyplot as plt
import seaborn as sns

from skimage.filters import threshold_otsu


'''Threshold data using Otus methods'''
def norm_thres(data): 
    maxd = np.max(data)
    data = np.nan_to_num(data / maxd)
    if np.max(data)  > 0:
        thresh = threshold_otsu(data)
        binary = data > thresh
    else: 
        binary = data
    return binary*1.0


'''Calculate surface using a mesh'''
def get_surface(obj):
    verts_pred, faces_pred, _, _ = measure.marching_cubes(
        obj * 255.,
        method='lewiner'
    )
    surface_pred = measure.mesh_surface_area(verts_pred, faces_pred)
    return surface_pred


'''Calculate IoU between GT and prediction'''
def IoU(y_true,y_pred): 
    intersection = y_true + y_pred
    intersection = np.count_nonzero(intersection > 1.5)
    union = y_true + y_pred
    union = np.count_nonzero(union > 0.5)
    return intersection / union


'''Calculate surface roughness using a mesh'''
def get_roughness(obj):
    verts, faces, _, _ = measure.marching_cubes(
        obj * 255.,
        method='lewiner',
    )
    mesh = trimesh.Trimesh(vertices = verts,faces=faces,process=False)
    smesh = trimesh.smoothing.filter_humphrey(mesh)

    # Additional conversion requird since we are getting
    # a `TrackedArray` that does not play nice with SNS.
    roughness = np.mean(np.sqrt((np.sum((verts-smesh.vertices)**2))))
    roughness = np.asarray(roughness)

    return roughness


def swarmplot(data, label, ax):
    """Create swarmplot with specific label."""
    ax = sns.violinplot(
        x=data,
        showfliers=False,
        color='lightgray',
        boxprops={'facecolor': 'None'},
        orient='h',
        ax=ax
    )

    ax = sns.swarmplot(
        x=data,
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
    parser.add_argument('TARGET', type=str, help='Target directory')

    args = parser.parse_args()

    iou = []
    volume = []
    surface = []
    roughness = []

    filenames = sorted(os.listdir(args.SOURCE))
    processed = []

    for filename in tqdm(filenames, desc='File'):
        source = imread(os.path.join(args.SOURCE, filename)) / 255.0
        target = np.squeeze(
            norm_thres(np.nan_to_num(
                    imread(os.path.join(args.TARGET, filename))
                )
            )
        )

        if np.mean(source) > 0.1:
            iou.append(IoU(source, target))
            volume.append(
                np.abs(np.sum(target) - np.sum(source)) / np.sum(source)
            )

            source_surface = get_surface(source)
            surface.append(
                np.abs(get_surface(target) - source_surface) / source_surface
            )

            source_roughness = get_roughness(source)
            roughness.append(
                np.abs(get_roughness(target) - source_roughness)
                / source_roughness
            )

    print(len(iou))
    print(np.mean(iou) * 100, np.std(iou) * 100)

    fig, axes = plt.subplots(nrows=4, squeeze=True, figsize=(5, 6))

    swarmplot(1 - np.asarray(iou), '1 - IoU', axes[0])
    swarmplot(volume, 'Volume error', axes[1])
    swarmplot(surface, 'Surface error', axes[2])
    swarmplot(roughness, 'Roughness error', axes[3])

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
