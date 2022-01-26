"""Evaluate data with respect to ground truth."""

import argparse
import os

from tqdm import tqdm

import numpy as np
from skimage.io import imread, imsave
from skimage import measure
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
import seaborn as sns
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import gaussian_filter
import copy
from scipy.ndimage.morphology import binary_fill_holes
from scipy.stats import wilcoxon, mannwhitneyu


#tf_path = "/media/dominik/LaCie/ShapeAE/Organoid_Manual_Segmentation/ShapeAE_dataset/Results_paper/ShapeAE_results/"
#pytorch_path = "/media/dominik/LaCie/SHAPR_pytorch/Organoid/results/"
#dataset_path = "/media/dominik/LaCie/ShapeAE/Organoid_Manual_Segmentation/ShapeAE_dataset/"


tf_path = "./results/first-try"
#pytorch_path = "/media/dominik/LaCie/SHAPR_pytorch/Organoid/results/"
#dataset_path = "/media/dominik/LaCie/ShapeAE/Organoid_Manual_Segmentation/ShapeAE_dataset/"

data_path_org = "../docs/sample/"
data_path_res = "../docs/sample/results/"

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


def swarmplot(data, label):
    """Create swarmplot with specific label."""
    fig, ax = plt.subplots(figsize=(5, 6))
    ax = sns.violinplot(
        data=data,
        showfliers=False,
        color='lightgray',
        boxprops={'facecolor': 'None'},
        orient='h'
    )

    ax = sns.swarmplot(data=data, color='.25', size=1.5, orient='h')
    plt.xlabel(label, size=15)
    plt.xlim(-0.01, 1.01)
    plt.xticks(size=15)
    plt.locator_params(axis='x', nbins=4)
    plt.tight_layout()
    plt.grid(b=None)


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

    for filename in tqdm(filenames[:10], desc='File'):
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

            print(type(source_roughness))
            print(dir(source_roughness))

    print(len(iou))
    print(np.mean(iou) * 100, np.std(iou) * 100)

    swarmplot(1 - np.asarray(iou), '1 - IoU')
    swarmplot(volume, 'Volume error')
    swarmplot(surface, 'Surface error')
    swarmplot(roughness, 'Roughness error')

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
