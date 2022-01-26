"""Evaluate data with respect to ground truth."""

import os

import numpy as np
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
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
    return np.mean(np.sqrt((np.sum((verts-smesh.vertices)**2))))


'''Loop over files to obtain error metrics'''

iou = []
volume = []
surface = []
roughness = []

files = os.listdir(data_path_res)
print(len(files))

fname = []
for index, file in enumerate(files):
    groundtruth = imread(os.path.join(data_path_org, "obj", file)) / 255.
    data = np.squeeze(norm_thres(np.nan_to_num(imread(data_path_res + file))))

    if all([np.mean(groundtruth)]) > 0.1:
        print(f'Processing {file}...')

        iou.append(IoU(groundtruth, data))
        volume.append(np.abs(np.sum(data)-np.sum(groundtruth))/np.sum(groundtruth))

        gt_surface = get_surface(groundtruth)
        surface.append(np.abs(get_surface(data) -gt_surface )/gt_surface)

        gt_roughness = get_roughness(groundtruth)
        roughness.append(np.abs(get_roughness(data)-gt_roughness)/gt_roughness)

        fname.append(file)
    else:
        print(f'Skipping {file}...')

raise 'heck'

# In[148]:


'''Plot 1-IoU'''
print(len(IoU_tf))
print(np.mean(IoU_tf)*100, np.std(IoU_tf)*100)
print(np.mean(IoU_pytorch)*100,np.std(IoU_pytorch)*100)
print(wilcoxon(IoU_pytorch, IoU_tf))

IoU_pytorch_inv = [1.-i for i in IoU_pytorch]
IoU_tf_inv = [1.-i for i in IoU_tf]

plt.figure(figsize =(5,6))
ax = sns.violinplot(data = [IoU_tf_inv, IoU_pytorch_inv], 
            showfliers=False,color='lightgray', boxprops={'facecolor':'None'}, orient = "h")
ax = sns.swarmplot(data = [IoU_tf_inv, IoU_pytorch_inv],color=".25", size =1.5, orient = "h")
plt.xlabel('IoU', size = 15)
plt.yticks([0,1],["TF", "pytorch"],size = 15)
plt.xticks(size = 15)
#plt.xlim(-0.01,0.99)
plt.locator_params(axis='x', nbins=4)
plt.tight_layout()
plt.grid(b=None)
plt.grid(b=None)

plt.show()


# In[149]:


'''Plot volume error'''
print(len(volume_pytorch))
print(np.mean(volume_tf)*100, np.std(volume_tf)*100)
print(np.mean(volume_pytorch)*100,np.std(volume_pytorch)*100 )
print(wilcoxon(volume_tf, volume_pytorch))

plt.figure(figsize =(5,6))
ax = sns.violinplot(data = [volume_tf, volume_pytorch], 
            showfliers=False,color='lightgray', boxprops={'facecolor':'None'}, orient = "h")
ax = sns.swarmplot(data = [volume_tf, volume_pytorch],color=".25", size =1.5, orient = "h")
plt.xlabel('Volume error [%]', size = 15)
plt.yticks([0,1],["TF", "pytorch"],size = 15)
plt.xticks(size = 15)
plt.xlim(-0.01,1.1)
plt.locator_params(axis='x', nbins=4)
plt.tight_layout()
plt.grid(b=None)
plt.grid(b=None)

plt.show()


# In[150]:


'''Plot surface error'''
print(np.mean(surface_tf)*100, np.std(surface_tf)*100)
print(np.mean(surface_pytorch)*100,np.std(surface_pytorch)*100 )
print(wilcoxon(volume_tf, volume_pytorch))

plt.figure(figsize =(5,6))
ax = sns.violinplot(data = [surface_tf, surface_pytorch], 
            showfliers=False,color='lightgray', boxprops={'facecolor':'None'}, orient = "h")
ax = sns.swarmplot(data = [surface_tf, surface_pytorch],color=".25", size =1.5, orient = "h")
plt.xlabel('Surface error [%]', size = 15)
plt.yticks([0,1],["TF", "pytorch"],size = 15)
plt.xticks(size = 15)
plt.xlim(-0.01,1.1)
plt.locator_params(axis='x', nbins=4)
plt.tight_layout()
plt.grid(b=None)
plt.grid(b=None)

plt.show()


# In[151]:


'''Plot surface roughness error'''
print(len(roughness_pytorch))
print(np.mean(roughness_tf)*100, np.std(roughness_tf)*100)
print(np.mean(roughness_pytorch)*100,np.std(roughness_pytorch)*100 )
print(wilcoxon(volume_tf, volume_pytorch))

plt.figure(figsize =(5,6))
ax = sns.violinplot(data = [roughness_tf, roughness_pytorch], 
            showfliers=False,color='lightgray', boxprops={'facecolor':'None'}, orient = "h")
ax = sns.swarmplot(data = [roughness_tf, roughness_pytorch],color=".25", size =1.5, orient = "h")
plt.xlabel('Roughness error [%]', size = 15)
plt.yticks([0,1],["TF", "pytorch"],size = 15)
plt.xticks(size = 15)
plt.xlim(-0.01,1.1)
plt.locator_params(axis='x', nbins=4)
plt.tight_layout()
plt.grid(b=None)
plt.grid(b=None)

plt.show()


# In[152]:


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


# In[153]:


plt.scatter(mask_area, volume, s = 1)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




