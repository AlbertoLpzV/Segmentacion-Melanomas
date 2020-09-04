import project1 as project
import test_csv_for_kaggle as csv_test

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

#-----------------------------------------------------------------------------------------------------------------------------------------

#CARGAMOS LAS IMÁGENES Y MÁSCARAS DE TRAIN
data_dir= '../Segmentación Melanomas'

train_imgs_files = [os.path.join(data_dir,'train/images',f) for f in sorted(os.listdir(os.path.join(data_dir,'train/images'))) 
            if (os.path.isfile(os.path.join(data_dir,'train/images',f)) and f.endswith('.jpg'))]

train_masks_files = [os.path.join(data_dir,'train/masks',f) for f in sorted(os.listdir(os.path.join(data_dir,'train/masks'))) 
            if (os.path.isfile(os.path.join(data_dir,'train/masks',f)) and f.endswith('.png'))]

#Ordenamos para que cada imagen se corresponda con cada máscara
train_imgs_files.sort()
train_masks_files.sort()
print("Número de imágenes de train", len(train_imgs_files))
print("Número de máscaras de train", len(train_masks_files))

#CARGAMOS LAS IMÁGENES DE TEST
test_imgs_files = [os.path.join(data_dir,'test/images',f) for f in sorted(os.listdir(os.path.join(data_dir,'test/images'))) 
            if (os.path.isfile(os.path.join(data_dir,'test/images',f)) and f.endswith('.jpg'))]

test_imgs_files.sort()
print("Número de imágenes de test", len(test_imgs_files))

#-----------------------------------------------------------------------------------------------------------------------------------------

import copy

#1.Con el algoritmo creado en "skin_lesion_segmentation", 
#comprobamos qué tal es nuestra nota de segmentación mediante 

img_roots = train_imgs_files.copy()
gt_masks_roots = train_masks_files.copy()

mean_score = project.evaluate_masks(img_roots, gt_masks_roots)

#-----------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from skimage import io

images = io.ImageCollection(train_imgs_files)
masks = io.ImageCollection(train_masks_files)

index = 1
plt.figure(figsize=(15,8))
for i in range (4):
    plt.subplot(2,4,index)
    plt.imshow(images[i])
    index+=1
    plt.title("Image %i"%(i))
    plt.subplot(2,4,index)
    plt.imshow(masks[i], cmap='gray')
    index+=1
    plt.title("Mask %i"%(i))
    
#-----------------------------------------------------------------------------------------------------------------------------------------

#Una vez satisfechos con el resultado, generamos el fichero para hacer la submission en Kaggle
dir_images_name = '../Segmentación Melanomas/test/images'
csv_name='test_prediction_rgb2g_otsu_fill_holes.csv'
csv_test.test_prediction_csv(dir_images_name, csv_name)
