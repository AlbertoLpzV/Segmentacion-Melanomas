#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage import io, color, draw, measure, filters
from skimage import feature, segmentation
from skimage.exposure import cumulative_distribution
from skimage import exposure, morphology 
from scipy import ndimage
from skimage.morphology import square
from sklearn.cluster import KMeans
from skimage.segmentation import random_walker
from skimage.transform import rescale
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.metrics import jaccard_score



def skin_lesion_segmentation(img_root):
    """ SKIN_LESION_SEGMENTATION: ... 
    """
    image = imread(img_root)    
    
    line_y = len(image[0])
    line_x = len(image)

    image_gray = color.rgb2gray(image)
    image_gray = image_gray * 255

    pelos_gray = morphology.black_tophat(image_gray, np.ones((9,9)))  # Detectamos los pelos
    gray_sin_pelo = image_gray + pelos_gray       # Eliminamos los pelos

    image_gauss = ndimage.median_filter(gray_sin_pelo, 18)  # 35

    area_min = 500 
    if (image_gauss[0,0]<30) and (image_gauss[0,line_y-1]<30) and  (image_gauss[line_x-1,0]<30) and  (image_gauss[line_x-1,line_y-1]<30) and  (image_gauss[line_x-10,0]<30):
     
        if (image_gauss[500,30]<30) and (image_gauss[150,150]<30) and (image_gauss[(line_x-10),500]<30):
            radio = line_x/2 - 60
        
        elif (image_gauss[500,30]<30) and (image_gauss[150,150]<30):
            radio = line_x/2 - 30
        
        elif (image_gauss[0,500]<30):
            radio = line_x/2 - 10
        
        else :
            radio = line_x/2
        
        unos = np.ones((line_x, line_y), dtype=np.uint8)
        unos = unos * 255
        rr, cc = draw.circle(line_x/2, line_y/2, radio)
        unos[rr, cc] = 0
        image_gauss = unos + image_gauss
    
        for i in range(len(image_gauss)):
            for j in range(len(image_gauss[i])):
                if image_gauss[i,j]>255:
                    image_gauss[i,j] = 255
     
           
        image_gauss = image_gauss - ((unos/255)*50)           
 
    mascara =  image_gauss   

    p19, p89 = np.percentile(mascara, (19,89))
    img_rescale = exposure.rescale_intensity(mascara, in_range=(p19,p89))
    img_rescale = img_rescale * 255


####### SEGMENTACION

    otsu_th = filters.threshold_otsu(img_rescale)  #  UMBRAL OTSU
    mask_otsu = (img_rescale < otsu_th)


####### POST-PROCESADO

    opening = morphology.opening(mask_otsu, np.ones((33,33)))
    close = morphology.closing(opening, np.ones((30,30)))

    label_img= morphology.label(close, connectivity=2, background=0)            #  LABEL (para elegir label central)
    medidas = measure.regionprops(label_img)
    numlabel = len(medidas)

    lesion = []
    for i in range(numlabel) :
    
        area = medidas[i].area
        xx,yy = medidas[i].centroid
    
        if ((400 <= yy < 650) and (250 <= xx < 520) and (area>area_min)):

            lesion.append(label_img==i+1)
        
    if (len(lesion)==0):
        for i in range(numlabel) :
    
            area = medidas[i].area
            xx,yy = medidas[i].centroid
            
            if ((50 <= yy < 950) and (50 <= xx < 750) and (area>area_min)):   
            
                lesion.append(label_img==i+1)

    
    melanoma = lesion.pop()  
    melanoma = melanoma*1
                  
    dilata = morphology.dilation(melanoma, np.ones((7,7)))
    post_predicted_mask = morphology.convex_hull_image(dilata, tolerance=50)


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    

    
    return post_predicted_mask
	
def evaluate_masks(img_roots, gt_masks_roots):
    """ EVALUATE_MASKS: Función que, dadas dos listas, una con las rutas
        a las imágenes a evaluar y otra con sus máscaras Ground-Truth (GT)
        correspondientes, determina el Mean Average Precision a diferentes
        umbrales de Intersección sobre Unión (IoU) para todo el conjunto de
        imágenes.
    """
    score = []
    for i in np.arange(np.size(img_roots)):
        predicted_mask = skin_lesion_segmentation(img_roots[i])
        gt_mask = io.imread(gt_masks_roots[i])/255     
        score.append(jaccard_score(np.ndarray.flatten(gt_mask),np.ndarray.flatten(predicted_mask)))
    mean_score = np.mean(score)
    print('Jaccard Score sobre el conjunto de imágenes proporcionado: '+str(mean_score))
    return mean_score