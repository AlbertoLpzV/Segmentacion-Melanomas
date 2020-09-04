#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %      
% TDImagen PROYECTO 1: SEGMENTACIÓN DE IMÁGENES                         %
% Creación de CSV para subir la solución a Kaggle                       %
%                                                                       %
% --------------------------------------------------------------------- %
%                                                                       %
% Created on Wed Feb 26 16:37:41 2020                                   %
% @author: Miguel-Angel Fernandez-Torres                                %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import numpy as np
import os, csv
from skimage import io
import skimage.color
from rle import rle_encode
from project1 import skin_lesion_segmentation


def test_prediction_csv(dir_images_name='database/test/images', csv_name='test_prediction.csv'):
    dir_images = np.sort(os.listdir(dir_images_name))
    mask_otsu_all = []
    mask_aux_all = []
    score = []
    with open(csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImageId", "EncodedPixels"])
        for i in np.arange(np.size(dir_images)):        
            # - - - Llamada a la función 'skin_lesion_segmentation'
            # Implementa el método propuesto para la segmentación de la lesión
            # y proporciona a su salida la máscara predicha.
            predicted_mask = skin_lesion_segmentation(dir_images_name+'/'+dir_images[i])
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            
            # - - - Codificación RLE y escritura en fichero .csv
            encoded_pixels = rle_encode(predicted_mask)
            writer.writerow([dir_images[i][:-4], encoded_pixels])
            print('Máscara '+str(i)+' codificada.')
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -