import math, numpy as np, os
from PIL import Image
from rgb_basis_trans import *

def load_images(source_path, annotated_int = False, constant_int = 10, max_energy = 3, annotated_t = False, constant_t = 2):
    file_list = os.listdir(source_path) #load the list of image names in the source path
    try:
        file_list.remove(".DS_Store")
    except ValueError:
        pass
    try:
        file_list.remove("parameters.rtf")
    except ValueError:
        pass

    images = [np.asarray(Image.open(source_path + filename)) for filename in file_list[:]]
    if annotated_int:
        interaction_list = [concat_rgb_sum(image_array[1,14,:])*constant_int for image_array in images]
    else:
        interaction_list = [constant_int for i in range(len(images))]
    
    if annotated_t:
        time_list = [concat_rgb_sum(image_array[14,14,:])*constant_t for image_array in images]
    else:
        time_list = [constant_t for i in range(len(images))]

    image_list = [im[3:13,2:13,:] for im in images[:]] #convert the images to a list of numpy arrays and truncate the irrelevant pixels
    image_list_single_color = [np.array([concat_rgb_sum(image[i,j,:]) for i in range(10) for j in range(11)]).reshape((10,11)) for image in image_list] #convert each rgb pixel to a single colored pixel
    E = [image[:,0]*max_energy for image in image_list_single_color]
    corr = [2 *((image[:,1:])**2) for image in image_list_single_color]
    
    if annotated_t:
        return (corr, E, interaction_list, file_list, time_list)
    else:
        return (corr, E, interaction_list, file_list)
