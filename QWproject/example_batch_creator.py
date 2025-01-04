import math, numpy as np, os, scipy as sp
from matplotlib import pyplot as plt
import PIL
from TwoParticlesQW import *
from rgb_basis_trans import *
from datetime import datetime
"""
the function "create_standard_example_batch" creates examples of 2 particles on 10-site lattice.
the resulted images are 16X16 images 8bit rgb png images.
"rgb_trans_function" is a function that converts a float between [0,1] into a 3-slot rgb array, each between [0,255]
"""

def test_plot(image):
    plt.imshow(image)
    plt.show()

def transform_to_rgb(image_data, rgb_trans_func): #image_data is a 16X16 np array
    im_size = len(image_data)
    final_image = np.zeros((im_size,im_size,3)).astype(np.uint8)
    for i in range(im_size):
        for j in range(im_size):
            final_image[i,j,:] = rgb_trans_func(image_data[i,j])
    return final_image


def create_standard_example_batch(N_examples = 1, seed = 1, E_range = (0,3), Jnm = np.ones((10,10)), gamma_range = [0,10], time_range = [2,2], psi0 = [0]*35 + [1] + [0]*19, save_path="", rgb_trans_func = [[base256_to_rgb, "base256"], [base_sum_to_rgb, "base_sum"]], annotate_gamma = True, annotate_time = True, save_as_array = False):
    np.random.seed(seed)
    start_time = datetime.now()
    corr_data_array = []
    time_data_array = []
    interaction_data_array = []

    if save_as_array: #to avoid division by 0 when normalized
        if time_range[1] == time_range[0]:
            time_div_range = [0, time_range[1]]
        else:
            time_div_range = time_range
        if gamma_range[1] == gamma_range[0]:
            gamma_div_range = [0, gamma_range[1]]
        else:
            gamma_div_range = gamma_range
        


    for i in range(N_examples):
        gamma = np.random.uniform(gamma_range[0], gamma_range[1])
        prop_time = np.random.uniform(time_range[0], time_range[1])
        En = np.random.uniform(E_range[0], E_range[1], 10) #generate the random energies. note that np.random.uniform(a,b) includes a but excludes b
        corr = initial_condition_to_corr(N_sites = 10, En = En, Jnm = Jnm, gamma = gamma, psi0 = psi0, prop_time = prop_time, plot_result=False, plot_path="") # find the correlation
        image_data = np.zeros((16,16)) # create empty np array
        image_data[3:13, 2] = np.array(En).copy()/E_range[1] # insert the energy data, normalized to 1
        image_data[3:13, 3:13] = (corr.copy()/2)**0.5 # insert the correlation data, normalized with the number of particles *without square root*
        
        if annotate_gamma:
            image_data[1,14] = gamma/gamma_range[1]
        if annotate_time:
            image_data[14,14] = prop_time/time_range[1]
        
        if save_as_array: 
            corr_data_array.append(image_data)
            time_data_array.append((prop_time - time_div_range[0])/(time_div_range[1]-time_div_range[0]))
            interaction_data_array.append((gamma - gamma_div_range[0])/(gamma_div_range[1]-gamma_div_range[0]))
        
        for func in rgb_trans_func:
            rgb_image = transform_to_rgb(image_data, func[0]) #transform to rgb image
            PIL.Image.fromarray(rgb_image).save(save_path + "/" + func[1] + "/" + str(i) + ".png") # save image
        
        if i%500 == 0: # monitor the progress
            now = datetime.now()
            print("step: " + str(i) + ", time took: " + str(now-start_time))
    if save_as_array:
        corr_data_np_file = np.stack(corr_data_array, axis = 0)
        with open(save_path+"corr_data.npy", "wb") as f:
            np.save(f, corr_data_np_file)
        with open(save_path+"time_data.npy", "wb") as f:
            np.save(f, np.array(time_data_array)[:, None])
        with open(save_path+"interaction_data.npy", "wb") as f:
            np.save(f, np.array(interaction_data_array)[:, None])


    now = datetime.now()
    print("total time took: " + str(now - start_time))


create_standard_example_batch(N_examples=100000, save_path=save_path, annotate_gamma=False, annotate_time = False, gamma_range = [3,3], time_range = [2, 2], rgb_trans_func = [], save_as_array = True)
print("data creation completed, saved to:")
print(save_path)



