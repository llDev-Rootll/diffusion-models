import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# Load the npz file
data = np.load('/home/dev_root/Desktop/Workspace/Fall22/828W/diffusion-models/cifar_unc_samples/samples_0_1024x32x32x3_20221128-000329.npz')
# data_2 = np.load('datasets/uncond_data/samples_1_1024x32x32x3_20221128-000v453.npz')

# Extract the images
images = data['arr_0']
# images_2 = data_2['arr_0']

# Create a directory to save the images
if not os.path.exists('datasets/uncond_data/images'):
    os.makedirs('datasets/uncond_data/images')

# Save the images

for i in range(len(images)):
    plt.imsave('datasets/uncond_data/images/image_{}.png'.format(i), images[i])

# for i in range(len(images_2)):
#     plt.imsave('datasets/uncond_data/images/image_{}.png'.format(i+len(images)), images_2[i])