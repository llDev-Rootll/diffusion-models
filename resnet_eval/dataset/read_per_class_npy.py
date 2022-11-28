from pydoc import classname
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
append = True
def extract_npz(filepath):
    data = np.load(filepath)
    images = data['arr_0']
    # labels = data['arr_1']
    return images

def save_images(images, num, label, folder):
    for i in tqdm(range(len(images))):
        num += 1
        image = images[i]
        filename = os.path.join(folder, str(label)+ "_" + str(num) + ".png")
        # plt.imshow(image)
        # plt.show()
        plt.imsave(filename, image)
    return num
        
class_name = 'cifar_uncond3'
directory = './cifar_npy_fold/' + class_name
new_folder = './cifar_uncond_img/' + class_name
i = 0

if os.path.exists(new_folder):
    # If the new class folder exists already then you cant add more images to it.
    print(new_folder,' folder exists!')
else:
    os.mkdir(new_folder)

write_folder = os.path.join(new_folder, 'c/')
if not os.path.exists(write_folder):
    os.mkdir(write_folder)

if append == True:
    # If you want to append to the new dataset samples in the class folder
    i = len(os.listdir(write_folder)) + 1

print('starting index = ', i)

for file in os.listdir(directory):
    filepath = os.path.join(directory, file)
    images = extract_npz(filepath)
    i = save_images(images, i, class_name, write_folder)

print('Total number of images:', i)