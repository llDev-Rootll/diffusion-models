filepath = "./samples_128x64x64x3.npz"

from pydoc import classname
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

    
def extract_npz(filepath):
    data = np.load(filepath)
    images = data['arr_0']
    labels = data['arr_1']
    return images, labels

def save_images(images, num, label, folder):
    for i in tqdm(range(len(images))):
        # i += num
        num += 1
        image = images[i]
       
        filename = folder + str(label)+ "_" + str(num) + ".png"
        # plt.imshow(image)
        # plt.show()
        plt.imsave(filename, image)
    print(num)
    return num
        
class_name = 'ship'
directory = './' + class_name
new_folder = './' + class_name + '_new'
i = 0
if os.path.exists(new_folder):
    # If the new class folder exists already then you cant add more images to it.
    print(class_name,'_new folder exists!')
    exit()

    # # If you want to append to the new dataset samples in the class folder
    # for img in sorted(os.listdir(new_folder), reverse=True):
    #     print(img)
    #     i = int(img[len(class_name)+1:len(img)-4]) + 1
    #     break
else:
    os.mkdir(class_name+'_new')

for file in os.listdir(directory):
    filepath = directory+'/'+file
    print(filepath)
    images, labels = extract_npz(filepath)
    
    i = save_images(images, i, class_name, "./" + class_name + "_new/")
