from pydoc import classname
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

def extract_npz(filepath):
    data = np.load(filepath)
    images = data['arr_0']
    # labels = data['arr_1']
    return images

def save_images(images, num, label, folder):
    for i in tqdm(range(len(images))):
        # i += num
        num += 1
        if(num > 500):
            break
        image = images[i]
        filename = os.path.join(folder, str(label)+ "_" + str(num) + ".png")

        # filename = folder + str(label)+ "_" + str(num) + ".png"
        # plt.imshow(image)
        # plt.show()
        plt.imsave(filename, image)
    print(num)
    return num
        
class_name = 'cifar_uncond1'
directory = './cifar_npy_fold/' + class_name
new_folder = './cifar_uncond_img/' + class_name
i = 0
if os.path.exists(new_folder):
    # If the new class folder exists already then you cant add more images to it.
    print(new_folder,' folder exists!')
    # exit()

    # # If you want to append to the new dataset samples in the class folder
    # for img in sorted(os.listdir(new_folder), reverse=True):
    #     print(img)
    #     i = int(img[len(class_name)+1:len(img)-4]) + 1
    #     break
else:
    os.mkdir(new_folder)
# print()
for file in os.listdir(directory):
    # filepath = directory+'/'+file
    filepath = os.path.join(directory, file)
    print(filepath)
    images = extract_npz(filepath)
    
    i = save_images(images, i, class_name, new_folder)
    if(i > 500):
        break
