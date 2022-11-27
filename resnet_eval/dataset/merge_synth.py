'''
replace r images with u
'''
import os
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import json

u_folder = './cifar_uncond_label/cifar_uncond1_label/'
r_folder = './cifar10_real/train/'
log_folder = './cifar_synth_sets/cifar_synth_2/'
new_folder = log_folder + 'train/'

CLASSES = np.array([
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
])

class_count = [0]*10

print('Creating empty class folders')
for c in tqdm(CLASSES):
    out_dir = os.path.join(new_folder, c)
    if os.path.exists(out_dir):
        print(f"skipping split {c} since {out_dir} already exists.")
    else:
        os.mkdir(out_dir)

print('Saving synthetic data in class folders')
for filename in tqdm(os.listdir(u_folder)):
    c = filename.split('_')
    idx = np.where(CLASSES == c[0])[0]
    class_count[idx[0]] += 1
    read_f = os.path.join(u_folder, filename)
    write_f = os.path.join(new_folder, c[0])
    write_f = os.path.join(write_f, filename)
    image = plt.imread(read_f)
    plt.imsave(write_f, image)

print('\nClasses:', CLASSES)
print("Total number of synthetic data per class:", class_count)
# class_count = [28, 28, 26, 21, 23, 22, 23, 33, 28, 21]
print('Saving synthtic data count...\n')
with open(os.path.join(log_folder, "synth_log.json"), "w") as outfile:
    json.dump(class_count, outfile)

print('Adding real data\n')
for i, c in enumerate(CLASSES):
    read_f = os.path.join(r_folder, c)
    # print(read_f)
    print('Class ', c)
    for filename in tqdm(os.listdir(read_f)):
        if(class_count[i] != 0):
            class_count[i] -= 1
            continue
        else:
            write_f = os.path.join(new_folder, c)
            write_f = os.path.join(write_f, filename)
            read_img = os.path.join(read_f, filename)
            image = plt.imread(read_img)
            plt.imsave(write_f, image)