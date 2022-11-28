import os
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

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

ablation_count = 600
ab_folder = './cifar_ablation_sets/cifar_ablation' + str(ablation_count)
if not os.path.exists(ab_folder):
    os.mkdir(ab_folder)

synth_folder = './cifar_synth_sets/cifar_synth_1_2/'
real_folder = './cifar10_real/train/'

print('Creating empty class folders')
for c in tqdm(CLASSES):
    out_dir = os.path.join(ab_folder, c)
    if os.path.exists(out_dir):
        print(f"skipping split {c} since {out_dir} already exists.")
    else:
        os.mkdir(out_dir)

for i, c in enumerate(CLASSES):

    # synthetic images
    synth_read_f = os.path.join(synth_folder, c)
    set_write_f = os.path.join(ab_folder, c)
    read_f = os.path.join(real_folder, c)

    print('Class ', c)
    print("Number of real images:", len(os.listdir(read_f)) - ablation_count)
    print("Number of synthetic images:", min(ablation_count, len(os.listdir(synth_read_f))))

    j = 0
    for filename in tqdm(os.listdir(synth_read_f), total = ablation_count):
        read_img = os.path.join(synth_read_f, filename)
        image = plt.imread(read_img)

        write_f = os.path.join(set_write_f, filename)
        plt.imsave(write_f, image)
        j += 1
        if j >= ablation_count:
            break
    # real images
    k = 0
    for filename in tqdm(os.listdir(read_f), total=len(os.listdir(read_f)) - ablation_count):
        read_img = os.path.join(read_f, filename)
        image = plt.imread(read_img)

        write_f = os.path.join(set_write_f, filename)
        plt.imsave(write_f, image)
        k += 1
        if k >= len(os.listdir(read_f)) - ablation_count:
            break