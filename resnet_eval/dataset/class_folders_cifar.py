import os
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

CLASSES = (
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
)

main_folder = './cifar10/'
all_paths = ['train/', 'test/']
save_folder = "./cifar10_real/"

def main():
    # os.mkdir(save_folder)
    for _, split in enumerate(["test/", "train/"]):
        out_dir = save_folder + f"{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            # continue
        else:
            os.mkdir(out_dir)
        og_path = main_folder + split

        classes_dir = []
        for c in CLASSES:
            cl_dir = f"{out_dir}"+f"{c}"
            if os.path.exists(cl_dir):
                print(f"skipping split {c} since {cl_dir} already exists.")
                continue
            os.mkdir(cl_dir)
            classes_dir.append(cl_dir)
        
        print(split)
        for filename in tqdm(os.listdir(og_path)):
            c = filename.split('_')
            new_f = c[0] + '_r_' + c[1]
            save_f = out_dir + c[0]
            save_f = os.path.join(save_f, new_f)
            if os.path.exists(save_f):
                continue
            read_f = os.path.join(og_path, filename)
            image = plt.imread(read_f)
            plt.imsave(save_f, image)

if __name__ == "__main__":
    main()