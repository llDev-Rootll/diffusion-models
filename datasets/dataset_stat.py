import numpy as np
import matplotlib.pyplot as plt
import os

path = "cifar_im"

def main():
    for split in ["train", "test"]:
        out_dir = os.path.join(path, f"cifar_im_{split}")
        img_names = os.listdir(out_dir)
        count = [name.split("_")[0] for name in img_names]
        classes, count = np.unique(np.array(count), return_counts=True)
        order = [7, 1, 0, 2, 3, 4, 5, 6, 8, 9]
        dist = dict(zip(classes[order], count[order]))
        plt.figure(split)
        plt.bar(dist.keys(), dist.values())
    
    plt.show()
        



if __name__ == "__main__":
    main()