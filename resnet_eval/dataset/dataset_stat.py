import numpy as np
import matplotlib.pyplot as plt
import os

PATH = "./cifar_im_truck_ship_500"
ORDER = [7, 1, 0, 2, 3, 4, 5, 6, 8, 9]

def main():
    for split in ["train"]:
        out_dir = os.path.join(PATH, split)
        class_names = np.array(sorted(os.listdir(out_dir)))[ORDER]
        dist = {cls_name:len(os.listdir(os.path.join(out_dir, cls_name))) for cls_name in class_names}
        plt.figure(split)
        plt.bar(dist.keys(), dist.values())
    
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.savefig(os.path.join(os.path.join('./', '_dist.png')))
        plt.show()
        



if __name__ == "__main__":
    main()