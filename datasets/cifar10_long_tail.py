import os
import torch
import torchvision.transforms as T
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from tfrecord.torch.dataset import TFRecordDataset

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

def main():
    tfrecord_paths = ["./data/cifar-10-data-im-0.1/train.tfrecords","./data/cifar-10-data-im-0.1/eval.tfrecords"]
    for i, split in enumerate(["train", "eval"]):
        out_dir = f"./cifar_im_0.1_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue
        os.mkdir(out_dir)

        tfrecord_path = tfrecord_paths[i]
        index_path = None
        description = {"image": "byte", "label": "int"}
        dataset = TFRecordDataset(tfrecord_path, index_path, description)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)

        k = 0
        transform = T.ToPILImage()

        for _, data in tqdm(enumerate(loader)):
            batch_imgs = data['image']
            batch_lables = data['label']

            # iterate over 32 items: batch size
            for i, l in enumerate(batch_lables):
                filename = os.path.join(out_dir, f"{CLASSES[l]}_{k:05d}.png")
                cur_img = batch_imgs[i]
                cur_img = cur_img.view(3, 32, 32)
                img = transform(cur_img)
                # img.show()
                im1 = img.save(filename)
                k += 1

                # n_img = cur_img.permute(1, 2, 0)
                # plt.imshow(n_img)
                # plt.show()

if __name__ == "__main__":
    main()
