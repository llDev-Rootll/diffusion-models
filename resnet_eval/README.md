
# Resnet

## Train on imbalanced dataset

Download [the per class dataset from here](https://drive.google.com/file/d/1iyuSEB6mHFu80IzDtJAJrm8KoaGJ3kQ8/view?usp=sharing)

    python train.py

## Extract and Dump new samples

Extract samples from .npy files saved in [here](https://drive.google.com/drive/folders/1g_4HOfbVTXnRxQabOmv_pxNkgLghzT8u?usp=sharing). The .npy files are saved segregated according to classes.

 1. Download the entire class folder with the .npy folders
 2. Extract and move the class folder to the `read_per_class_npy.py` directory. The name of the folder should be the name of the class. Eg, ship, truck, ...
 3. Change the name of the class inside of `read_per_class_npy.py`.
 4. Run `python read_per_class_npy.py` to extract the samples and to dump them in a new folder as .png files.

Ideally, we do this once a satisfactory amount of samples have been generated for a particular class so that we can run this file just once per class to avoid clashes in numbering the images while dumping.

Dataset link: [here](https://drive.google.com/drive/folders/1D6IXzvn9Yg6L3dPGfw91gHP8vLa5ELRK?usp=sharing)

These new samples will get added to the train folder in our [per_class_data](https://drive.google.com/file/d/1iyuSEB6mHFu80IzDtJAJrm8KoaGJ3kQ8/view?usp=sharing).

## Running training using config and seeing experiment result folders

Run training as before. But training params can be set by setting `train_config/initial_config.json`. Outputs for each experiment show up as a `results_*.json`, plot of per class accuracy and a `config.json` which shows the config that training job was run with. Each successful experiment creates a unique results directory.

