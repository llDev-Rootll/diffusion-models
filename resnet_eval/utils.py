import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

plt.style.use('ggplot')


def load_data_set(batch_size, train_data_dir, valid_data_dir, transforms):
    """Load generic/syth datasets 
    """

    dataset_train = datasets.ImageFolder(root=train_data_dir, transform=transforms)
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)

    dataset_valid = datasets.ImageFolder(root=valid_data_dir, transform=transforms)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=True)

    assert(dataset_train.class_to_idx==dataset_valid.class_to_idx)
    
    return train_loader, valid_loader, dataset_train.class_to_idx



def get_original_cifar_data(batch_size=64):

    
    # CIFAR10 training dataset.
    dataset_train = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # CIFAR10 validation dataset.
    dataset_valid = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_valid, 
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, valid_loader

def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_loss.png'))