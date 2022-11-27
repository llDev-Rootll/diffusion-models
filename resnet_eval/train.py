# from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import argparse
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from resnet18 import ResNet, BasicBlock
from resnet18_torchvision import build_model
from training_utils import train, validate
from utils import *
import time


parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='scratch',
    help='choose model built from scratch or the Torchvision model',
    choices=['scratch', 'torchvision']
)
args = vars(parser.parse_args())

# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

f = open("./resnet_eval/train_configs/initial_config.json")
train_params = json.load(f)

# Learning and training parameters.
epochs = train_params["epochs"]
batch_size = train_params["batch_size"]
learning_rate = train_params["learning_rate"]
DATASET_PATH = train_params["dataset_path"]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_transforms = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])

train_set_path = os.path.join(DATASET_PATH, "train")
val_set_path = os.path.join(DATASET_PATH, "test")

train_loader, valid_loader, class_to_idx = load_data_set(batch_size=batch_size, train_data_dir=train_set_path, valid_data_dir=val_set_path, transforms=train_transforms)

# Define model based on the argument parser string.
if args['model'] == 'scratch':
    print('[INFO]: Training ResNet18 built from scratch...')
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
    plot_name = 'resnet_scratch'
if args['model'] == 'torchvision':
    print('[INFO]: Training the Torchvision ResNet18 model...')
    model = build_model(pretrained=False, fine_tune=True, num_classes=10).to(device) 
    plot_name = 'resnet_torchvision'
# print(model)

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Optimizer.
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# Loss function.
criterion = nn.CrossEntropyLoss()
best_model = model
best_val = 0
if __name__ == '__main__':
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    # epochs = 1
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, 
            train_loader, 
            optimizer, 
            criterion,
            device
        )
        valid_epoch_loss, valid_epoch_acc, per_class_accuracy, confusion_matrix = validate(
            model, 
            valid_loader, 
            criterion,
            device
        )
        print(per_class_accuracy)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

        if(valid_epoch_acc >= best_val):
            print("##### Saving Model #####")
            best_val = valid_epoch_acc
            best_model = model
        print('-'*50)
        

    accuracy = {'train_acc':train_epoch_acc, 'eval_acc':valid_epoch_acc}
    loss = {'train_loss':train_epoch_loss, 'eval_loss':valid_epoch_loss}

    accuracy_ep = {'train_acc_per_ep':train_acc, 'eval_acc_per_ep':valid_acc}
    loss_ep = {'train_loss_per_ep':train_loss, 'eval_loss_per_ep':valid_loss}
    # Save the loss and accuracy plots.
    # save_plots(
    #     train_acc, 
    #     valid_acc, 
    #     train_loss, 
    #     valid_loss, 
    #     name=plot_name
    # )
    TIMESTR = time.strftime("%Y%m%d-%H%M%S")
    ORDER = [7, 1, 0, 2, 3, 4, 5, 6, 8, 9]
    DATASET_NAME = DATASET_PATH.split("/")[-1]
    
    per_class_accuracy_w_names = {cls_name:val.cpu().numpy().item() for cls_name, val in zip(class_to_idx, per_class_accuracy)}
    print(per_class_accuracy_w_names)
    print('TRAINING COMPLETE')
    results_name = f"results_{DATASET_NAME}_{TIMESTR}"
    results_path = os.path.join(train_params["results_path"], results_name+".json")
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        with open(os.path.join(results_path, results_name + ".json"), "w") as outfile:
            json.dump(per_class_accuracy_w_names, outfile)
        
        with open(os.path.join(results_path, "config.json"), "w") as outfile:
            json.dump(train_params, outfile)

        torch.save(best_model.state_dict(), os.path.join(results_path, "model.pt"))

        plt.bar(np.array(list(per_class_accuracy_w_names.keys()))[ORDER], np.array(list(per_class_accuracy_w_names.values()))[ORDER])
        plt.title(DATASET_NAME)
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(os.path.join(results_path, results_name+'_acc.png')))
        np.save(os.path.join(os.path.join(results_path, "confusion_matrix.npy")), confusion_matrix)

        class_names = np.array(sorted(os.listdir(train_set_path)))[ORDER]
        dist = {cls_name:len(os.listdir(os.path.join(train_set_path, cls_name))) for cls_name in class_names}
        plt.figure("Training Set")
        plt.bar(dist.keys(), dist.values(), color=['blue'])
    
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.savefig(os.path.join(os.path.join(results_path, results_name + '_dist.png')))
        plt.figure("ACC")
        # print(valid_acc)
        plt.plot(valid_acc)
        
        plt.show()