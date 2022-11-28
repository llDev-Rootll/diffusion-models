import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import os

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

class convNet(nn.Module):
    def __init__(self):
        super(convNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.conv5=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.b1=nn.BatchNorm2d(16)
        self.b2=nn.BatchNorm2d(64)
        self.b3=nn.BatchNorm2d(256)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)  

        self.dropout=nn.Dropout(0.1)
        self.fc1=nn.Linear(256,128)
        self.fc2=nn.Linear(128,64)
        self.out=nn.Linear(64,10)

    def forward(self,x):
        x=self.pool(F.relu(self.b1(self.conv1(x))))
        x=self.pool(F.relu(self.conv2(x)))
        x=self.pool(F.relu(self.b2(self.conv3(x))))
        x=self.pool(F.relu(self.conv4(x)))
        x=self.pool(F.relu(self.b3(self.conv5(x))))
        x=x.view(-1,256)
        x = self.dropout(x)
        x=self.dropout(F.relu(self.fc1(x)))
        x=self.dropout(F.relu(self.fc2(x)))
        x=self.out(x)   
        return x

def test(model, test_loader):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval() # test the model with dropout layers off
    i = 0
    for images, labels in tqdm(test_loader):
        output = model(images)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(labels.data.view_as(pred)))

        for idx in range(len(labels)):
            # print()
            img1 = images[idx] 
            label = pred[idx]
            l = CLASSES[label]
            img_path = os.path.join(write_folder, l + "_u_{:04d}".format(i) + '.png')
            save_image(img1, img_path)
            class_correct[label] += correct[idx].item()
            class_total[label] += 1
            i += 1
    # print(f"Correctly predicted per class : {class_correct}, Total correctly perdicted : {sum(class_correct)}")
    print(f"Total Predictions per class : {class_total}, Total predictions to be made : {sum(class_total)}\n")


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 1000

folder = 'cifar_uncond2'
read_folder = os.path.join("./cifar_uncond_img", folder)
write_folder = os.path.join("./cifar_uncond_label", folder + '_label')
print("read_folder:", read_folder)
print("write_folder:", write_folder)

if not os.path.exists(write_folder):
    os.mkdir(write_folder)

print("Loading data...")
dataset_valid = datasets.ImageFolder(root = read_folder, transform = transform_test)
test_loader = DataLoader(dataset = dataset_valid, batch_size = batch_size)
print(test_loader)

print('Model extracted...')
model_2 = convNet()
model_2.load_state_dict(torch.load('./model/convNet_model.pth'))

print('Labelling...')
# print(model_2.state_dict)
test(model_2, test_loader)

