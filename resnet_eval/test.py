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
# PATH = "./model/resnet18_.pt"

# model = torch.load(PATH)
# # model.eval()
# y = model(torch.randn(1, 3, 32, 32))
# print(y.size())
# print(model)

# import torch
# from pprint import pprint
# pprint(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))
# model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a2", pretrained=True)

# model = torch.load("./model/cifar10_repvgg_a2.pt")
# print(model)

# model.eval()
# y = model(torch.randn(1, 3, 32, 32))
# print(y.size())
# import pytest

# @pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
# @pytest.mark.parametrize("model_name", ["repvgg_a0", "repvgg_a1", "repvgg_a2"])

# num_classes = 10 if dataset == "cifar10" else 100
# model = getattr(repvgg, "cifar10_repvgg_a2")()
model = torch.load("./model/cifar10_repvgg_a2.pt")
model.load_state_dict(torch.load("./model/cifar10_repvgg_a2.pt"))

from torchsummary import summary
summary(model, (1, 3, 32, 32))
print(model)
x = torch.empty((1, 3, 32, 32))
y = model(x)
assert y.shape == (1, 10)