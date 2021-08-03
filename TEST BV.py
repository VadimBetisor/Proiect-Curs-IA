#Importing all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from torch import optim

# Importing libraries related to neurala networks
import torch
import torch.nn as nn
import os
from os import listdir
from os.path import isfile, join
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
from math import e
import torch.nn.functional as F
import itertools
import random
from torchvision import datasets, transforms

# Defining the Dataset
class SNN_Dataset(data.Dataset):
    def __init__(self, root, ME_DIR, NOT_ME_DIR):
        self.root = root
        ME_PATHS = listdir(ME_DIR)
        NOT_ME_PATHS = listdir(NOT_ME_DIR)
        ME_PATHS = [join(ME_DIR, img_path) for img_path in ME_PATHS]
        NOT_ME_PATHS = [join(NOT_ME_DIR, img_path) for img_path in NOT_ME_PATHS]
        ME_PAIRS = list(itertools.permutations(ME_PATHS, 2))
        self.TRIPLETS = list(itertools.product(ME_PAIRS, NOT_ME_PATHS))
        self.TRIPLETS = [list(triplet) for triplet in self.TRIPLETS]
        for i in range(len(self.TRIPLETS)):
            self.TRIPLETS[i] = list(self.TRIPLETS[i])
            self.TRIPLETS[i][0] = list(self.TRIPLETS[i][0])
            self.TRIPLETS[i][0].append(self.TRIPLETS[i][1])
            self.TRIPLETS[i].pop(-1)
            self.TRIPLETS[i] = self.TRIPLETS[i][0]
        random.shuffle(self.TRIPLETS)
    def __getitem__(self, index):
        img_triplet = [Image.open(img).convert('RGB') for img in self.TRIPLETS[index]]
        img_triplet = [transforms.Scale((244, 244))(img) for img in img_triplet]
        img_triplet = [transforms.ToTensor()(img) for img in img_triplet]
        return img_triplet[0], img_triplet[1], img_triplet[2]
    def __len__(self):
        return self.TRIPLETS

snn_dataset = SNN_Dataset(r'C:\Users\user\Desktop\SNN-main\photos',r'C:\Users\user\Desktop\SNN-main\photos\ME', r'C:\Users\user\Desktop\SNN-main\photos\NOT_ME')

# Creating the data laoder.
snn_loader = torch.utils.data.DataLoader(
    snn_dataset,
    batch_size=1,
    num_workers=2
)

# Creating the CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=1)
        self.conv1_dropuot = nn.Dropout2d(0.5)
        self.comv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_dropuot = nn.Dropout2d(0.5)
        self.maxpool2 = nn.MaxPool2d(3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_dropout = nn.Dropout2d(0.5)
        self.maxpool3 = nn.MaxPool2d(3, stride=1)
        self.linear1 = nn.Linear(3625216, 32)
        self.linear2 = nn.Linear(32, 16)
    def forward(self, x):
        out = self.conv1_dropuot(self.maxpool1(self.conv1(x)))
        out = self.conv2_dropuot(self.maxpool2(self.comv2(out)))
        out = self.conv3_dropout(self.maxpool3(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out

# Creatting the model
model = CNN()

def loos(tested_out, known_out, non_obj_out, alpha):
    ''' This function is calcultating the loss
        :param tested_out: tensor
            The linear representation of the tested image
        :param known_out: tensor
            The linear representation of the knwon image
        :param non_obj_out: tensor
            The linear representation of the random image
        :param alpha: float
            The senzivity parameter
    '''
    norm1 = torch.norm(tested_out - known_out, p=2)
    norm2 = torch.norm(tested_out - non_obj_out, p=2)
    return max(norm1 - norm2 + alpha, torch.zeros(1, requires_grad=True))
# Definign the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.000001, momentum=0.5)

# Training process
for epoch in range(10):
    print("Epoch {}".format(epoch))
    train_loss = 0.0
    for triplet in snn_dataset:
        out1 = model(triplet[0].unsqueeze(1).permute(1, 0, 2, 3))
        out2 = model(triplet[1].unsqueeze(1).permute(1, 0, 2, 3))
        out3 = model(triplet[2].unsqueeze(1).permute(1, 0, 2, 3))
        loss = loos(out1, out2, out3, alpha=0.5)
        loss.backward()
        optimizer.step()
        print(loss)

Epoch 0
tensor(0.5622, grad_fn=<AddBackward0>)
tensor(0.4224, grad_fn=<AddBackward0>)
tensor(0.3875, grad_fn=<AddBackward0>)
tensor(0.6228, grad_fn=<AddBackward0>)
tensor(0.1484, grad_fn=<AddBackward0>)
tensor(0.7787, grad_fn=<AddBackward0>)
Epoch 1
tensor(0.7224, grad_fn=<AddBackward0>)
tensor(0.4561, grad_fn=<AddBackward0>)
tensor(0.4819, grad_fn=<AddBackward0>)
tensor(0.1999, grad_fn=<AddBackward0>)
tensor(1.7736, grad_fn=<AddBackward0>)
tensor(1.1796, grad_fn=<AddBackward0>)
Epoch 2
tensor(2.9611, grad_fn=<AddBackward0>)
tensor(2.2283, grad_fn=<AddBackward0>)
tensor(3.4101, grad_fn=<AddBackward0>)
tensor([0.], requires_grad=True)
tensor([0.], requires_grad=True)
tensor(1.5980, grad_fn=<AddBackward0>)
Epoch 3
tensor(3.2073, grad_fn=<AddBackward0>)
tensor(3.6162, grad_fn=<AddBackward0>)
tensor([0.], requires_grad=True)
tensor(6.8160, grad_fn=<AddBackward0>)
tensor(4.7230, grad_fn=<AddBackward0>)
tensor(1.2789, grad_fn=<AddBackward0>)
Epoch 4
tensor(0.9604, grad_fn=<AddBackward0>)
tensor(3.4249, grad_fn=<AddBackward0>)
tensor([0.], requires_grad=True)
tensor(1.5595, grad_fn=<AddBackward0>)
tensor(2.3176, grad_fn=<AddBackward0>)
tensor(11.1287, grad_fn=<AddBackward0>)
Epoch 5
tensor(9.8319, grad_fn=<AddBackward0>)
tensor(2.0458, grad_fn=<AddBackward0>)
tensor(3.7131, grad_fn=<AddBackward0>)
tensor(2.5915, grad_fn=<AddBackward0>)
tensor(2.2747, grad_fn=<AddBackward0>)
tensor(2.4474, grad_fn=<AddBackward0>)
Epoch 6
tensor([0.], requires_grad=True)
tensor(0.7385, grad_fn=<AddBackward0>)
tensor(10.9662, grad_fn=<AddBackward0>)
tensor(8.5499, grad_fn=<AddBackward0>)
tensor([0.], requires_grad=True)
tensor([0.], requires_grad=True)
Epoch 7
tensor(5.9020, grad_fn=<AddBackward0>)
tensor([0.], requires_grad=True)
tensor(0.5641, grad_fn=<AddBackward0>)
tensor([0.], requires_grad=True)
tensor([0.], requires_grad=True)
tensor(10.7903, grad_fn=<AddBackward0>)
Epoch 8
tensor([0.], requires_grad=True)
tensor([0.], requires_grad=True)
tensor([0.], requires_grad=True)
tensor(8.7582, grad_fn=<AddBackward0>)
tensor([0.], requires_grad=True)
tensor([0.], requires_grad=True)
Epoch 9
tensor(5.6105, grad_fn=<AddBackward0>)
tensor(3.9385, grad_fn=<AddBackward0>)
tensor(4.0335, grad_fn=<AddBackward0>)
tensor(7.8433, grad_fn=<AddBackward0>)
tensor(4.4903, grad_fn=<AddBackward0>)
tensor([0.], requires_grad=True)

# Saving the model
torch.save(model, 'VADIM.pt')
