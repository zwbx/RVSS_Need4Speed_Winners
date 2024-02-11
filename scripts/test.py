# %%
#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from IPython.display import Image
# import natsort
# %%cd co
import torch
print(torch.__version__)
print(torch.cuda.device_count())
print(torch.cuda.is_available())

# %%
import cv2
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# %%
dataroot = "data/t2"
ckptroot = "/Users/zhangwenbo/Documents/RVSS_Need4Speed/both-nvidia-model-80.h5"

lr = 1e-5
weight_decay = 1e-5
batch_size = 16
num_workers = 8
test_size = 0.8
shuffle = True

epochs = 80
start_epoch = 0
resume = False

# %%
test_root = '/Users/zhangwenbo/Documents/RVSS_Need4Speed/data/data/0000000.00.jpg' 

# %%
class NetworkNvidia(nn.Module):
    """NVIDIA model used in the paper."""

    def __init__(self):
        """Initialize NVIDIA model.

        NVIDIA model used
            Image normalization to avoid saturation and make gradients work better.
            Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Drop out (0.5)
            Fully connected: neurons: 100, activation: ELU
            Fully connected: neurons: 50, activation: ELU
            Fully connected: neurons: 10, activation: ELU
            Fully connected: neurons: 1 (output)

        the convolution layers are meant to handle feature engineering
        the fully connected layer for predicting the steering angle.
        """
        super(NetworkNvidia, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 32, 3),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=7392, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        """Forward pass."""
        # input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output


# Define model
print("==> Initialize model ...")
model = NetworkNvidia()
print("==> Initialize model done ...")

# %%
model = NetworkNvidia()

# %%
criterion = nn.MSELoss()

# %%
teslamodel = torch.load(ckptroot, map_location=lambda storage, loc: storage)
start_epoch = teslamodel['epoch']
model.load_state_dict(teslamodel['state_dict'])

# %%
model.eval()

# %%
def toDevice(datas, device):
    """Enable cuda."""
    imgs, angles = datas
    return imgs.float().to(device), angles.float().to(device)

def augment(dataroot, imgName, angle):
    """Data augmentation."""
    current_image = cv2.imread('/kaggle/input/selfdriving-car-simulator/track1data'+imgName)
    current_image = current_image[65:-25, :, :]
    return current_image, angle

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
num_correct = 0
total = 0
model.eval()
model.to(device)

def img_process(img):
    # print(img)
    # img = cv2.imread(img)
    img = img[130:, :, :]
    # img = cv2.resize(img, (320, 70))
    # img = img[20:70, :]
    img = img / 255.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)
    return img

def test_model(img):
    with torch.no_grad():
        imgs = img_process(img).to(device).to(torch.float32)
        outputs = model(imgs)
        return round(outputs.item(),3)

# print(test_model(test_root))
# %% [markdown]
# # Calculating Test Accuracy Using all the 3 Cameras
