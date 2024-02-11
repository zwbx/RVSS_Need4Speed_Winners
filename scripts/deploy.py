#!/usr/bin/env python3
import time
import click
import math
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
# from .test import test_model
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

ckptroot = "/Users/zhangwenbo/Documents/RVSS_Need4Speed/tta_0209_aug_align.h5"


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
        # print(device)
        outputs = model(imgs)
        return round(outputs.item(),3)

def adjust_velocity(angle):
    """
    Adjusts Kd and Ka values smoothly based on the steering angle.
    
    :param angle: The predicted steering angle.
    :return: Tuple of (Kd, Ka) representing the base wheel speed and turning speed.
    """
    # Define the thresholds and corresponding Kd and Ka values
    angle = np.clip(angle, -0.4, 0.4)
    
    # Interpolate Kd and Ka values based on the factor
    Kd = 50 - abs(angle)/0.2 * 25
    Kd = np.clip(Kd, 20, 40)
    Ka = 2* Kd
    
    return Kd, Ka


parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--num', type=int, default='0', help='number of debug folder')
args = parser.parse_args()

bot = PiBot(ip=args.ip)

# stop the robot 
bot.setVelocity(0, 0)

#INITIALISE NETWORK HERE

#LOAD NETWORK WEIGHTS HERE
im_number=0
#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

try:
        # 定义红色的两个HSV范围
    lower_red1 = np.array([0, 160, 140])
    upper_red1 = np.array([5, 220, 210])
    lower_red2 = np.array([170, 160, 140])
    upper_red2 = np.array([180, 220, 210])
    angle = 0
    e= 0
    s=0
    left = 0
    right = 0
    Kd = 20
    Ka = 30
    folder_name = "debug-0209-0"
    angles = []  # 存储过去角度值的列表
    latency = []
    flag= 0
    # N = 2  # 滑动窗口的大小，可以根据需要调整
    # if os.path.exists(script_path+"/../data/"+folder_name) == False:
    #     os.makedirs(script_path+"/../data/"+folder_name)
    # else:
    #     folder_name = folder_name[:-1]+str(int(folder_name[-1:])+1)+str(args.num)
    #     os.makedirs(script_path+"/../data/"+folder_name)
    while True:
        # bot.setVelocity(int(0/2), int(0/2))
        # get an image from the the robot
        
        # while e > 0.05:
        e = time.time() 
        lat = e-s
        if im_number > 0:
            latency.append(float(lat))
        # print(lat,len(latency),sum(latency))
            
            print("latency_mean: ",sum(latency)/len(latency))
        
        img = bot.getImage()
        s = time.time() 
        
        angle = test_model(img) 
        

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 根据定义的颜色范围创建两个掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # 合并掩码
        mask = cv2.bitwise_or(mask1, mask2)

        # 将掩码和原图进行位运算，只显示红色区域
        result = cv2.bitwise_and(img, img, mask=mask)
        red = np.sum(mask)
        if red>300000 and flag == 0:
            print("stop")
            bot.setVelocity(0, 0)
            flag = 10
        flag-=1
        # 添加当前角度到列表，并确保列表只保存最近N个值
        # angles.append(angle)
        # angles = angles[-N:]

        # 计算滑动平均角度
        # smoothed_angle = sum(angles) / len(angles)
        # print("angle:",smoothed_angle)
        # cv2.imwrite(script_path+"/../data/"+folder_name+"/"+str(im_number).zfill(6)+'_%.2f'%angle+'_%.3f'%lat+".jpg", img) 
        #TO DO: convert prediction into a meaningful steering angle
        print(angle)
        im_number += 1
     
        Kd, Ka = adjust_velocity(angle)

        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
        
        
        # print(e-s)
        bot.setVelocity(left, right)
        
        # duration = 0.05  # 指定的持续时间，单位为秒
        # time.sleep(duration)  # 暂停执行指定时间

        # # 如果需要在指定时间后停止机器人，可以将速度设置为0
        # bot.setVelocity(0, 0)  # 停止机器人
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
