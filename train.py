# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#add the path to the sys
import sys
sys.path.append('/home/wenbo/Documents/data/Automating-Car-Steering')
import natsort

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("data/t2"))
# print(os.listdir("../input/dataset/dataset"))
# Any results you write to the current directory are saved as output.

# %% [markdown]
# ## Install Dependencies

# %%
import torch
print(torch.__version__)
print(torch.cuda.device_count())
print(torch.cuda.is_available())

# %% [markdown]
# ## Import libraries

# %%
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# %% [markdown]
# ## Hyper-parameters

# %%
dataroot = "./t2"
ckptroot = "./model/all_0207"

lr = 1e-4
weight_decay = 1e-5
batch_size = 32
num_workers = 8
test_size = 0.8
shuffle = True

epochs = 80
start_epoch = 0
resume = False

# %% [markdown]
# ## Helper functions

# %%
def toDevice(datas, device):
    """Enable cuda."""
    imgs, angles = datas
    return imgs.float().to(device), angles.float().to(device)


def augment(dataroot, imgName, angle):
    """Data augmentation."""
    name = os.path.join(dataroot, imgName)
    current_image = cv2.imread(name)/255

    if current_image is None:
        print(name)

    current_image = current_image[130:, :, :]
    if np.random.rand() < 0.5:
        current_image = cv2.flip(current_image, 1)
        angle = angle * -1.0

    return current_image, angle

# %% [markdown]
# ## Load data

# %%
# def load_data(data_dir, test_size):
#     """Load training data and train validation split"""
#     pass

#     # reads CSV file into a single dataframe variable
#     data_df = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'),
#                           names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

#     # Divide the data into training set and validation set
#     train_len = int(test_size * data_df.shape[0])
#     valid_len = data_df.shape[0] - train_len
#     trainset, valset = data.random_split(
#         data_df.values.tolist(), lengths=[train_len, valid_len])

#     return trainset, valset

# trainset, valset = load_data(dataroot, test_size)

# %% [markdown]
# ## Create dataset

# %%
class CustomDataset(data.Dataset):

    def __init__(self, dataroot, transform=None):
        self.dataroot = dataroot
        self.transform = transform
        self.samples = []
        for root, dirs, files in os.walk(dataroot):
            for file in files:
                if file.endswith(".jpg"):
                    self.samples.append(os.path.join(root, file))
        self.samples = natsort.natsorted(self.samples)
    def __getitem__(self, index):
        batch_samples = self.samples[index]
        steering_angle = float(batch_samples[-8:-4])
        # batch_samples = batch_samples[:-8] + batch_samples[-4:]
        
        center_img, angle = augment(self.dataroot, batch_samples, steering_angle)
        # left_img, steering_angle_left     = augment(self.dataroot, batch_samples[1], steering_angle + 0.4)
        # right_img, steering_angle_right   = augment(self.dataroot, batch_samples[2], steering_angle - 0.4)

        center_img = self.transform(center_img)
        # left_img   = self.transform(left_img)
        # right_img  = self.transform(right_img)

        return center_img,angle

    def __len__(self):
        return len(self.samples)

# %% [markdown]
# ## Get data loader

# %%
print("==> Preparing dataset ...")
def data_loader(dataroot, batch_size, shuffle, num_workers):
    """Self-Driving vehicles simulator dataset Loader.

    Args:
        trainset: training set
        valset: validation set
        batch_size: training set input batch size
        shuffle: whether shuffle during training process
        num_workers: number of workers in DataLoader

    Returns:
        trainloader (torch.utils.data.DataLoader): DataLoader for training set
        testloader (torch.utils.data.DataLoader): DataLoader for validation set
    """
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # [transforms.Lambda(lambda x: (x / 127.5) - 1.0)])

    # Load training data and validation data
    training_set = CustomDataset(dataroot, transformations)
    trainloader = DataLoader(training_set,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)

    # validation_set = CustomDataset(dataroot, valset, transformations)
    # valloader = DataLoader(validation_set,
    #                        batch_size=batch_size,
    #                        shuffle=shuffle,
    #                        num_workers=num_workers)

    return trainloader#, valloader


trainloader = data_loader('/home/wenbo/Documents/data/Automating-Car-Steering/data/train/',
                                            batch_size,
                                            shuffle,
                                            num_workers)

validationloader = data_loader('/home/wenbo/Documents/data/Automating-Car-Steering/data/test/',
                                            64,
                                            False,
                                            num_workers)

# %% [markdown]
# ## Define model

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
print(list(model.parameters()))

# %% [markdown]
# ## Define optimizer and criterion

# %%
# Define optimizer and criterion
optimizer = optim.Adam(model.parameters(),
                       lr=lr,
                       weight_decay=weight_decay)
criterion = nn.MSELoss()

# %% [markdown]
# ## Learning rate scheduler

# %%
# learning rate scheduler
scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)

# transfer to gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %% [markdown]
# ## Resume training

# %%
if resume:
    print("==> Loading checkpoint ...")
    checkpoint = torch.load("../input/pretrainedmodels/both-nvidia-model-61.h5",
                            map_location=lambda storage, loc: storage)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

# %% [markdown]
# ## Train

# %%
class Trainer(object):
    """Trainer."""

    def __init__(self,
                 ckptroot,
                 model,
                 device,
                 epochs,
                 criterion,
                 optimizer,
                 scheduler,
                 start_epoch,
                 trainloader,
                 validationloader):
        """Self-Driving car Trainer.

        Args:
            model:
            device:
            epochs:
            criterion:
            optimizer:
            start_epoch:
            trainloader:
            validationloader:

        """
        super(Trainer, self).__init__()

        self.model = model
        self.device = device
        self.epochs = epochs
        self.ckptroot = ckptroot
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.trainingerror=[]
        self.validationerror=[]

    def train(self):
        """Training process."""
        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            self.scheduler.step()
            
            # Training
            train_loss = 0.0
            self.model.train()

            for local_batch, centers in enumerate(self.trainloader):
                # Transfer to GPU
                centers = toDevice(centers, self.device)

                # Model computations
                self.optimizer.zero_grad()
                # datas = [centers, lefts, rights]
                # for data in datas:
                imgs, angles = centers
                # print("training image: ", imgs.shape)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, angles.unsqueeze(1))
                loss.backward()
                self.optimizer.step()

                train_loss += loss.data.item()

                if local_batch % 100 == 0:
                    self.trainingerror.append(train_loss / (local_batch + 1))
                    print("Training Epoch: {} | Loss: {}".format(epoch, train_loss / (local_batch + 1)))


            # Validation
            self.model.eval()
            valid_loss = 0
            with torch.set_grad_enabled(False):
                for local_batch, centers in enumerate(self.validationloader):
                    # Transfer to GPU
                    centers= toDevice(centers, self.device)

                    # Model computations
                    self.optimizer.zero_grad()
                    # datas = [centers, lefts, rights]
                    # for data in datas:
                    imgs, angles = centers
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, angles.unsqueeze(1))

                    valid_loss += loss.data.item()

                    if local_batch % 100 == 0:
                        print("Validation Loss: {}".format(valid_loss / (local_batch + 1)))
                        self.validationerror.append(valid_loss / (local_batch + 1))


            print()
            # Save model
            if epoch % 5 == 0 or epoch == self.epochs + self.start_epoch - 1:

                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }

                self.save_checkpoint(state)

    def save_checkpoint(self, state):
        """Save checkpoint."""
        print("==> Save checkpoint ...")
        if not os.path.exists(self.ckptroot):
            os.makedirs(self.ckptroot)

        torch.save(state, os.path.join(self.ckptroot, 'both-nvidia-model-{}.h5'.format(state['epoch'])))


# %%
print("==> Start training ...")
trainer = Trainer(ckptroot,
                  model,
                  device,
                  epochs,
                  criterion,
                  optimizer,
                  scheduler,
                  start_epoch,
                  trainloader,
                  validationloader)
trainer.train()

# %%
# print(list(trainer.model.parameters()))

# %%
from matplotlib import pyplot as plt
plt.plot(trainer.trainingerror)
plt.plot(trainer.validationerror)
plt.title('Loss Plots')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# %%
fig, ax = plt.subplots( 2 )
ax[0].plot(trainer.trainingerror)
ax[0].set_title('training loss')
ax[1].plot(trainer.validationerror,'tab:orange')
ax[1].set_title('validation loss')
#ax.plot((trainer.validationerror)
fig.savefig('./losses2.png')
plt.close(fig) 


