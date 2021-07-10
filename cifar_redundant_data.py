import pdb
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision
import scipy.fft as fft
from utils import imshow
import sys
import matplotlib.pyplot as plt


def fft_image_channel(img, n=2):
    F1 = fft.fft2((img).astype(float))
    F2 = fft.fftshift(F1)
    (w, h) = img.shape
    half_w, half_h = int(w / 2), int(h / 2)

    # high pass filter, todo: make a circle
    mask_low = np.zeros_like(img)
    mask_high = np.ones_like(img)

    mask_high[half_w - n:half_w + n + 1, half_h - n:half_h + n + 1] = 0
    mask_low[half_w - n:half_w + n + 1, half_h - n:half_h + n + 1] = 1
    F_high_pass = F2 * mask_high
    F_low_pass = F2 * mask_low
    # F_high_pass = F2 - F_low_pass
    # F2[half_w - n:half_w + n + 1, half_h - n:half_h + n + 1] = 0  # select all but the first 50x50 (low) frequencies
    channel_transform_high = fft.ifft2(fft.ifftshift(F_high_pass)).real
    channel_transform_low = fft.ifft2(fft.ifftshift(F_low_pass)).real

    return channel_transform_high, channel_transform_low


def fft_image(img):
    image_transform_h = np.zeros_like(img)
    image_transform_l = np.zeros_like(img)
    for i in range(3):
        image_transform_h[:, :, i], image_transform_l[:, :, i] = fft_image_channel(img[:, :, i])
    return image_transform_h, image_transform_l


# This is for the redundant info, can add the channels and length here as well
class DoubleCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True,
                 transform=None, transform_b=None,
                 target_transform=None, download=False, operation='fft', length_image=16):
        super(DoubleCIFAR10, self).__init__(root, train=train, transform=None, target_transform=target_transform,
                                            download=download)
        self.transform = transform
        self.transform_b = transform_b
        self.operation = operation
        self.length_image = length_image

        if train and (operation == 'length' or operation == 'channel'):
            self.default_transform = transforms.Compose([
                transforms.Pad(padding=4, fill=(125, 123, 113)),
                transforms.RandomCrop(32, padding=0),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.default_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def __getitem__(self, index):

        img, img_b, target = self.data[index], self.data[index], self.targets[index]
        if self.operation == 'fft':
            img, img_b = fft_image(img)

        # Return a PIL Image
        img = Image.fromarray(img)
        img_b = Image.fromarray(img_b)
        img = self.default_transform(img)
        img_b = self.default_transform(img_b)

        # TODO: Maybe I need to copy img_b so operations don't change both
        if self.operation == 'length':
            img[:, :, :(32 - self.length_image)] = 0.
            if self.length_image < 32:
                img_b[:, :, self.length_image:] = 0.

        # This only keeps one of the channels, by zeroing out the other two
        if self.operation == 'channel':
            img[:2, :, :] = 0.
            img_b[1:, :, :] = 0.

        return img, img_b, target

def get_cifar_redundant_loaders(operation, length=32, workers=0, batch_size=128):

    trainset = DoubleCIFAR10(root=os.path.join(os.environ['HOME'], 'data'), train=True, download=False, transform=None, transform_b=None, operation=operation, length_image=length)
    testset = DoubleCIFAR10(root=os.path.join(os.environ['HOME'], 'data'), train=False, download=False, transform=None, transform_b=None, operation=operation, length_image=length)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=workers)
    return trainloader, testloader

# This should be deprecated for get_cifar_redundant_loaders
def get_cifar_frequency_loaders(workers=0, batch_size=128, augment=True, deficit=[]):

    trainset = DoubleCIFAR10(root=os.path.join(os.environ['HOME'], 'data'), train=True, download=False, transform=None, transform_b=None)
    testset = DoubleCIFAR10(root=os.path.join(os.environ['HOME'], 'data'), train=False, download=False, transform=None, transform_b=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=workers)
    return trainloader, testloader

# This was used in the paper for crops and channels
# def get_cifar_variational_loaders(workers=0, batch_size=128, augment=True, deficit=[]):

#     transform = []
#     transform += [
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ]

#     transform_train = []
#     if augment and 'noise' not in deficit:
#         transform_train += [
#             transforms.Pad(padding=4, fill=(125, 123, 113)),
#             transforms.RandomCrop(32, padding=0)]
#         # transforms.RandomHorizontalFlip()]
#     transform_train += transform
#     transform_train = transforms.Compose(transform_train)

#     transform_test = []
#     transform_test += transform
#     transform_test = transforms.Compose(transform_test)

#     trainset = torchvision.datasets.CIFAR10(root=os.path.join(os.environ['HOME'], 'data'), train=True, download=True, transform=transform_train)
#     testset = torchvision.datasets.CIFAR10(root=os.path.join(os.environ['HOME'], 'data'), train=False, download=False, transform=transform_test)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=workers)

#     return trainloader, testloader
