import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.backends.cudnn as cudnn
import numpy as np

from IPython import embed
import os
#import cPickle as pickle
import pickle
import random
import math


def get_entropy_bits(n):
    # assuming each class is equiprobable
    # return math.log2(n)
    return math.log(n, 2)


def nats_to_bits(x):
    return x * 1.44


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def imshow(image, colormap=False, video=False):
    import imageio
    import iterm2_tools
    from matplotlib import cm
    import matplotlib.pyplot as plt
    # from scipy.misc import bytescale
    import skimage
    from PIL import Image
    from iterm2_tools.images import display_image_bytes

    if type(image).__name__ == 'Variable':
        image = image.data
    if 'torch.cuda' in type(image).__module__:
        image = image.cpu()
    if 'Tensor' in type(image).__name__:
        image = image.numpy()

    if colormap:
        image = (cm.Blues(image) * 255).astype(np.uint8)
    else:
        image = skimage.util.img_as_ubyte(image)

    if image.ndim == 4:
        video = True
    if image.ndim == 3 and (image.shape[0] not in [1, 3] and image.shape[-1] not in [1, 3]):
        video = True

    if video:
        if image.shape[1] == 3:
            image = image.transpose([2, 3, 1]).astype(np.uint8)
        image = image.squeeze()
        if image.ndim == 2:
            image = image[None]
        images = [im for im in image]
        s = imageio.mimsave(imageio.RETURN_BYTES, images, format='gif', duration=0.3)
        print (display_image_bytes(s))
    else:
        if image.shape[0] == 3:
            image = image.transpose([1, 2, 0]).astype(np.uint8)
        image = image.squeeze()
        s = imageio.imsave(imageio.RETURN_BYTES, image, format='png')
        print (display_image_bytes(s))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_error(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(100. - correct_k.mul_(100.0 / batch_size))
    return res


def set_norm(model, train=True):
    if isinstance(model, nn.BatchNorm1d) or isinstance(model, nn.BatchNorm2d):
        if train:
            model.train()
        else:
            model.eval()
    for l in model.children():
        set_norm(l, train=train)


def set_batchnorm_mode(model, train=True):
    if isinstance(model, nn.BatchNorm1d) or isinstance(model, nn.BatchNorm2d):
        if train:
            model.train()
        else:
            model.eval()
    for l in model.children():
        set_norm(l, train=train)


def interpolate(t0, t, T, v_initial=0., v_final=1.):
    return np.clip(v_initial * (1 - float(t - t0) / T) + v_final * (float(t - t0) / T), min(v_initial, v_final), max(v_initial, v_final))


def flatten(x):
    return x.view(x.size(0), -1)
