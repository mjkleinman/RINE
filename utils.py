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


def get_parameter(model, parameter):
    result = []
    if hasattr(model, parameter):
        result.append(getattr(model, parameter))
    for l in model.children():
        result += get_parameter(l, parameter)
    return result


def KL_div2(mu, logsigma, mu1, logsigma1):
    sigma_2 = logsigma.mul(2).exp_()
    mu1 = mu1.expand_as(mu)
    logsigma1 = logsigma1.expand_as(logsigma)
    sigma1_2 = logsigma1.mul(2).exp_().add(1e-7)
    return (mu - mu1).pow(2).div(sigma1_2).add_(sigma_2.div(sigma1_2)).mul_(-1).add_(1).add_(logsigma.mul(2)).add_(logsigma1.mul(-2)).mul_(-0.5)


def tanh_scale(x, min_val, max_val):
    d = (max_val - min_val) / 2.
    y = torch.tanh(x / d) * d + (max_val + min_val) / 2.
    c = 1. / torch.cosh(x / d)**2
    return y, c


def sigmoid_scale(x, min_val, max_val):
    d = (max_val - min_val)
    y = F.sigmoid(x / d) * d + min_val
    return y


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


def call_on_model(model, name, *args, **kwargs):
    results = []
    if hasattr(model, name):
        results += [getattr(model, name)(*args, **kwargs)]
    for l in model.children():
        results += call_on_model(l, name, *args, **kwargs)
    return results


def print_sigma(model):
    if hasattr(model, 'logsigma'):
        print (model.logsigma.data.exp().mean() / model.prior_sigma)
    for l in model.children():
        print_sigma(l)


def _collect_kl(model, mean=False):
    r = []
    if hasattr(model, 'logsigma'):
        if mean:
            r += [model._kl(model.net_conv.weight, model.logsigma).data.mean()]
        else:
            r += [model._kl(model.net_conv.weight, model.logsigma).data.sum()]
    for l in model.children():
        r += _collect_kl(l, mean=mean)
    return r


def _collect_gradient(model, logsigma=False):
    r = []
    if hasattr(model, 'logsigma'):
        if not logsigma:
            r += [model.net_conv.weight.grad.data.abs().mean()]
        else:
            r += [model.logsigma.grad.data.abs().mean()]
    for l in model.children():
        r += _collect_gradient(l, logsigma=logsigma)
    return r


def set_parameter(model, parameter, value):
    if hasattr(model, parameter):
        setattr(model, parameter, value)
    for l in model.children():
        set_parameter(l, parameter, value)


def interpolate(t0, t, T, v_initial=0., v_final=1.):
    return np.clip(v_initial * (1 - float(t - t0) / T) + v_final * (float(t - t0) / T), min(v_initial, v_final), max(v_initial, v_final))


def flatten(x):
    return x.view(x.size(0), -1)
