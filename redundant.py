#!/usr/bin/env python
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from utils import *
from logger import Logger
from models import *
import pdb

parser = argparse.ArgumentParser(description='Redundant information experiments')
parser.add_argument('--arch', default='resnet', type=str,
                    help='architecture to use')
parser.add_argument('--log-name', default=None, type=str,
                    help='index for the log file')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--schedule', nargs='+', default=[80], type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--beta', default=10, type=int,
                    help='value of beta (default: 10)')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-f', '--filters', default=.25, type=float,
                    help='percentage of filters to use')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.001, type=float,
                    metavar='W', help='weight decay (default: 0.)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--name', default='model', type=str,
                    help='name of the run')
parser.add_argument('--decay', default=0.97, type=float,
                    help='Learning rate exponential decay')
parser.add_argument('--slow', dest='slow', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save', dest='save', action='store_true',
                    help='save the model every x epochs')
parser.add_argument('--save-final', action='store_true',
                    help='save last version of the model')
parser.add_argument('--save-every', default=10, type=int,
                    help='save every N epochs (only used with --save)')
parser.add_argument('-o', '--optimizer', default='sgd', type=str,
                    help='Optimizer to use')
parser.add_argument('--no-bn', dest='batch_norm', action='store_false',
                    help='disable batch normalization')
parser.add_argument('--nclasses', default=2, type=int,
                    help='number of classes')
parser.add_argument('--length_image', default=16, type=int,
                    help='size of each part of image (32 x 32 total)')
parser.add_argument('--log_dir', '-l', type=str, default='logs')
parser.add_argument('--device', '-d', default='cuda')
parser.add_argument('--beta_schedule', action='store_true',
                    help='schedule the beta')
parser.add_argument('--nclasses_per_input', default=6, type=int,
                    help='number of classes per input')
parser.add_argument('--mode', default='delay', type=str,
                    help='modality to use options:[synthetic_large, time, center_out, delay]')
parser.add_argument('--num_inputs', default=3, type=int,
                    help='number of appended inputs (noise or common)')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--get-confusion', action='store_true',
                    help='compute confusion')

# Argmuents RELATED TO NEURAL DATA
parser.add_argument('--path1', type=str, default='')
parser.add_argument('--path2', type=str, default='')
parser.add_argument('--raster', type=str, default='spikeRaster2')
parser.add_argument('--time1', default=150, type=int,
                    help='start of neural recording window 1')
parser.add_argument('--time2', default=350, type=int,
                    help='start of neural recording window 2')
parser.add_argument('--deltat', default=200, type=int,
                    help='window length (ms)')

# Argmunets RELATED TO TOY
parser.add_argument('--operation', default='xor', type=str,
                    help='Operation to perform, options: xor, unq, imperfectrdn, and, rdnxor')


args = parser.parse_args()
device = args.device


# Takes two models and a datloader of (X1, X2, Y)
def train(data_loader, model1, model2, criterion, optimizer, epoch, train=True):
    kls = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    accuracies1 = AverageMeter()
    accuracies2 = AverageMeter()

    # switch to train mode
    if train:
        model1.train()
        model2.train()
    else:
        model1.eval()
        model2.eval()

    for i, (input1, input2, target) in enumerate(data_loader):
        # target = target[0]
        input1 = input1.to(device).float()
        input2 = input2.to(device).float()
        target = target.to(device)

        # compute output
        output1, _ = model1(input1)
        output2, _ = model2(input2)
        loss1 = criterion(output1, target)
        loss2 = criterion(output2, target)
        loss3 = torch.mean(torch.abs(torch.softmax(output1, dim=-1) - torch.softmax(output2, dim=-1)))  # can play around with different distance metrics
        beta = args.beta
        if args.beta_schedule:
            beta = args.beta * (1 - args.decay ** epoch)
        loss = 0.5 * (loss1 + loss2) + beta * loss3

        acc, = get_error(output1, target)
        acc2, = get_error(output2, target)

        losses1.update(loss1.item(), target.size(0))
        losses2.update(loss2.item(), target.size(0))
        losses3.update(loss3.item(), target.size(0))

        accuracies1.update(acc.item(), target.size(0))
        accuracies2.update(acc2.item(), target.size(0))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("loss1 :" + str(losses1.avg))
    print("loss2 :" + str(losses2.avg))
    print("loss3 :" + str(losses3.avg))
    print('[{}] Epoch: [{epoch}] lr: {lr:.4f} Loss1 {loss1.avg:.3f} Loss2 {loss2.avg:.3f} Loss3 {loss3.avg:.3f} Error1: {acc.avg:.2f} Error2: {acc2.avg:.2f}'
          .format('train' if train else 'test', epoch=epoch, lr=optimizer.param_groups[0]['lr'], loss1=losses1, loss2=losses2, loss3=losses3, acc=accuracies1, acc2=accuracies2))

    logger.append('train' if train else 'valid', epoch=epoch, loss1=losses1.avg, loss2=losses2.avg, loss3=losses3.avg, error1=accuracies1.avg, error2=accuracies2.avg, lr=optimizer.param_groups[0]['lr'])


def validate(val_loader, model1, model2, criterion, optimizer, epoch, label=''):
    train(val_loader, model1, model2, criterion, optimizer, epoch, train=False)


def dry_run(train_loader, model1, model2):
    # Simply forward the train data through the model to make sure the stats
    # of batch norm are up to date before running validation
    model1.train()
    model2.train()
    set_batchnorm_mode(model1, train=True)
    set_batchnorm_mode(model2, train=True)
    for i, (input1, input2, target) in enumerate(train_loader):
        input1 = input1.to(device)
        input2 = input2.to(device)
        output1, _ = model1(input1)
        output2, _ = model2(input2)


def save_checkpoint(state_m1, state_m2, step=True):
    if step:
        epoch = state_m1['epoch']
        target_file1 = logger['checkpoint_step_m1'].format(epoch)
        target_file2 = logger['checkpoint_step_m2'].format(epoch)
    else:
        target_file1 = logger['checkpoint_m1']
        target_file2 = logger['checkpoint_m2']
    print ("Saving {}".format(target_file1))
    torch.save(state_m1, target_file1)
    print ("Saving {}".format(target_file2))
    torch.save(state_m2, target_file2)


def adjust_learning_rate(optimizer, epoch, schedule):
    if not args.slow:
        lr = args.lr * (0.1 ** np.less(schedule, epoch).sum())
    else:
        lr = args.lr * args.decay**epoch

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_confusion(val_loader, model1, model2, criterion, save_log=True):
    model1.eval()
    model2.eval()
    losses = []

    for i in range(args.nclasses):
        losses.append(AverageMeter())

    for i, (input1, input2, target) in enumerate(val_loader):
        input1 = input1.to(device).float()
        input2 = input2.to(device).float()
        target = target.to(device)

        # compute output
        output1, _ = model1(input1)
        output2, _ = model2(input2)

        for j in range(args.nclasses):
            idx_class = [idx for idx, val in enumerate(target) if val == j]
            loss1 = criterion(output1[idx_class], target[idx_class])
            loss2 = criterion(output2[idx_class], target[idx_class])
            losses[j].update(0.5 * (loss1.item() + loss2.item()), len(idx_class))

    if save_log:
        with open(os.path.join(args.log_dir, 'confusion_info.txt'), 'w') as f:
            for j in range(args.nclasses):
                f.write(f"Class {j}, info in bits: {get_entropy_bits(args.nclasses) - nats_to_bits(losses[j].avg)} \n")

        logger.append('confusion', losses_confusion=losses)


if __name__ == '__main__':

    # This puts the log file and models/checkpoints into a particular folder name `args.log_dir'
    logger = Logger(index=args.log_name, path=args.log_dir)
    logger['args'] = args
    logger['checkpoint_m1'] = os.path.join(args.log_dir, 'model_m1.pth')
    logger['checkpoint_step_m1'] = os.path.join(args.log_dir, 'model_{}_m1.pth')
    logger['checkpoint_m2'] = os.path.join(args.log_dir, 'model_m2.pth')
    logger['checkpoint_step_m2'] = os.path.join(args.log_dir, 'model_{}_m2.pth')
    print ("[Logging in {}]".format(logger.index))

    n_neurons = args.num_inputs  # this is the number of inputs
    # n_neurons = 97 if args.time else 96  # FIXME: this is very hack

    n_channels = 3
    if args.arch == 'TOYFCnet':
        model1 = TOYFCnet(n_classes=args.nclasses, n_inputs=n_neurons)
        model2 = TOYFCnet(n_classes=args.nclasses, n_inputs=n_neurons)
    elif args.arch == 'SimpleNet':
        model1 = SimpleNet(n_classes=args.nclasses, n_inputs=n_neurons)
        model2 = SimpleNet(n_classes=args.nclasses, n_inputs=n_neurons)
    elif args.arch == 'resnet':
        model1 = ResNet18(n_channels=n_channels, n_classes=args.nclasses)
        model2 = ResNet18(n_channels=n_channels, n_classes=args.nclasses)
    else:
        raise ValueError("Architecture {} not valid.".format(args.arch))

    model1 = model1.to(device)
    model2 = model2.to(device)

    # Get the appropriate dataloader
    # Right now should work for toy (small and large), and neural experiments, fft
    from data_toy import get_redundant_dataset_loader
    from data_neural import get_neural_nocorr_loader, get_neural_time_loader, get_neural_center_loader, get_neural_delay_loader
    from data_toy import get_toy_dataset_loader
    from cifar_redundant_data import get_cifar_frequency_loaders, get_cifar_redundant_loaders

    if args.mode == 'synthetic_large':
        train_loader, test_loader = get_redundant_dataset_loader(num_classes_per_input=args.nclasses_per_input, operation='append_noise', num_append=args.num_inputs - 1)
    elif args.mode == 'toy':
        train_loader, test_loader = get_toy_dataset_loader(operation=args.operation)
    elif args.mode == 'time':
        train_loader, test_loader = get_neural_time_loader(batch_size=args.batch_size, time1=args.time1, time2=args.time2, deltat=args.deltat, path1=args.path1, path2=args.path2, raster=args.raster)
    elif args.mode == 'center_out':
        train_loader, test_loader = get_neural_center_loader(batch_size=args.batch_size, time1=args.time1, time2=args.time2, deltat=args.deltat, path1=args.path1, path2=args.path2, raster=args.raster)
    elif args.mode == 'delay':
        train_loader, test_loader = get_neural_delay_loader(batch_size=args.batch_size, time1=args.time1, time2=args.time2, deltat=args.deltat, path1=args.path1, path2=args.path2, raster=args.raster)
    elif args.mode == 'time_new':
        train_loader, test_loader = get_neural_delay_loader(batch_size=args.batch_size, time1=args.time1, time2=args.time2, deltat=args.deltat, path1=args.path1, path2=args.path2, raster=args.raster, useTime=True)
    elif args.mode == 'nocorr':
        train_loader, test_loader = get_neural_nocorr_loader(batch_size=args.batch_size, time1=args.time1, time2=args.time2, deltat=args.deltat, path1=args.path1, path2=args.path2, raster=args.raster)
    # elif args.mode == 'fft':
    #     train_loader, test_loader = get_cifar_frequency_loaders(batch_size=args.batch_size, workers=args.workers)
    # TODO: test this
    elif args.mode == 'length' or args.mode == 'channel' or args.mode == 'fft':
        train_loader, test_loader = get_cifar_redundant_loaders(operation=args.mode, length=args.length_image, batch_size=args.batch_size, workers=args.workers)
    else:
        raise ValueError("Mode {} not valid.".format(args.mode))

    criterion = nn.CrossEntropyLoss().to(device)

    # Creates the optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(list(model1.parameters()) + list(model2.parameters()), args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), args.lr,
                                     betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer {} not valid.".format(args.optimizer))

    # If a checkpoint is provided, load weights from checkpoint
    if args.resume:
        checkpoint_file = args.resume
        if not os.path.isfile(checkpoint_file):
            print ("=== waiting for checkpoint to exist ===")
            try:
                while not os.path.isfile(checkpoint_file):
                    time.sleep(1)
            except KeyboardInterrupt:
                print ("=== waiting stopped by user ===")
                import sys
                sys.exit()
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_file, checkpoint['epoch']))

    # Main training loop
    for epoch in range(args.start_epoch, args.schedule[-1]):
        adjust_learning_rate(optimizer, epoch, args.schedule[:-1])
        loss = train(train_loader, model1, model2, criterion, optimizer, epoch)
        # dry_run(train_loader, model1)
        validate(test_loader, model1, model2, criterion, optimizer, epoch)

    if args.get_confusion:
        get_confusion(test_loader, model1, model2, criterion)

    # Save some results
    with open(os.path.join(args.log_dir, 'test_info.txt'), 'w') as f:
        entropy = get_entropy_bits(args.nclasses)
        test_info = [entropy - nats_to_bits((v['loss1'] + v['loss2']) / 2) for v in logger.get('valid')]
        f.write("{}\n".format(test_info))

    with open(os.path.join(args.log_dir, 'train_info.txt'), 'w') as f:
        entropy = get_entropy_bits(args.nclasses)
        train_info = [entropy - nats_to_bits((v['loss1'] + v['loss2']) / 2) for v in logger.get('train')]
        f.write("{}\n".format(train_info))

    logger['finished'] = True
    print ("[Logs in {}]".format(logger.index))
