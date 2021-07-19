# RINE: Redundant Information Neural Estimation

## What is this?

A method to approximate the component of information shared by a set of sources about a target.

## Requirements

- python 3.6+
- torch
- torchvision
- scipy
- seaborn

## Quick Start

To run on CIFAR-10 where each view of the image is 16 pixels wide, run the following:

`python redundant.py --slow --arch=resnet --lr=0.0075 --schedule 40 --length_image 16 --wd 0.005 --beta_schedule -b 50 --save-final --log-name data -l logs/cifar/length/lr=0.0075-e40-width=16-beta=50 --mode length -d cuda --nclasses 10`

To run on they toy example UNQ:

`python redundant.py --slow --weight-decay=0.005 --arch=TOYFCnet --lr=0.01 --schedule 30 -b 5 --log-name data --nclasses 4 --operation unq -d cpu --beta_schedule --mode toy --num_inputs 1 --seed 0 -l logs/canonical/mode=toy-operation=unq-beta5-e30-seed0`

These commands (and many others used in the paper) are generated in `scripts/generate_redundant_commands.py`

## Running on new Dataset

1. Create a new dataloader (similar to what is done in `cifar_redundant_data.py`, `data_toy.py`, and `data_neural.py`)

This is a codebase to accompany the paper: https://openreview.net/forum?id=2MYADuf2o1b

