# RINE: Redundant Information Neural Estimation

## What is this?

A method to approximate the component of information shared by a set of sources.

## Quick Start

To run on CIFAR-10, for a width of 16:

`python redundant.py --slow --arch=resnet --lr=0.0075 --schedule 40 --length_image 16 --wd 0.005 --beta_schedule -b 50 --save-final --log-name data -l logs/cifar/length/lr=0.0075-e40-width=16-beta=50 --mode length -d cuda --nclasses 10`

To run on a toy example:

`Update`

The script to generate the commands used in the paper can be found in `scripts/generate_redundant_commands.py`

## Running on new Dataset

1. Create a new Dataloader (similar to what is done in `cifar_redundant_data.py` and `data_toy.py`)

This is a codebase to accompany the paper: https://openreview.net/forum?id=2MYADuf2o1b
