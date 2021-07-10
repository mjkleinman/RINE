from logger import Logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
import seaborn as sns
from utils import mkdir, get_entropy_bits, nats_to_bits
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--beta', default=15, type=int)
parser.add_argument('--num-seeds', default=1, type=int)
args = parser.parse_args()

basepath = 'logs/canonical/'
operations = ['and', 'unq', 'imperfectrdn', 'rdnxor']
nclasses = [2, 4, 2, 4]
num_seeds = args.num_seeds
beta = args.beta
test_info = np.zeros((len(operations), num_seeds))

# take the mean
for i, (operation, nclass) in enumerate(zip(operations, nclasses)):
    for j, seed in enumerate(range(num_seeds)):
        path = f"mode=toy-operation={operation}-beta{beta}-e30-seed{seed}"
        logger = Logger.load(filename='data', path=basepath + path)
        redundant_loss = np.mean([(v['loss1'] + v['loss2']) / 2 for v in logger.get('valid')][-5:])
        if operation == 'and':  # hardcoded since H(Y) is lower for AND
            test_info[i, j] = 0.811 - nats_to_bits(redundant_loss)
        else:
            test_info[i, j] = get_entropy_bits(nclass) - nats_to_bits(redundant_loss)

test_info_mean = np.mean(test_info, axis=1)
test_info_std = np.std(test_info, axis=1)
test_info = {'and': test_info_mean[0], 'unq': test_info_mean[1], 'imperfectrdn': test_info_mean[2], 'rdnxor': test_info_mean[3]}
std = {'and': test_info_std[0], 'unq': test_info_std[1], 'imperfectrdn': test_info_std[2], 'rdnxor': test_info_std[3]}

print("mean: ##########")
print (test_info)
print("std: ##########")
print(std)
