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
parser.add_argument('--run_ids', nargs='+', default=None, type=str)
parser.add_argument('--name', dest='plot_path', default='plots/redundancy_var_beta50.pdf', type=str)
# parser.add_argument('--legend', nargs='+', default=['direction','color'], type=str)
args = parser.parse_args()

run_ids = args.run_ids
num_classes = 10

basepath = 'logs/cifar/length/'
betas = [0, 50]
widths = [12, 16, 24, 32]
epoch = 40
lr = 0.0075
run_id = 'data'
test_info = np.zeros((len(betas), len(widths)))

for i, beta in enumerate(betas):
    for j, width in enumerate(widths):
        path = f"lr={lr}-e{epoch}-width={width}-beta={beta}"
        logger = Logger.load(run_id, path=basepath + path)
        redundant_loss = np.mean([(v['loss1'] + v['loss2']) / 2 for v in logger.get('valid')][-5:])
        test_info[i, j] = get_entropy_bits(num_classes) - nats_to_bits(redundant_loss)


sns.set(font_scale=1.5)
plt.plot(widths, test_info[0, :], label='B=0', color='blue', marker='o')
plt.plot(widths, test_info[1, :], label='B=50', color='green', marker='o')


# plt.plot(lengths, usable_info, label='usable info', color='black', marker='o')
plt.ylabel('Redundant Information (bits)')
plt.ylim((-0.05, get_entropy_bits(num_classes) + 0.05))
plt.xlabel('Width of image')
plt.xticks([12, 16, 20, 24, 28, 32])
plt.legend(loc='lower right')
# if args.plot_path == '':
#     plt.savefig(os.path.join(plot_path, 'mi_epochs.pdf'), format='pdf')
# else:
plt.savefig(args.plot_path, format='pdf', dpi=None, bbox_inches='tight')
plt.close()
