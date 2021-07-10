from logger import Logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils import get_entropy_bits, nats_to_bits
import argparse
import numpy as np
import pdb
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--plot-path', default='plots/redundant_cifar_confusion.pdf', type=str)
args = parser.parse_args()

nclasses = 10
run_id = 'data'
basepath = 'logs/cifar/confusion/'

path = 'channel/lr=0.0075-e40-beta=50'
logger = Logger.load(run_id, path=basepath + path)
confusion = logger.get('confusion')
confusion_channels = [confusion[0]['losses_confusion'][j].avg for j in range(nclasses)]

path = 'length/lr=0.0075-e40-width=16-beta=50'
logger = Logger.load(run_id, path=basepath + path)
confusion = logger.get('confusion')
confusion_crops = [confusion[0]['losses_confusion'][j].avg for j in range(nclasses)]

path = 'fft/lr=0.05-e40-beta=50'
logger = Logger.load(run_id, path=basepath + path)
confusion = logger.get('confusion')
confusion_fft = [confusion[0]['losses_confusion'][j].avg for j in range(nclasses)]

confusion_matrix = np.zeros((3, 10))
confusion_matrix[0, :] = confusion_channels
confusion_matrix[1, :] = confusion_crops
confusion_matrix[2, :] = confusion_fft

# convert from loss to info
confusion_matrix = get_entropy_bits(nclasses) - nats_to_bits(confusion_matrix)
print(confusion_matrix)
fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix[:, :])
ax.set_xticks(np.arange(confusion_matrix.shape[1]))
ax.set_yticks(np.arange(confusion_matrix.shape[0]))
ax.set_xticklabels(['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
ax.set_yticklabels(['Channels', 'Crops', 'Freq'])


for i in range(3):
    for j in range(10):
        text = ax.text(j, i, "{:.2f}".format(confusion_matrix[i, j]),
                       ha="center", va="center", color="w")

fig.tight_layout()
plot_path = args.plot_path
#ax.set_title("Beta {}".format(beta))
plt.savefig(plot_path, format='pdf', dpi=None, bbox_inches='tight')
plt.close()
plt.clf()
