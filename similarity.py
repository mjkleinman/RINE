# compute the similarity between inputs for the redundant info task:

from data_toy import get_redundant_dataset_loader
import sys
import scipy.spatial.distance as distance
import numpy as np
from utils import AverageMeter
import seaborn as sns
import matplotlib.pyplot as plt


def cosyne_dist(x1, x2):
    return distance.cosine(x1, x2)


if __name__ == '__main__':
    dist = []
    num_appends = [0, 1, 2, 5, 10, 20]
    num_data = 2000
    common_dist = []

    for num_append in num_appends:
        trainloader, testloader = get_redundant_dataset_loader(num_data_samples=num_data, batch_size=num_data, num_append=num_append, num_classes_per_input=8, operation='append_noise')
        for x1, x2, y in trainloader:
            dist.append(np.nanmean([1 - cosyne_dist(x1[i], x2[i]) for i in range(y.shape[0])]))

    for num_append in num_appends:
        trainloader, testloader = get_redundant_dataset_loader(num_data_samples=num_data, batch_size=num_data, num_append=num_append, num_classes_per_input=0, operation='append_common')
        for x1, x2, y in trainloader:
            common_dist.append(np.nanmean([1 - cosyne_dist(x1[i], x2[i]) for i in range(y.shape[0])]))

    sns.set()

    plt.ylabel('Cosine Similarity')
    # plt.ylim((-0.1, get_entropy_bits(nclasses)))
    plt.xlabel('Number of uncorrelated inputs')
    plt.plot(num_appends, dist, 'k--', label='noisy inputs', marker='o')
    plt.plot(num_appends, common_dist, 'k', label='common inputs', marker='o')

    # plt.xticks([12, 16, 20, 24, 28, 32])
    plt.legend()
    plot_path = 'plots/distance_cosyne.pdf'
    plt.savefig(plot_path, format='pdf', dpi=None, bbox_inches='tight')
    plt.close()
