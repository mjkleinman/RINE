from logger import Logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils import get_entropy_bits, nats_to_bits
import argparse
import numpy as np
import pdb
import utils
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument('--run_ids', nargs='+', default=None, type=str)
parser.add_argument('--name', dest='plot_path', default='plots/redundant_demonstration2.pdf', type=str)
args = parser.parse_args()
run_ids = args.run_ids
run_id = 'data'

mode = 'synthetic_large'
lr = 0.01  # 0.01
epoch = 50
# beta = 50
nclasses = 8

for lr in [0.01]:
    for beta in [50]:
        num_inputs_list = [2, 3, 4, 11, 21]
        num_inputs_plot_list = [n - 1 for n in num_inputs_list]

        sns.set()
        for nclasses_per_input in range(4, 9):
            redundant_info_list = []
            for num_inputs in num_inputs_list:
                path = f"logs/synthetic/mode={mode}-e{epoch}-lr{lr}-beta{beta}-nclasses={nclasses}-nclasses_per_input={nclasses_per_input}-num_inputs={num_inputs}-optimizer=adam"
                logger = Logger.load(run_id, path=path)
                redundant_info = np.mean([get_entropy_bits(nclasses) - nats_to_bits((v['loss1'] + v['loss2']) / 2) for v in logger.get('valid')][-5:])
                redundant_info_list.append(redundant_info)

            # num_inputs_list = [n - 1 for n in num_inputs_list]
            plt.plot(num_inputs_plot_list, redundant_info_list, label=f"Classes: {nclasses_per_input}", marker='o')

            # plt.plot(lengths, usable_info, label='usable info', color='black', marker='o')

        # if args.plot_path == '':
        #     plt.savefig(os.path.join(plot_path, 'mi_epochs.pdf'), format='pdf')
        # else:
        plt.ylabel('redundant info')
        plt.ylim((-0.1, get_entropy_bits(nclasses) + 0.1))
        plt.xlabel('Number of uncorrelated inputs')
        # plt.xticks([12, 16, 20, 24, 28, 32])
        plt.legend()
        plot_path = f"plots/redundant_demo-e{epoch}-lr{lr}-beta{beta}-optimizer=adam.pdf"
        plt.savefig(plot_path, format='pdf', dpi=None, bbox_inches='tight')
        plt.close()
