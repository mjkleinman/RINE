import os
import sys
import random
import pdb


def merge_commands(commands, gpu_cnt=10, max_job_cnt=10000, shuffle=True, put_device_id=False):
    sys.stderr.write(f"Created {len(commands)} commands \n")
    if len(commands) == 0:
        return
    if shuffle:
        random.shuffle(commands)
    merge_cnt = (len(commands) + gpu_cnt - 1) // gpu_cnt
    merge_cnt = min(merge_cnt, max_job_cnt)
    current_device_idx = 0
    for idx in range(0, len(commands), merge_cnt):
        end = min(len(commands), idx + merge_cnt)
        concatenated_commands = "; ".join(commands[idx:end])
        if put_device_id:
            concatenated_commands = concatenated_commands.replace('cuda', f'cuda:{current_device_idx}')
        print(concatenated_commands)
        current_device_idx += 1
        current_device_idx %= gpu_cnt


def check_exists(logdir):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.exists(os.path.join(root_dir, '../', logdir, 'test_info.txt'))


def process_command(command):
    arr = command.split(' ')
    logdir = arr[arr.index('-l') + 1]
    if check_exists(logdir):
        sys.stderr.write(f"Skipping {logdir}\n")
        return []
    else:
        return [command]


#######################################################################################
# Large synthetic examples demonstrating redundancy
#######################################################################################

device = 'cpu'
commands = []
arch = 'TOYFCnet'
mode = 'synthetic_large'
# lr = 0.01  # 0.01
epoch = 50
betas = [50]
nclasses = 8
weight_decay = 0.0005
optimizer = 'adam'


for lr in [0.01, 0.005]:
    for beta in betas:
        for num_inputs in [2, 3, 4, 11, 21]:  # [3, 10, 20]
            for nclasses_per_input in range(4, 9):  # 4-9
                command = f"python redundant.py -d {device} --optimizer={optimizer} --weight-decay={weight_decay} --arch={arch} --lr={lr} --schedule {epoch} -b {beta} --log-name data --beta_schedule --nclasses {nclasses} --nclasses_per_input {nclasses_per_input} --num_inputs {num_inputs} --mode {mode} -l logs/synthetic/mode={mode}-e{epoch}-lr{lr}-beta{beta}-nclasses={nclasses}-nclasses_per_input={nclasses_per_input}-num_inputs={num_inputs}-optimizer={optimizer}"
                commands += process_command(command)


#######################################################################################
# Canonical synthetic examples demonstrating redundancy
#######################################################################################

# commands = []
device = 'cpu'
arch = 'TOYFCnet'
mode = 'toy'
epoch = 30
beta = 15
num_input = 1
operations = ['and', 'unq', 'imperfectrdn', 'rdnxor']
nclasses = [2, 4, 2, 4]
seeds = range(1)

for operation, nclass in zip(operations, nclasses):
    for seed in seeds:
        for beta in [0, 5, 15]:
            command = f"python redundant.py --slow --weight-decay=0.005 --arch={arch} --lr=0.01 --schedule {epoch} -b {beta} --log-name data --nclasses {nclass} --operation {operation} -d {device} --beta_schedule --mode {mode} --num_inputs {num_input} --seed {seed} -l logs/canonical1/mode={mode}-operation={operation}-beta{beta}-e{epoch}-seed{seed}"
            commands += process_command(command)

#######################################################################################
# Neural Data
#######################################################################################


# # Run whatever commands
# lr = 0.01
# epoch = 50
# beta = 100
# modality = 'delay'
# basepath1 = os.path.join('data', 'center_out_data', 'Jenkins_delay', 'Cleaned')
# basepath2 = os.path.join('data', 'center_out_data', 'Reggie_delay', 'Cleaned')
# filenames = {"J1": 'truncated_R_2014-11-05_1.mat',
#              "J2": 'truncated_R_2014-11-06_11.mat',
#              "J3": 'truncated_R_2014-11-07_1.mat',
#              "J4": 'truncated_R_2014-11-10_1.mat',
#              "R1": 'truncated_R_2015-06-11_1.mat',
#              "R2": 'truncated_R_2015-06-15_1.mat',
#              "R3": 'truncated_R_2015-06-16_1.mat'}
# commands = []
# path1 = ''
# path2 = ''
# for key1 in filenames.keys():
#     if key1[0] == 'R':
#         path1 = os.path.join(basepath2, filenames[key1])
#     elif key1[0] == 'J':
#         path1 = os.path.join(basepath1, filenames[key1])
#     else:
#         raise ValueError("Data {} not valid.".format(key1))
#     for key2 in filenames.keys():
#         if key2[0] == 'R':
#             path2 = os.path.join(basepath2, filenames[key2])
#         elif key2[0] == 'J':
#             path2 = os.path.join(basepath1, filenames[key2])

#         # PMD - PMD
#         command = f"python redundant_all.py -d {device} --slow --weight-decay=0.005 --arch=SimpleNet --lr={lr} --schedule {epoch} -b {beta} --log-name data --optimizer adam --batch_size 10 --num_inputs 96 --nclasses 8 --mode {modality} --beta_schedule --path1 {path1} --path2 {path2} -l logs/neural/center_out-e{epoch}-lr{lr}-beta{beta}-mode-{modality}-path1-{key1}-path2-{key2}"
#         commands += process_command(command)

#         # PMD and Motor cortex
#         command = f"python redundant_all.py -d {device} --slow --weight-decay=0.005 --arch=SimpleNet --lr={lr} --schedule {epoch} -b {beta} --log-name data --optimizer adam --batch_size 10 --num_inputs 96 --nclasses 8 --mode {modality} --beta_schedule --path1 {path1} --path2 {path2} --raster spikeRaster -l logs/neural/pmd-motor-e{epoch}-lr{lr}-beta{beta}-mode-{modality}-path1-{key1}-path2-{key2}"
#         commands += process_command(command)

#         # Motor - motor
#         command = f"python redundant_all.py -d {device} --slow --weight-decay=0.005 --arch=SimpleNet --lr={lr} --schedule {epoch} -b {beta} --log-name data --optimizer adam --batch_size 10 --num_inputs 96 --nclasses 8 --mode {modality} --beta_schedule --path1 {path1} --path2 {path2} --raster spikeRaster -l logs/neural_test/motor-motor-e{epoch}-lr{lr}-beta{beta}-mode-{modality}-path1-{key1}-path2-{key2}"
#         commands += process_command(command)


#######################################################################################
# Neural Data Time Analysis on Sergey Data
#######################################################################################


# basepath = os.path.join('data', 'center_out_data', 'Jenkins_delay', 'Cleaned')
# filename = 'truncated_R_2014-11-07_1.mat'
# path1 = path2 = os.path.join(basepath, filename)

# mode = 'time_new'
# for beta in [0, 50]:
#     for time1 in [0, 100, 200, 300]:
#         for time2 in [0, 100, 200, 300]:
#             # run_id = "neural_times_newdata_t1${time1}_t2${time2}_dt100_beta${beta}"
#             command = f"python redundant_all.py -d cpu --slow --weight-decay=0.0005 --arch=SimpleNet --lr=0.01 --schedule 50 -b {beta} --log-name data --optimizer sgd --batch_size 10 --mode {mode} --beta_schedule --time1 {time1} --time2 {time2} --deltat 100 --num_inputs 96 --nclasses 8 --path1 {path1} --path2 {path2} -l logs/neural_time_new_sergey/t1={time1}_t2={time2}_dt100_beta={beta}_mode={mode}"
#             commands += process_command(command)

# merge_commands(commands, gpu_cnt=1, put_device_id=True, shuffle=False)


#######################################################################################
# CIFAR Analyses Confusion Matrix
#######################################################################################

device = 'cuda'
epoch = 40
lr = 0.0075
beta = 50
wd = 0.005
command = f"python redundant.py --slow --arch=resnet --lr={lr} --schedule 40 --length_image 16 --beta_schedule --wd {wd} -b {beta} --save-final --log-name data -l logs/cifar/confusion/length/lr={lr}-e{epoch}-width=16-beta={beta} --mode length -d {device} --nclasses 10 --get-confusion"
commands += process_command(command)
command = f"python redundant.py --slow --arch=resnet --lr={lr} --schedule 40 --length_image 32 --beta_schedule --wd {wd} -b {beta} --save-final --log-name data -l logs/cifar/confusion/channel/lr={lr}-e{epoch}-beta={beta} --mode channel -d {device} --nclasses 10 --get-confusion"
commands += process_command(command)
lr = 0.05
command = f"python redundant.py --slow --arch=resnet --lr={lr} --schedule {epoch} --length_image 32 --wd {wd} --beta_schedule -b {beta} --save-final --log-name data -l logs/cifar/confusion/fft/lr={lr}-e{epoch}-beta={beta} --mode fft -d {device} --nclasses 10 --get-confusion"
commands += process_command(command)
# merge_commands(commands, gpu_cnt=4, put_device_id=True)

#######################################################################################
# CIFAR Analyses for different beta and Ablation for fixed beta
#######################################################################################

device = 'cuda'
wd = 0.005
epoch = 40
lr = 0.0075
betas = [0, 50]
widths = [12, 16, 24, 32]
for beta in betas:
    for width in widths:
        command = f"python redundant.py --slow --arch=resnet --lr={lr} --schedule 40 --length_image {width} --wd {wd} --beta_schedule -b {beta} --save-final --log-name data -l logs/cifar/length/lr={lr}-e{epoch}-width={width}-beta={beta} --mode length -d {device} --nclasses 10"
        commands += process_command(command)
        # ablation for fixed beta
        if beta == 50:
            command = f"python redundant.py --slow --arch=resnet --lr={lr} --schedule 40 --length_image {width} --wd {wd} -b {beta} --save-final --log-name data -l logs/cifar/length/lr={lr}-e{epoch}-width={width}-beta={beta}-nosched --mode length -d {device} --nclasses 10"
            commands += process_command(command)
merge_commands(commands, gpu_cnt=1, put_device_id=True)
