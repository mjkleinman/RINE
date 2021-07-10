import numpy as np
import scipy.io as sio
import torch.utils.data
from torch.utils.data import DataLoader
import pdb


class NeuralData(torch.utils.data.Dataset):
    def __init__(self, data, data2, num_trials_per_class=91):
        self.data = data
        self.data2 = data2
        self.num_trials_per_class = num_trials_per_class
        self.size = data.shape[0]

    def __getitem__(self, index):

        input1_data = self.data[index]
        input2_data = self.data2[index]
        target = index // self.num_trials_per_class
        return input1_data, input2_data, target

    def __len__(self):
        return self.size


def break_correlations(data):
    # data is a TxN matrix, representing trials by neurons (and I want to permute the neurons across trials differently to break single trial correlations)
    permuted_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        permuted_data[:, i] = np.random.permutation(data[:, i])
    return permuted_data

def get_neural_nocorr_loader(workers=0, batch_size=10, time1=None, time2=None, deltat=None):

    data = sio.loadmat('data/ps4_realdata.mat')  # load the .mat file.
    NumTrainData = data['train_trial'].shape[0]
    NumClass = data['train_trial'].shape[1]
    NumTestData = data['test_trial'].shape[0]
    trainDataArr = np.zeros((NumClass, NumTrainData, 97))  # contains the firing rates for all neurons on all 8 x 91 trials in the training set
    testDataArr = np.zeros((NumClass, NumTestData, 97))  # for the testing set.
    for classIX in range(NumClass):
        for trainDataIX in range(NumTrainData):
            trainDataArr[classIX, trainDataIX, :] = np.sum(data['train_trial'][trainDataIX, classIX][1][:, 350:550], 1)
        for testDataIX in range(NumTestData):
            testDataArr[classIX, testDataIX, :] = np.sum(data['test_trial'][testDataIX, classIX][1][:, 350:550], 1)

    # permute the data to break the single trial correlations
    trainDataArrNoCorr = np.zeros((NumClass, NumTrainData, 97))
    for classIX in range(NumClass):
        trainDataArrNoCorr[classIX, :, :] = break_correlations(trainDataArr[classIX, :, :])

    trainData = trainDataArr.reshape(-1, 97)
    trainDataNoCorr = trainDataArrNoCorr.reshape(-1, 97)
    testData = testDataArr.reshape(-1, 97)

    trainset = NeuralData(data=trainData, data2=trainDataNoCorr)
    testset = NeuralData(data=testData, data2=testData)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=workers)

    return trainloader, testloader

# get different time windows
def get_neural_time_loader(workers=0, batch_size=10, time1=150, time2=350, deltat=100):

    data = sio.loadmat('data/ps4_realdata.mat')  # load the .mat file.
    NumTrainData = data['train_trial'].shape[0]
    NumClass = data['train_trial'].shape[1]
    NumTestData = data['test_trial'].shape[0]
    trainDataArr = np.zeros((NumClass, NumTrainData, 97))  # contains the firing rates for all neurons on all 8 x 91 trials in the training set
    trainDataArr2 = np.zeros((NumClass, NumTrainData, 97))
    testDataArr = np.zeros((NumClass, NumTestData, 97))  # for the testing set.
    testDataArr2 = np.zeros((NumClass, NumTestData, 97))  # for the testing set.

    for classIX in range(NumClass):
        for trainDataIX in range(NumTrainData):
            trainDataArr[classIX, trainDataIX, :] = np.sum(data['train_trial'][trainDataIX, classIX][1][:, time1:time1 + deltat], 1)
            trainDataArr2[classIX, trainDataIX, :] = np.sum(data['train_trial'][trainDataIX, classIX][1][:, time2:time2 + deltat], 1)

        for testDataIX in range(NumTestData):
            testDataArr[classIX, testDataIX, :] = np.sum(data['test_trial'][testDataIX, classIX][1][:, time1:time1 + deltat], 1)
            testDataArr2[classIX, testDataIX, :] = np.sum(data['test_trial'][testDataIX, classIX][1][:, time2:time2 + deltat], 1)

    trainData = trainDataArr.reshape(-1, 97)
    trainData2 = trainDataArr2.reshape(-1, 97)
    testData = testDataArr.reshape(-1, 97)
    testData2 = testDataArr2.reshape(-1, 97)

    trainset = NeuralData(data=trainData, data2=trainData2)
    testset = NeuralData(data=testData, data2=testData2)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=workers)

    return trainloader, testloader


# CENTER OUT
class NeuralDataCenter(torch.utils.data.Dataset):
    def __init__(self, data, data2, targets):
        self.data = data
        self.data2 = data2
        self.targets = targets
        self.size = data.shape[0]

    def __getitem__(self, index):

        input_data = self.data[index]
        input_data2 = self.data2[index]
        target = self.targets[index]
        return input_data, input_data2, target

    def __len__(self):
        return self.size


# some helper functions
def get_target_class(point, U):
    target_class = -1
    for i, e in enumerate(U):
        if (point == e).all():
            target_class = i
    return target_class


def get_out_indices(data):
    return ~np.all(data == 0, axis=1)


def remove_zeros(data):
    return data[get_out_indices(data)]


# basically, conditioned on the target class, sample data (but in a convenient manner for the dataloader)
def align_data(targets1, targets2):
    target_idx1 = []
    target_idx2 = []
    for i in range(np.max(targets1) + 1):
        idx1 = [idx for idx, val in enumerate(targets1) if val == i]
        idx2 = [idx for idx, val in enumerate(targets2) if val == i]
        min_overlap = min(len(idx1), len(idx2))
        target_idx1.append(idx1[:min_overlap])
        target_idx2.append(idx2[:min_overlap])

    return target_idx1, target_idx2


def test_align_data():
    targets1 = [0, 0, 0, 1, 1]
    targets2 = [0, 0, 1]
    t1, t2 = align_data(targets1, targets2)
    print(t1)
    print(t2)


# TODO: add in time_avg, slightly clean up code
def load_neural_data(path, delay=False, raster='spikeRaster2'):
    data = sio.loadmat(path)
    R = data['R'][0, :]

    # a bit messy code, but this loads the targets and removes the center targets
    t = R[0:]['posTargets1']
    targets = np.zeros((len(t), 2))
    for i in range(len(t)):
        for j in range(2):
            targets[i][j] = t[i][j]
    U = remove_zeros(np.unique(targets, axis=0))

    features = []
    classes = []
    for i, e in enumerate(get_out_indices(targets)):
        if e:
            if delay:
                # For the delay data, spikeRaster2 works a lot better than spikeRaster, 2 is from PMd
                time_end = R[i]['timeTargetOn'].item()  # this is bad naming
                time_start = time_end - R[i]['delayTime'].item()
                features.append(100 * np.mean(R[i][raster][:, time_start:time_end], axis=1))
            else:
                features.append(np.sum(R[i]['spikeRaster'], axis=1))
            classes.append(get_target_class(targets[i], U))
    return features, classes


def load_neural_data_time(path, delay=False, time=0, deltat=100, raster='spikeRaster2'):
    data = sio.loadmat(path)
    R = data['R'][0, :]

    # a bit messy code, but this loads the targets and removes the center targets
    t = R[0:]['posTargets1']
    targets = np.zeros((len(t), 2))
    for i in range(len(t)):
        for j in range(2):
            targets[i][j] = t[i][j]
    U = remove_zeros(np.unique(targets, axis=0))

    features = []
    classes = []
    for i, e in enumerate(get_out_indices(targets)):
        if e:
            if delay:
                # For the delay data, spikeRaster2 works a lot better than spikeRaster, 2 is from PMd
                time_end = R[i]['timeTargetOn'].item()  # this is bad naming
                time_start = time_end - R[i]['delayTime'].item() + time
                features.append(100 * np.mean(R[i][raster][:, time_start:time_start + deltat], axis=1))
            else:
                features.append(np.sum(R[i]['spikeRaster'], axis=1))
            classes.append(get_target_class(targets[i], U))
    return features, classes


def get_overlapped_data(features, classes, idxs):
    features = np.array(features).squeeze()
    trainData = [features[idx] for idx in idxs]
    trainData = np.concatenate(trainData, axis=0)

    classes = np.array(classes).squeeze()
    trainTargets = [classes[idx] for idx in idxs]
    trainTargets = np.concatenate(trainTargets, axis=0)
    return trainData, trainTargets


# TODO clean this method up, and create train/test set from different datasets (as efficiently as possible)
# TODO: in parallel, update truncation script so the data to load is compatible
def get_neural_center_loader(workers=0, batch_size=10, time1=150, time2=350, deltat=100):
    path1 = 'data/center_out_data/Jenkins_R/cleaned/truncated_R_2013-01-15_1.mat'
    #path2 = 'data/center_out_data/Jenkins_R/cleaned/truncated_R_2013-02-15_1.mat'
    path2 = 'data/center_out_data/Larry_R/Cleaned/truncated_R_2013-01-28_1.mat'
    features, classes = load_neural_data(path1)
    features2, classes2 = load_neural_data(path2)
    idxs1, idxs2 = align_data(classes, classes2)
    trainData, trainTargets = get_overlapped_data(features, classes, idxs1)
    trainData2, trainTargets2 = get_overlapped_data(features2, classes2, idxs2)

    test_frac = 0.2
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(trainData, trainTargets, test_size=test_frac, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(trainData2, trainTargets2, test_size=test_frac, random_state=42)
    trainset = NeuralDataCenter(data=X_train, data2=X_train2, targets=y_train)
    testset = NeuralDataCenter(data=X_test, data2=X_test2, targets=y_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers)
    return trainloader, testloader


# Delay Data
def get_neural_delay_loader(workers=0, batch_size=10, time1=150, time2=350, deltat=100, path1=None, path2=None, raster='spikeRaster2', useTime=False):
    # modify raster manually here, should really have raster1 and raster2 as inputs
    if not useTime:
        features, classes = load_neural_data(path1, delay=True, raster=raster)
        features2, classes2 = load_neural_data(path2, delay=True, raster=raster)
    else:
        features, classes = load_neural_data_time(path1, delay=True, raster=raster, time=time1, deltat=100)
        features2, classes2 = load_neural_data_time(path2, delay=True, raster=raster, time=time2, deltat=100)

    idxs1, idxs2 = align_data(classes, classes2)
    trainData, trainTargets = get_overlapped_data(features, classes, idxs1)
    trainData2, trainTargets2 = get_overlapped_data(features2, classes2, idxs2)

    test_frac = 0.1
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(trainData, trainTargets, test_size=test_frac, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(trainData2, trainTargets2, test_size=test_frac, random_state=42)
    trainset = NeuralDataCenter(data=X_train, data2=X_train2, targets=y_train)
    testset = NeuralDataCenter(data=X_test, data2=X_test2, targets=y_test)

    # drop last doesn't use last sample since there are issues with batch norm using only one sample
    # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/3
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=True)
    return trainloader, testloader


if __name__ == "__main__":
    # test_align_data()
    # train_loader, test_loader = get_neural_center_loader()
    train_loader, test_loader = get_neural_delay_loader()
    for i1, i2, t in test_loader:
        print(t)
        sys.exit()
