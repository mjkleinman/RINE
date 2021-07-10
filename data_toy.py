import torch
from torch.utils.data import Dataset, DataLoader
import pdb
import random

class Toy_dataset(Dataset):

    def __init__(self, num_samples=10000, operation=''):
        super(Toy_dataset, self).__init__()
        self.num_samples = num_samples

        self.X1 = torch.randint(0, 2, (num_samples, 1)).long()
        self.X2 = torch.randint(0, 2, (num_samples, 1)).long()

        # # XOR
        if operation == 'xor':
            self.out = torch.logical_xor(self.X1, self.X2).long()

        # # AND
        elif operation == 'and':
            self.out = torch.logical_and(self.X1, self.X2).long()

        # #  UNQ
        elif operation == 'unq':
            self.out = torch.zeros_like(self.X1)
            for i in range(self.num_samples):
                if self.X1[i] == 0 and self.X2[i] == 0:
                    self.out[i] = 0
                if self.X1[i] == 0 and self.X2[i] == 1:
                    self.out[i] = 1
                if self.X1[i] == 1 and self.X2[i] == 0:
                    self.out[i] = 2
                if self.X1[i] == 1 and self.X2[i] == 1:
                    self.out[i] = 3

        # IMPERFECTRDN
        elif operation == 'imperfectrdn':
            self.X2 = torch.clone(self.X1)
            for i in range(self.num_samples):
                if self.X1[i] == 0:
                    if random.random() > 0.998:
                        self.X2[i] = 1
            self.out = torch.clone(self.X1)

        # RDNXOR
        elif operation == 'rdnxor':
            self.out = torch.logical_xor(self.X1, self.X2).long()
            for i in range(self.num_samples):
                if random.random() > 0.5:
                    self.X1[i] += 2
                    self.X2[i] += 2
                    self.out[i] += 2
        else:
            raise ValueError("Operation {} not valid.".format(operation))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.X1[index].squeeze(), self.X2[index].squeeze(), self.out[index].squeeze()

def get_toy_dataset_loader(batch_size=128, num_data_samples=10000, operation=''):

    dataset = Toy_dataset(num_samples=num_data_samples, operation=operation)
    train_size = int(0.8 * num_data_samples)
    test_size = num_data_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)
    return trainloader, testloader

# TODO: extend this so inputs are  higher dimensional
class Redundant_Dataset(Dataset):
    def __init__(self, operation='', num_classes=8, num_classes_per_input=8, num_samples=10000, num_append=2):
        super(Redundant_Dataset, self).__init__()
        self.num_samples = num_samples
        self.out = (torch.arange(self.num_samples) % num_classes).long()
        # self.X1 = torch.ones_like(self.out) * -1
        # self.X2 = torch.ones_like(self.out) * -1

        # have two inputs that are common for cosine distance to make sense
        num_task_inputs = 1
        self.X1 = torch.randint(0, num_classes, (num_samples, num_task_inputs)).long() - num_classes // 2
        self.X2 = torch.randint(0, num_classes, (num_samples, num_task_inputs)).long() - num_classes // 2
        inds_x1 = self.out < num_classes_per_input
        inds_x2 = self.out >= (num_classes - num_classes_per_input)  # num_classes - (num_classes - num_cl_per_inp)
        self.X1[inds_x1, :] = (self.out[inds_x1]).reshape(-1, 1)
        self.X2[inds_x2, :] = (self.out[inds_x2]).reshape(-1, 1)
        self.X1 = self.X1.reshape(-1, num_task_inputs)
        self.X2 = self.X2.reshape(-1, num_task_inputs)

        # Here X3 is a common factor that doesnt affect the output, hence I'm just multiplying it by a constant
        # if the scale of X3 is larger than X1 or X2 (the components affecting the output), high similarity but low redundant
        if operation == 'append_common':
            if num_append > 0:
                self.X3 = torch.ones((num_samples, num_append)) * - 1 * 2
                self.X1 = torch.cat((self.X1, self.X3.reshape(-1, num_append)), dim=1)
                self.X2 = torch.cat((self.X2, self.X3.reshape(-1, num_append)), dim=1)

        # In the other case, where we have high redundant information but low distance metric, we could have a random common component that is task independent, with a shared task-component.
        elif operation == 'append_noise':
            if num_append > 0:
                # self.X3 = torch.randint(-num_classes // 4, num_classes // 4, (num_samples, num_append)).long()
                # self.X4 = torch.randint(-num_classes // 4, num_classes // 4, (num_samples, num_append)).long()
                self.X3 = torch.randn(num_samples, num_append) * 2
                self.X4 = torch.randn(num_samples, num_append) * 2
                self.X1 = torch.cat((self.X1, self.X3.reshape(-1, num_append)), dim=1)
                self.X2 = torch.cat((self.X2, self.X4.reshape(-1, num_append)), dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.X1[index].squeeze(), self.X2[index].squeeze(), self.out[index].squeeze()


def get_redundant_dataset_loader(batch_size=128, num_data_samples=10000, operation='', num_classes=8, num_classes_per_input=6, num_append=2):
    dataset = Redundant_Dataset(operation, num_classes, num_classes_per_input, num_samples=num_data_samples, num_append=num_append)
    train_size = int(0.7 * num_data_samples)
    test_size = num_data_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)
    return trainloader, testloader


if __name__ == '__main__':
    import sys
    trainloader, testloader = get_toy_dataset_break_corr_loader(operation='xor')
    for x1, y in trainloader:
        print(x1)
        print(y)
        sys.exit()
