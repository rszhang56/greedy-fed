import torchvision.datasets
import torchvision.transforms as transforms
import copy
import random

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import Subset

coarse_labels = [ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                  3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                  6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                  0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                  5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                  16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                  10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                  2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13]

root_dir = './data/'

transform = transforms.Compose([
    transforms.Resize((28, 28)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR100(
    root_dir, 
    train=True, 
    transform=transform, 
    download=True
)
test_dataset = torchvision.datasets.CIFAR100(
    root_dir, 
    train=False, 
    transform=transform, 
    download=True
)

def niid_split(train_dataset, test_dataset, s: float, num_user: int, func=(lambda x: x[1])):
    trainset_targets = [(i, func(item)) for i, item in enumerate(train_dataset)]
    random.shuffle(trainset_targets)
    validation_len = 1000
    validation_idx = trainset_targets[:validation_len]
    validation_idx = list(map(lambda x: x[0], validation_idx))
    trainset_targets = trainset_targets[validation_len:]
    class_targets = {}
    for i in range(100):
        class_targets[i] = list(filter(lambda x: x[1] == i, trainset_targets))
    
    coarse_labels_inv = []
    for i in range(20): coarse_labels_inv.append([])
    for i in range(100):
        coarse_labels_inv[coarse_labels[i]].append(i)
    
    user_labels = []
    for i in range(num_user):
        user_label = np.zeros((100), dtype=np.int32)
        for j in range(20):
            k = random.choice(coarse_labels_inv[j])
            user_label[k] = 1
        user_labels.append(user_label)
    print(user_labels)
    
    user_labels_sum = np.sum(user_labels, axis=0)
    print(user_labels_sum)

    class_idx = [0 for i in range(100)]
    class_delta = [len(class_targets[i]) // user_labels_sum[i] for i in range(100)]
    print(class_delta)

    dataset_split = []
    for i in range(num_user):
        train_idx = []
        for j in range(100):
            if user_labels[i][j] == 1:
                train_idx.extend(class_targets[j][
                    class_idx[j]: class_idx[j] + class_delta[j]
                ])
                class_idx[j] += class_delta[j]
        train_idx = list(map(lambda x: x[0], train_idx))
        # print(user_labels[i])
        # print(train_idx)
        dataset_split.append(
            {
                'train': Subset(train_dataset, train_idx),
                'test': None,
                'validation': Subset(train_dataset, validation_idx), 
            }
        )
    return dataset_split

def merge_target(dataset):
    for idx in dataset.indices:
        dataset.dataset.targets[idx] = coarse_labels[dataset.dataset.targets[idx]]

def niid(params):
    num_user = params['Trainer']['n_clients']
    testset_dict = {
        'train': None,
        'test': test_dataset,
        'validation': None, 
    }
    dataset_split = niid_split(train_dataset, test_dataset, -1, num_user)
    testset_dict['validation'] = dataset_split[0]['validation']
    merge_target(testset_dict['validation'])
    for i, label in enumerate(test_dataset.targets):
        test_dataset.targets[i] = coarse_labels[label]
    for i, dataset in enumerate(dataset_split):
        merge_target(dataset['train'])
    return dataset_split, testset_dict

