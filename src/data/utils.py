import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import Subset

def split_dataset_by_percent(train_dataset, test_dataset, s: float, num_user: int, func=(lambda x: x[1])):
    trainset_targets = [(i, func(item)) for i, item in enumerate(train_dataset)]
    testset_targets = [(i, func(item)) for i, item in enumerate(test_dataset)]
    random.shuffle(trainset_targets)
    random.shuffle(testset_targets)
    validation_len = 1000
    validation_idx = trainset_targets[:validation_len]
    validation_idx = list(map(lambda x: x[0], validation_idx))
    trainset_targets = trainset_targets[validation_len:]
    len_train_iid = round(s * len(train_dataset))
    len_test_iid = round(s * len(test_dataset))
    trainset_iid_idx = trainset_targets[:len_train_iid]
    trainset_niid_idx = trainset_targets[len_train_iid:]
    testset_iid_idx = testset_targets[:len_test_iid]
    testset_niid_idx = testset_targets[len_test_iid:]
    trainset_niid_idx = sorted(trainset_niid_idx, key=lambda x: x[1])
    testset_niid_idx = sorted(testset_niid_idx, key=lambda x: x[1])
    p_train_iid = 0
    p_train_niid = 0
    p_test_iid = 0
    p_test_niid = 0

    delta_list = np.random.lognormal(4, 1, (num_user)).astype(int) + 10
    delta_list = delta_list / delta_list.sum()
    delta_list.sort()
    delta_list = delta_list[::-1]
    '''
    delta_train_iid = len(trainset_iid_idx) // num_user
    delta_train_niid = len(trainset_niid_idx) // num_user
    '''
    delta_test_iid = len(testset_iid_idx) // num_user
    delta_test_niid = len(testset_niid_idx) // num_user

    dataset_split = []
    for i in range(num_user):
        train_idx = []
        test_idx = []
        delta_train_iid = int(delta_list[i] * len(trainset_iid_idx))
        delta_train_niid = int(delta_list[i] * len(trainset_niid_idx))
        #delta_test_iid = int(delta_list[i] * len(testset_iid_idx))
        #delta_test_niid = int(delta_list[i] * len(testset_niid_idx))
        if delta_train_iid > 0:
            train_idx.extend(
                trainset_iid_idx[
                    p_train_iid: p_train_iid + delta_train_iid
                ]
            )
        if delta_train_niid > 0:
            train_idx.extend(
                trainset_niid_idx[
                    p_train_niid: p_train_niid + delta_train_niid
                ]
            )
        if delta_test_iid > 0:
            test_idx.extend(
                testset_iid_idx[
                    p_test_iid: p_test_iid + delta_test_iid
                ]
            )
        if delta_test_niid > 0:
            test_idx.extend(
                testset_niid_idx[
                    p_test_niid: p_test_niid + delta_test_niid
                ]
            )
        train_idx = list(map(lambda x: x[0], train_idx))
        test_idx = list(map(lambda x: x[0], test_idx))
        dataset_split.append(
            {
                'train': Subset(train_dataset, train_idx),
                'test': Subset(test_dataset, test_idx),
                'validation': Subset(train_dataset, validation_idx), 
            }
        )
        p_train_iid += delta_train_iid
        p_train_niid += delta_train_niid
        p_test_iid += delta_test_iid
        p_test_niid += delta_test_niid
    print("ok")
    return dataset_split

def noise_split(train_dataset, test_dataset, s: float, num_user: int, func=(lambda x: x[1])):
    trainset_targets = [(i, func(item)) for i, item in enumerate(train_dataset)]
    random.shuffle(trainset_targets)
    validation_len = 1000
    validation_idx = trainset_targets[:validation_len]
    validation_idx = list(map(lambda x: x[0], validation_idx))
    trainset_targets = trainset_targets[validation_len:]
    class_targets = {}
    for i in range(10):
        class_targets[i] = list(filter(lambda x: x[1] == i, trainset_targets))
    class_merge_targets = []
    class_merge_targets.append(class_targets[0] + class_targets[1])
    class_merge_targets.append(class_targets[2] + class_targets[3])
    class_merge_targets.append(class_targets[4] + class_targets[5])
    class_merge_targets.append(class_targets[6] + class_targets[7])
    class_merge_targets.append(class_targets[8] + class_targets[9])

    for i in range(5):
        random.shuffle(class_merge_targets[i])

    dataset_split = []
    for i in range(5):
        count_ds = len(class_merge_targets[i]) // 100
        p_train = 0
        for j in range(100):
            train_idx = class_merge_targets[i][
                p_train: p_train + count_ds
            ]
            train_idx = list(map(lambda x: x[0], train_idx))
            dataset_split.append(
                {
                    'train': Subset(train_dataset, train_idx),
                    'test': None,
                    'validation': Subset(train_dataset, validation_idx), 
                }
            )
            p_train += count_ds
    print("ok")
    return dataset_split
