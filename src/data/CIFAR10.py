import torchvision.datasets
import torchvision.transforms as transforms
import copy
from src.data.utils import *
import random

root_dir = './data/'

transform = transforms.Compose([
    transforms.Resize((28, 28)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root_dir, 
    train=True, 
    transform=transform, 
    download=True
)
test_dataset = torchvision.datasets.CIFAR10(
    root_dir, 
    train=False, 
    transform=transform, 
    download=True
)

def shuffle_target(dataset, percent_sample):
    dataset_len = len(dataset.indices)
    shuffle_target_num = int(dataset_len * percent_sample)
    shuffle_idx = random.sample(dataset.indices, shuffle_target_num)
    for idx in shuffle_idx:
        dataset.dataset.targets[idx] = random.choice(
            list(dataset.dataset.class_to_idx.values()), 
        )

def remap_target(dataset, percent_sample, target_map):
    dataset_len = len(dataset.indices)
    shuffle_target_num = int(dataset_len * percent_sample)
    shuffle_idx = random.sample(dataset.indices, shuffle_target_num)
    for idx in shuffle_idx:
        dataset.dataset.targets[idx] = target_map[dataset.dataset.targets[idx]]

def niid(params):
    num_user = params['Trainer']['n_clients']
    s = params['Dataset']['s']
    testset_dict = {
        'train': None,
        'test': copy.deepcopy(test_dataset),
        'validation': None, 
    }
    r = random.random()
    dataset_split = noise_split(train_dataset, test_dataset, s, num_user)
    testset_dict['validation'] = dataset_split[0]['validation']
    for i, dataset in enumerate(dataset_split):
    #   j = i % 100
        if 0 == i % 2:
            mp = {}
            for i in range(10):
                mp[i] = (i + 1) % 10
            remap_target(dataset['train'], params['Dataset']['noise_client_percent'], mp)
    print("add noise ... (clean)")
    return dataset_split, testset_dict
