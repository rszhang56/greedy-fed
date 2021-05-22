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
    shuffle_idx = random.sample(dataset.indices, percent_sample)
    for idx in shuffle_idx:
        dataset.dataset.targets[idx] = random.choice(
            list(dataset.dataset.class_to_idx.values()), 
        )

def niid(params):
    num_user = params['Trainer']['n_clients']
    s = params['Dataset']['s']
    testset_dict = {
        'train': None,
        'test': copy.deepcopy(test_dataset),
    }
    r = random.random()
    dataset_split = split_dataset_by_percent(train_dataset, test_dataset, s, num_user)
    for dataset in dataset_split:
        r = random.random()
        if r < 0.8:
            shuffle_target(dataset['train'], 0.2)
            shuffle_target(dataset['test'], 0.2)
    return dataset_split, testset_dict