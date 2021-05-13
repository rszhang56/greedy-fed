import pandas as pd
import torch
import json
from torch.utils.data import Dataset
from itertools import chain
#import moxing as mox



class synthetic_dataset(Dataset):
    def __init__(self, features, labels, transform):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.transform(self.features[idx]), self.transform(self.labels[idx])

def niid(params):
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    # data load
    train_file = "./data/synthetic/train/"+params['Dataset']['class']+".json"
    test_file = "./data/synthetic/test/"+params['Dataset']['class']+".json"

    #with mox.file.File('obs://fed/fed-selection/code/data/synthetic/train/'+params['Dataset']['class']+".json", 'r') as inf:
    with open(train_file, 'r') as inf:
        cdata = json.load(inf)
    clients.extend(cdata['users'])
    train_data.update(cdata['user_data'])

    #with mox.file.File('obs://fed/fed-selection/code/data/synthetic/test/'+params['Dataset']['class']+".json", 'r') as inf:
    with open(test_file, 'r') as inf:
        cdata = json.load(inf)
    test_data.update(cdata['user_data'])
    clients = list(sorted(train_data.keys()))

    test_data_all = {'x':[], 'y':[]}
    dataset_split = []
    for client in clients:
        client_label = map(int, train_data[client]['y'])
        dataset_split.append(
            {
                'train': synthetic_dataset(train_data[client]['x'], list(client_label), transform=torch.tensor),
                'test': synthetic_dataset(test_data[client]['x'], list(map(int, test_data[client]['y'])), transform=torch.tensor),
            }
        )
        for e in test_data[client]['x']:
            test_data_all['x'].append(e)
        test_data_all['y'].append(test_data[client]['y'])
        '''
        if(params['Dataset']['user'] == 500 and (client[-1] == '0' or client[-1] == '4')):
            for e in test_data[client]['x']:
                test_data_all['x'].append(e)
            test_data_all['y'].append(test_data[client]['y'])

        if(params['Dataset']['user'] == 20 and int(client[-1])%2 == 0):
            for e in test_data[client]['x']:
                test_data_all['x'].append(e)
            test_data_all['y'].append(test_data[client]['y'])
        '''
    test_label = map(int, list(chain.from_iterable(test_data_all['y'])))
    testset_dict = {
        'train': None,
        'test': synthetic_dataset(test_data_all['x'], list(test_label), transform=torch.tensor),
    }
    return dataset_split, testset_dict

