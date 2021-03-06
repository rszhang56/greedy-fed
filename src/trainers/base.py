import torch
import random
import importlib
import numpy as np
import torch.optim as optim
import torch.nn as nn
import time
import sys
import yaml
from tqdm import tqdm
from src.trainers.utils import *
from collections import OrderedDict
import os

# import moxing as mox


class BaseClient():
    def __init__(self, id, params, dataset):
        self.batch_size = params['Trainer']['batch_size']
        self.trainset = dataset['train']
        self.testset = dataset['test']
        self.validation = dataset['validation']
        self.id = id
        collate_fn = None
        dataset_type = 'Image'
        if 'type' in params['Dataset'] and params['Dataset']['type'] == 'NLP':
            dataset_type = 'NLP'
        if dataset_type == 'NLP':
            collate_fn = nlp_collate_fn
        if self.trainset != None:
            self.trainloader = torch.utils.data.DataLoader(
                self.trainset,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
                collate_fn=collate_fn,
            )
        else:
            self.trainloader = None
        if self.testset != None:
            self.testloader = torch.utils.data.DataLoader(
                self.testset,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=True,
                collate_fn=collate_fn,
            )
        else:
            self.testloader = None
        if self.validation != None:
            self.valloader = torch.utils.data.DataLoader(
                self.validation,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=True,
                collate_fn=collate_fn,
            )
        else:
            self.valloader = None
        self.E = params['Trainer']['E']
        self.device = torch.device(params['Trainer']['device'])
        models = importlib.import_module('src.models')
        self.model = eval('models.%s' % params['Model']['name'])(params)
        if dataset_type == 'NLP':
            self.model.embedding.weight.data.copy_(dataset['vocab'].vectors)
        self.model = self.model.to(self.device)
        self.optimizer = eval('optim.%s' % params['Trainer']['optimizer']['name'])(
            self.model.parameters(),
            **params['Trainer']['optimizer']['params'],
        )

    def local_train(self):
        raise NotImplementedError()

    def clone_model(self, target):
        p_tensor = target.model.parameters_to_tensor()
        self.model.tensor_to_parameters(p_tensor)
        return

    def test_accuracy(self, val=False, batch=-1, acc=True):
        dl = self.testloader
        if val: dl = self.valloader
        if dl == None: return -1
        correct = 0
        total = 0
        loss = 0.0
        batch_count = 0
        classifier_criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i, data in enumerate(dl):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                classifier_loss = classifier_criterion(
                    outputs,
                    labels,
                )
                loss += classifier_loss.item()
                batch_count += 1
                if i >= batch and batch >= 0: break
        accuracy = correct / total
        loss = loss / batch_count
        if acc:
            return accuracy
        else:
            return loss

    def get_features_and_labels(self, train=True, batch=-1):
        dataloader = None
        if train:
            dataloader = self.trainloader
        else:
            dataloader = self.testloader
        features_batch = []
        labels_batch = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                if i == batch: break
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                _, f_s = self.model(inputs, features=True)
                features_batch.append(f_s)
                labels_batch.append(labels)
        features = torch.cat(features_batch)
        labels = torch.cat(labels_batch)
        return features, labels

    def save_features_and_labels(self, fn, train=True, batch=-1):
        features, labels = self.get_features_and_labels(train, batch)
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        np.save('%s_features.npy' % fn, features)
        np.save('%s_labels.npy' % fn, labels)
        return


class BaseServer(BaseClient):
    def __init__(self, id, params, dataset):
        super().__init__(id, params, dataset)
        self.n_clients = params['Trainer']['n_clients']
        self.n_clients_per_round = round(params['Trainer']['C'] * self.n_clients)
        self.learning_rate = params['Trainer']['optimizer']['params']['lr']
        self.params = params

    def aggregate_model(self):
        raise NotImplementedError()

    def train(self):
        # finish 1 comm round
        raise NotImplementedError()

    def sample_client(self):
        return random.sample(
            self.clients,
            self.n_clients_per_round,
        )

    def select_client(self):
        select_num = min(self.n_clients_per_round, len(self.clients))
        return self.clients[:select_num]


class Trainer():
    def __init__(self, config):
        # set seed
        set_seed(config['Trainer']['seed'])
        # import module
        trainer_module = importlib.import_module(
            'src.trainers.%s' % config['Trainer']['name']
        )
        dataset_module = importlib.import_module(
            'src.data.%s' % config['Dataset']['name']
        )
        # init meters
        self.meters = {
            'accuracy': AvgMeter(),
            'clients': {},
        }
        # get dataset
        dataset_func = eval('dataset_module.%s' % config['Dataset']['divide'])
        dataset_split, testset = dataset_func(config)
        # init clients
        self.clients = []
        for i in range(config['Trainer']['n_clients']):
            id = i + 1
            client = eval('trainer_module.Client')(
                id,
                config,
                dataset_split[i],
            )

            self.clients.append(client)
        # init server
        self.server = eval('trainer_module.Server')(0, config, testset)
        self.server.clients = self.clients
        # save config
        self.config = config

    # def greedy_select(self, lazy_list, output):
    #     output.write('==========selection begin==========\n')
    #     old_parameters = self.server.model.parameters_to_tensor()
    #     time_begin = time.time()
    #     # local client train their model E epoch
    #     for client in self.server.clients:
    #         client.clone_model(self.server)
    #         client.local_train(self.server.params['Trainer']['E_select'])
    #     time_end = time.time()
    #     output.write('local train time: %.0f seconds\n' % (time_end - time_begin))

    #     time_begin = time.time()
    #     select_num = self.server.n_clients_per_round
    #     selected_clients = []
    #     unselect_clients = self.server.clients
    #     if lazy_list == []:
    #         lazy_list = [[c, 10] for c in unselect_clients]
    #     lazy_list = [[c, 10] for c in unselect_clients]

    #     for j in range(select_num):
    #         self.server.clients = selected_clients
    #         best_client = lazy_list[-1][0]
    #         unselect_lazylist = []
    #         old_test_acc = self.server.test_accuracy(val=True, batch=200)
    #         for i in range(len(lazy_list)):
    #             client = lazy_list[i][0]
    #             if client in selected_clients:
    #                 continue
    #             selected_clients.append(client)
    #             self.server.aggregate_model(selected_clients)
    #             selected_clients.remove(client)
    #             new_test_acc = self.server.test_accuracy(val=True, batch=200)
    #             lazy_list[i][1] = min(lazy_list[i][1], new_test_acc - old_test_acc)
    #             unselect_lazylist.append(lazy_list[i])
    #             if(i != len(lazy_list) - 1 and lazy_list[i][1] >= lazy_list[i+1][1]):
    #                 best_client = client
    #                 break
    #         if best_client ==  lazy_list[-1][0]:
    #             unselect_lazylist.sort(key=lambda x: x[1], reverse=True)
    #             best_client = unselect_lazylist[0][0]
    #         selected_clients.append(best_client)
    #         unselect_clients.remove(best_client)
    #         lazy_list.sort(key=lambda x : x[1], reverse=True)
    #     '''
    #     for i in range(select_num):
    #         best_client = self.server.clients[-1]
    #         max_acc = 0.0
    #         for client in unselect_clients:
    #             selected_clients.append(client)
    #             self.server.aggregate_model(selected_clients)
    #             selected_clients.remove(client)
    #             if self.server.test_accuracy() >= max_acc:
    #                 max_acc = self.server.test_accuracy()
    #                 best_client = client
    #         selected_clients.append(best_client)
    #         unselect_clients.remove(best_client)
    #     '''
    #     for client in unselect_clients:
    #         selected_clients.append(client)

    #     time_end = time.time()
    #     self.clients = selected_clients
    #     self.server.clients = selected_clients
    #     self.server.aggregate_model(selected_clients)
    #     output.write('==========selection end==========\n')
    #     # output.write('server, accuracy: %.5f\n' % self.server.test_accuracy())
    #     output.write('selection time: %.0f seconds\n' % (time_end - time_begin))
    #     self.server.model.tensor_to_parameters(old_parameters)
    #     return selected_clients, lazy_list

    def greedy_select(self, lazy_list, output, acc=True):
        output.write('==========selection begin==========\n')
        old_parameters = self.server.model.parameters_to_tensor()
        time_begin = time.time()
        # local client train their model E epoch
        for client in self.server.clients:
            client.clone_model(self.server)
            client.local_train(self.server.params['Trainer']['E_select'])
        time_end = time.time()
        output.write('local train time: %.0f seconds\n' % (time_end - time_begin))

        time_begin = time.time()
        select_num = self.server.n_clients_per_round
        selected_clients = []
        unselect_clients = self.server.clients
        lazy_list = [[c, 10] for c in unselect_clients]

        # f = open('select_round%d.txt' % self.iter_round, 'w')
        for j in range(select_num):
            self.server.clients = selected_clients
            best_client = None
            max_gain = -10
            unselect_lazylist = []
            old_test = 0.0
            if acc:
                old_test = self.server.test_accuracy(val=True, batch=200)
                if j == 0: old_test = 0
            else:
                old_test = self.server.test_accuracy(val=True, batch=200, acc=False)
                if j == 0: old_test = 10
            for i in range(len(lazy_list)):
                client = lazy_list[i][0]
                if client in selected_clients:
                    continue
                selected_clients.append(client)
                self.server.aggregate_model(selected_clients)
                selected_clients.remove(client)
                gain = 0
                if acc:
                    gain = self.server.test_accuracy(val=True, batch=200) - old_test
                else:
                    gain = old_test - self.server.test_accuracy(val=True, batch=200, acc=False)
                if j == 0:
                    lazy_list[i][1] = gain
                else:
                    lazy_list[i][1] = 0.8 * lazy_list[i][1] + 0.2 * gain
                unselect_lazylist.append(lazy_list[i])
                if gain > max_gain:
                    max_gain = gain
                    best_client = client
                print('selecting %d client, clientid: %d, gain: %f' % (j, client.id, gain))
                if (i != len(lazy_list) - 1 and max_gain >= lazy_list[i + 1][1]):
                    break
            if best_client == None:
                unselect_lazylist.sort(key=lambda x: x[1], reverse=True)
                best_client = unselect_lazylist[0][0]
            selected_clients.append(best_client)
            unselect_clients.remove(best_client)
            self.server.aggregate_model(selected_clients)
            gain = self.server.test_accuracy(val=True, batch=200) - old_test
            print('client id: %d is selected' % best_client.id)
            # f.write('client id: %d, max gain: %f, gain: %f\n' % (best_client.id, max_gain, gain))
            lazy_list.sort(key=lambda x: x[1], reverse=True)
        for client in unselect_clients:
            selected_clients.append(client)
        # f.close()
        time_end = time.time()
        self.clients = selected_clients
        self.server.clients = selected_clients
        self.server.aggregate_model(selected_clients)
        output.write('==========selection end==========\n')
        output.write('selection time: %.0f seconds\n' % (time_end - time_begin))
        self.server.model.tensor_to_parameters(old_parameters)
        return selected_clients, lazy_list

    def train(self):
        output = sys.stdout
        lazy_list = []
        if 'Output' in self.config:
            os.makedirs(os.path.dirname(self.config['Output']), exist_ok=True)
            output = open(self.config['Output'], 'a')
        # if 'Output' in self.config: output = mox.file.File('obs://fed/fed-selection/code/result/' + self.config['Output'], 'a')
        output.write(yaml.dump(self.config, Dumper=yaml.Dumper))
        # greedy algorithm: select the best clients by greedy strategy
        self.iter_round = 0
        acc = True
        if self.config['Trainer']['evaluation'] == 'loss':
            acc = False
        if self.config['Trainer']['name'] == "greedyFed" or self.config['Trainer']['name'] == "greedyFed+":
            for client in self.server.clients:
                client.clone_model(self.server)
                client.local_train(self.server.params['Trainer']['E'])
            self.server.aggregate_model(self.server.clients)
            selected_clients, lazy_list = self.greedy_select([], output, acc)
        try:
            for round in tqdm(range(self.config['Trainer']['Round']), desc='Communication Round', leave=False):
                self.iter_round = round
                output.write('==========Round %d begin==========\n' % round)
                time_begin = time.time()
                # C_t = self.config['Trainer']['Round'] - (round + 1)
                # if C_t & (C_t - 1) == 0 and self.config['Trainer']['name'] == "greedyFed+":
                if round > 5 and (round + 1) % 1 == 0 and self.config['Trainer']['name'] == "greedyFed+" and self.meters['accuracy'].last() <= self.meters['accuracy'].avg(-5):
                    clients, lazy_list = self.greedy_select(lazy_list, output, acc)
                clients = self.server.train()
                self.meters['accuracy'].append(self.server.test_accuracy())
                time_end = time.time()
                for client in sorted(clients, key=lambda x: x.id):
                    client_summary = []
                    client_summary.append('client %d' % client.id)
                    for k, v in client.meters.items():
                        client_summary.append('%s: %.5f' % (k, v.last()))
                    output.write(', '.join(client_summary) + '\n')
                output.write('server, accuracy: %.5f\n' % self.meters['accuracy'].last())
                output.write('total time: %.0f seconds\n' % (time_end - time_begin))
                output.write('==========Round %d end==========\n' % round)
                output.flush()
        except KeyboardInterrupt:
            ...
        finally:
            acc_lst = self.meters['accuracy'].data
            avg_count = 5
            acc_avg = np.mean(acc_lst[-avg_count:])
            acc_std = np.std(acc_lst[-avg_count:])
            acc_max = np.max(acc_lst)
            output.write('==========Summary==========\n')
            for client in self.clients:
                client.clone_model(self.server)
                output.write('client %d, accuracy: %.5f\n' % (client.id, client.test_accuracy()))
            output.write('serve, max accuracy: %.5f\n' % acc_max)
            output.write('serve, final accuracy: %.5f +- %.5f\n' % (acc_avg, acc_std))
            output.write('===========================\n')
