import torch
import torch.nn as nn
import torch.utils.data
import time
from tqdm import tqdm
from src.trainers.base import *
from scipy.special import comb


class Client(BaseClient):
    def __init__(self, id, params, dataset):
        super().__init__(id, params, dataset)
        self.classifier_criterion = nn.CrossEntropyLoss()
        self.params = params
        self.meters = {
            'accuracy': AvgMeter(),
            'classifier_loss': AvgMeter(),
        }
    
    def local_train(self):
        omega = self.model.parameters_to_tensor()
        meters_classifier_loss = AvgMeter()
        for epoch in range(self.E):
            for i, data in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                classifier_loss = self.classifier_criterion(
                    outputs,
                    labels,
                )
                classifier_loss.backward()
                self.optimizer.step()
                meters_classifier_loss.append(classifier_loss.item())
        self.meters['accuracy'].append(self.test_accuracy())
        self.meters['classifier_loss'].append(meters_classifier_loss.avg())
        return self.model.parameters_to_tensor() - omega

class Server(BaseServer):
    def aggregate_model(self, clients):
        n = len(clients)
        p_tensors = []
        for _, client in enumerate(clients):
            p_tensors.append(client.model.parameters_to_tensor())
        avg_tensor = sum(p_tensors) / n
        self.model.tensor_to_parameters(avg_tensor)
        return
    
    def get_model(self, gradients, subset_bin):
        sum_size = 0
        tmp = subset_bin
        cid = 0
        while tmp > 0:
            if (tmp & 1) > 0: sum_size += 1
            tmp = (tmp >> 1)
            cid += 1
        omega = self.model.parameters_to_tensor()
        tmp = subset_bin
        cid = 0
        while tmp > 0:
            if (tmp & 1) > 0: omega += gradients[cid] * torch.tensor(1 / sum_size)
            tmp = (tmp >> 1)
            cid += 1
        return omega

    def train(self):
        # random clients
        C = self.params['Trainer']['C_weight']
        n = self.n_clients
        delta = []

        for client in self.clients:
            # send params
            client.clone_model(self)
            delta.append(client.local_train())
        
        u = []
        for i in range(1 << self.n_clients):
            self.model.tensor_to_parameters(self.get_model(delta, i))
            u.append(self.test_accuracy())
        
        def count(x):
            ret = 0
            while x > 0:
                if x & 1: ret += 1
                x = (x >> 1)
            return ret
        
        phi = []
        for i in range(self.n_clients):
            phi_i = 0
            for j in range(1 << self.n_clients):
                if (j & (1 << i)) > 0: continue
                phi_i += (u[j | (1 << i)] - u[j]) / comb(n - 1, count(j))
            phi_i *= C
            phi.append(phi_i)
        print(phi)
        
        sorted_id = sorted(enumerate(phi), key=lambda x: x[1])
        selected_id = sorted_id[-self.n_clients_per_round:]
        selected_clients = [self.clients[i[0]] for i in selected_id]

        self.aggregate_model(selected_clients)

        self.learning_rate *= self.params['Trainer']['optimizer']['lr_decay']
        
        return selected_clients
