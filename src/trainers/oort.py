import torch
import math
import torch.nn as nn
import torch.utils.data
import time
from tqdm import tqdm
from src.trainers.base import *


class Client(BaseClient):
    def __init__(self, id, params, dataset):
        super().__init__(id, params, dataset)
        self.classifier_criterion = nn.CrossEntropyLoss()
        self.params = params
        self.meters = {
            'accuracy': AvgMeter(),
            'classifier_loss': AvgMeter(),
        }
        self.last_involved = 0
    
    def local_train(self):
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
        self.sample_loss = meters_classifier_loss.avg()

class Server(BaseServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_delta = 1.0 # self.params['Trainer']['delta']
        self.s_alpha = 1.0 # self.params['Trainer']['alpha']
        self.s_E = []
        self.s_U = {}
        self.s_L = {}
        self.s_D = {}
        self.s_R = 0
        self.s_T = self.s_delta
        self.s_eps = 0.5 # self.params['Trainer']['eps']
        self.s_c = 0.95

    def aggregate_model(self, clients):
        n = len(clients)
        total_batch = torch.tensor(0)
        p_tensors = []
        for _, client in enumerate(clients):
            batch_cnt = torch.tensor(len(client.trainloader))
            p_tensors.append(client.model.parameters_to_tensor() * batch_cnt)
            total_batch.add_(batch_cnt)
        avg_tensor = sum(p_tensors) / total_batch
        self.model.tensor_to_parameters(avg_tensor)
        return
    
    def sample_client(self):
        self.s_R += 1
        util = {}
        # update information
        for client in self.clients:
            if client.last_involved > 0:
                if client not in self.s_E:
                    self.s_E.append(client)
                    self.s_L[client] = client.last_involved
                    self.s_D[client] = 1
                else:
                    self.s_L[client] = client.last_involved
                    if self.s_L[client] == self.s_R - 1:
                        self.s_D[client] += 1
                    else:
                        self.s_D[client] = 0
                l = client.sample_loss
                self.s_U[client] = l # TODO: 这里和oort的计算方法不太一样，oort那个好像不太好写
        for client in self.s_E:
            util[client] = self.s_U[client] + math.sqrt(0.1 * math.log(self.s_R) / self.s_L[client])
            if self.s_T < self.s_D[client]:
                util[client] *= math.pow(self.s_T / self.s_D[client], self.s_alpha)
        util_v = sorted([v for k, v in util.items()], reverse=True)
        n_exploration = math.ceil(self.n_clients_per_round * self.s_eps)
        n_exploitation = math.ceil(self.n_clients_per_round * (1 - self.s_eps))
        p = []
        n_exploitation = min(n_exploitation, len(self.s_E))
        if n_exploitation > 0:
            util_v_cutoff = util_v[n_exploitation - 1] * self.s_c
            w = [k for k, v in util.items() if v > util_v_cutoff]
            w_p = [v for k, v in util.items() if v > util_v_cutoff]
            p = p + random.choices(w, w_p, k=n_exploitation)
        untried_clients = [k for k in self.clients if k not in self.s_E]
        n_exploration = min(n_exploration, len(untried_clients))
        p = p + random.choices(untried_clients, k=n_exploration)
        return p

    def train(self):
        # random clients
        clients = self.sample_client()

        for client in clients:
            # send params
            client.clone_model(self)
            for p in client.optimizer.param_groups:
                p['lr'] = self.learning_rate
        
        for client in clients:
            # local train
            client.local_train()
            client.last_involved = self.s_R
        
        # aggregate params
        self.aggregate_model(clients)

        self.learning_rate *= self.params['Trainer']['optimizer']['lr_decay']
        
        return clients
