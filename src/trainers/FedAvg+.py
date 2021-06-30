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
        self.selected_client = []

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
        
        # aggregate params
        self.aggregate_model(clients)

        self.learning_rate *= self.params['Trainer']['optimizer']['lr_decay']
        
        return clients
    
    def sample_client(self):
        if len(self.selected_client) >= 250:
            return random.sample(self.selected_client, self.n_clients_per_round)

        random_client = random.sample(self.clients, self.n_clients_per_round)
        union_client = list(set(self.selected_client).union(set(random_client)))
        if len(union_client) >= 250:
            self.selected_client = union_client[0:250]
            return random.sample(self.selected_client, self.n_clients_per_round)
        else:
            self.selected_client = union_client
            return random_client


        
