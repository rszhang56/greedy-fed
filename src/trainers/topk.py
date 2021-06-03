import torch
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
    
    def local_train(self,num_epoch):
        meters_classifier_loss = AvgMeter()
        for epoch in range(num_epoch):
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

class Server(BaseServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_clients = []

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
        if len(self.selected_clients) == 0:
            print('----- select begin -----')
            acc = []
            for client in self.clients:
                client.clone_model(self)
                client.local_train(self.params['Trainer']['E_select'])
                tmp_acc = client.test_accuracy(val=True, batch=200)
                acc.append((client, tmp_acc))
            acc.sort(key=lambda x: x[1], reverse=True)
            for i in range(self.n_clients_per_round):
                self.selected_clients.append(acc[i][0])
            print('----- select end -----')

        for client in self.selected_clients:
            # send params
            client.clone_model(self)
            for p in client.optimizer.param_groups:
                p['lr'] = self.learning_rate
        
        for client in self.selected_clients:
            # local train
            client.local_train(self.E)
        
        # aggregate params
        self.aggregate_model(self.selected_clients)

        self.learning_rate *= self.params['Trainer']['optimizer']['lr_decay']
        
        return self.selected_clients
