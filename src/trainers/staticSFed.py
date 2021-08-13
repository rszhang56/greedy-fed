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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = 0
        self.single_clients = []
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
        # C = self.params['Trainer']['C_weight']
        if self.T == 0:
            self.single_clients = self.sample_client()
            self.T = 1
            n = self.n_clients
            R = self.params['Trainer']['R']
            delta = []

            for client in self.clients:
                # send params
                client.clone_model(self)
                delta.append(client.local_train())

            u = []
            simulation_list = []
            for i in range(R):
                simulation_list.append(np.random.permutation(self.n_clients))
            phi = []
            init_omega = self.model.parameters_to_tensor()
            for j in range(self.n_clients):
                phi_j = 0
                omega = init_omega
                for permutation in simulation_list:
                    cnt = 0
                    for i in permutation:
                        if (i != j):
                            omega += delta[i]
                            cnt += 1
                        else:
                            break
                    if (cnt == 0):
                        continue
                    omega = omega / cnt
                    self.model.tensor_to_parameters(omega)
                    old_acc = self.test_accuracy()
                    omega = (omega * cnt + delta[j]) / (cnt + 1)
                    self.model.tensor_to_parameters(omega)
                    new_acc = self.test_accuracy()
                    phi_j += (new_acc - old_acc) / R
                    self.model.tensor_to_parameters(init_omega)
                phi.append(phi_j)
            sorted_id = sorted(enumerate(phi), key=lambda x: x[1])
            selected_id = sorted_id[-self.n_clients_per_round:]
            selected_clients = [self.clients[i[0]] for i in selected_id]

            self.aggregate_model(selected_clients)

            self.learning_rate *= self.params['Trainer']['optimizer']['lr_decay']

            return selected_clients

        for client in self.single_clients:
            # send params
            client.clone_model(self)
            for p in client.optimizer.param_groups:
                p['lr'] = self.learning_rate

        for client in self.single_clients:
            # local train
            client.local_train()

        # aggregate params
        self.aggregate_model(self.single_clients)

        self.learning_rate *= self.params['Trainer']['optimizer']['lr_decay']
        return self.single_clients

