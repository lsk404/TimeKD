#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset,Subset
import numpy as np
import random
from sklearn import metrics
import time

class LocalDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (x, y, emb) = self.dataset[index]
        return (x, y, emb)
    def get_fractional_dataloader(self, args,shuffle=True):
        frac = args.local_data_frac
        batch_size = args.batch_size
        num_workers = args.num_workers
        if frac >= 1.0:
            indices = list(range(len(self.dataset)))
        else:
            num_samples = int(frac * len(self.dataset))
            indices = random.sample(range(len(self.dataset)), num_samples)
        subset = Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=num_workers)


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None,shared_dataLoader=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.dataset = LocalDataset(dataset)
        self.shared_dataLoader = shared_dataLoader

    def train(self, local_model,global_shared_result): # local_model belongs to class 'trainer'
        # global_shared_result: A Dictionary storing the output results of the global model on shared data
        # train and update
        self.dataLoader_train = self.dataset.get_fractional_dataloader(self.args)
        epoch_loss = []
        epoch_mse = []
        epoch_mae = []
        device = torch.device(self.args.device)
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_mse = []
            batch_mae = []
            batch_grad_norm = []
            for batch_idx, (x, y, emb) in enumerate(self.dataLoader_train):
                trainx = torch.Tensor(x).to(device).float()
                trainy = torch.Tensor(y).to(device).float()
                emb = torch.Tensor(emb).to(device).float()
                metrics = local_model.train(trainx, trainy, emb,self.shared_dataLoader,global_shared_result)
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tmse:{:.6f}\tmae:{:.6f}'.format(
                            iter, batch_idx * len(x), len(self.dataLoader_train.dataset),
                                100. * batch_idx / len(self.dataLoader_train), metrics[0],metrics[1],metrics[2]))
                batch_loss.append(metrics[0])
                batch_mse.append(metrics[1])
                batch_mae.append(metrics[2])

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_mse.append(sum(batch_mse)/len(batch_mse))
            epoch_mae.append(sum(batch_mae)/len(batch_mae))
        gradients = {}
        for name, param in local_model.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()
        return local_model.model.state_dict(), sum(epoch_loss) / len(epoch_loss),sum(epoch_mse)/len(epoch_mse),sum(epoch_mae)/len(epoch_mae),gradients

