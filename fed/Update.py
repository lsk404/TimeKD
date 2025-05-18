#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import time


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset,idxs), batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers)

    def train(self, engine):
        # train and update
        epoch_loss = []
        epoch_mse = []
        epoch_mae = []
        device = torch.device(self.args.device)
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_mse = []
            batch_mae = []
            for batch_idx, (x, y, emb) in enumerate(self.ldr_train):
                trainx = torch.Tensor(x).to(device).float()
                trainy = torch.Tensor(y).to(device).float()
                emb = torch.Tensor(emb).to(device).float()
                metrics = engine.train(trainx, trainy, emb)
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tmse:{:.6f}\tmae:{:.6f}'.format(
                            iter, batch_idx * len(x), len(self.ldr_train.dataset),
                                100. * batch_idx / len(self.ldr_train), metrics[0],metrics[1],metrics[2]))
                batch_loss.append(metrics[0])
                batch_mse.append(metrics[1])
                batch_mae.append(metrics[2])
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_mse.append(sum(batch_mse)/len(batch_mse))
            epoch_mae.append(sum(batch_mae)/len(batch_mae))
        return engine.model.state_dict(), sum(epoch_loss) / len(epoch_loss),sum(epoch_mse)/len(epoch_mse),sum(epoch_mae)/len(epoch_mae)

