import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))

def MSE(pred, true):
    return torch.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mse = MSE(pred, true).item() # loss function
    mae = MAE(pred, true).item()
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    
    # return mae,mse,rmse,mape,mspe
    return mse, mae

def cls_metric(pred, true):
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    acc = accuracy_score(pred, true).item()
    mf1 = f1_score(pred, true, average='macro').item() 
    kappa = cohen_kappa_score(pred, true).item()
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    
    # return mae,mse,rmse,mape,mspe
    return acc, mf1, kappa