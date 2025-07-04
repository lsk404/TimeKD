#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w_locals_dict, data_radio):
    """
    联邦加权平均聚合
    Args:
        w_locals_dict: dict {client_name: state_dict} 
        data_radio: dict {client_name: ratio} 
    Returns:
        聚合后的 state_dict
    """
    # 筛选有效客户端
    valid_clients = set(w_locals_dict.keys()) & set(data_radio.keys())
    # 计算权重总和(用于归一化)
    sum_radio = sum(data_radio[client] for client in valid_clients)
    # 初始化聚合后的 state_dict(复制第一个客户端的结构)
    avg_state_dict = {k: torch.zeros_like(v) for k, v in next(iter(w_locals_dict.values())).items()}
    # 加权聚合
    for client in valid_clients:
        weight = data_radio[client] / sum_radio  
        # 遍历该客户端的每一层参数
        for param_name, param in w_locals_dict[client].items():
            avg_state_dict[param_name] += weight * param
    
    return avg_state_dict
