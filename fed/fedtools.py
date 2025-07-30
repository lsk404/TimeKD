import torch

def compute_l2_norm_diff(w_global, w_local):
    total_norm = 0.0
    for key in w_global.keys():
        delta = w_global[key] - w_local[key]
        total_norm += torch.sum(delta ** 2)  # 计算平方和
    return torch.sqrt(total_norm)  # 开平方得到 L2 范数