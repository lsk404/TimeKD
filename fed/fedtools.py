import torch
import numpy as np
def compute_l2_norm_diff(w_global, w_local):
    total_norm = 0.0
    for key in w_global.keys():
        delta = w_global[key] - w_local[key]
        total_norm += torch.sum(delta ** 2)  # 计算平方和
    return torch.sqrt(total_norm)  # 开平方得到 L2 范数

def compute_gradient_similarity(grad1, grad2):
    """计算两个梯度字典之间的相似度"""
    similarity = 0
    dot_product = 0.0
    norm1 = 0
    norm2 = 0
    for key in grad1.keys():
        g1 = grad1[key].flatten().cpu().numpy() if hasattr(grad1[key], 'cpu') else grad1[key].flatten()
        g2 = grad2[key].flatten().cpu().numpy() if hasattr(grad2[key], 'cpu') else grad2[key].flatten()

        dot_product += np.dot(g1, g2)
        norm1 += np.sum(g1 ** 2)
        norm2 += np.sum(g2 ** 2)

    norm1 = np.sqrt(norm1)
    norm2 = np.sqrt(norm2)

    if norm1 == 0 or norm2 == 0:
        return 0.0  # 避免除以零

    cosine_similarity = dot_product / (norm1 * norm2)
    return float(cosine_similarity)

def compute_aggregation_weights(local_grad_dict,temperature = 1):
    aggregation_weights = {client_name:1 for client_name in local_grad_dict.keys()}
    avg_grad = {}
    # 初始化平均梯度
    for client_name, gradients in local_grad_dict.items():
        for name, param in gradients.items():
            if name not in avg_grad:
                avg_grad[name] = torch.zeros_like(param)
            avg_grad[name] += param
    
    # 计算平均梯度
    for name in avg_grad.keys():
        avg_grad[name] /= len(local_grad_dict)

    # 调整权重
    for client_name, gradients in local_grad_dict.items():
        sim = compute_gradient_similarity(gradients, avg_grad)
        aggregation_weights[client_name] = np.exp(sim / temperature)
    # 归一化权重
    sum_weights = sum(aggregation_weights.values())
    aggregation_weights = {k: v / sum_weights for k, v in aggregation_weights.items()}
    return aggregation_weights