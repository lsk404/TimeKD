import numpy as np
from torchvision import datasets, transforms
import torch


def Distribute_data(dataset, client_nums,method="contiguous"):
    """
    Distribute data to each client
    :param dataset: dataset
    :param client_nums: number of clients
    :param method: method of distribution(contiguous or random)
    :return: client_data_dict
    """
    item_nums = len(dataset)
    data_per_client = item_nums // client_nums
    remainder = item_nums % client_nums
    
    client_data_dict = {}
    if(method == "random"):
        indices = np.random.permutation(len(dataset))
        dataset = torch.utils.data.Subset(dataset, indices)
    start_idx = 0
    for i in range(client_nums):
        if(i < remainder):
            end_idx = (i + 1) * data_per_client + 1
        else:
            end_idx = (i + 1) * data_per_client
        sub_dataset = []
        for j in range(start_idx,end_idx):
            sub_dataset.append(dataset[j])
        client_data_dict[f"client_{i+1}"] = sub_dataset
        start_idx = end_idx
    return client_data_dict


# def dataset_iid(dataset, num_client):
#     """
#     Sample I.I.D. client data from dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index, radio of each client
#     """
#     num_items = len(dataset)
#     num_per_client = num_items//num_client
#     remainder = num_items % num_client
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_client):
#         if(i < remainder):
#             num_item = num_per_client + 1
#         else :
#             num_item = num_per_client
#         dict_users[i] = set(np.random.choice(all_idxs, num_item, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     # 分配给每个用户
#     data_radio = [len(dict_users[i])/num_items for i in range(num_client)]
#     return dict_users,data_radio

# def dataset_contiguous(dataset, num_client):
#     """
#     Sample non-IID client data by splitting the dataset into contiguous chunks,
#     and then randomly assign these chunkchunks to each user.

#     :param dataset: 数据集对象(需要支持 len())
#     :param num_users: 用户数量
#     :return: dict of image indices assigned to each user
#     """
#     num_items = len(dataset)
#     chunk_size = num_items // num_client
#     remainder = num_items % num_client
#     # 创建所有样本索引
#     all_idxs = np.arange(num_items)
#     chunks = []
#     start = 0
#     for i in range(num_client):
#         if i < remainder:
#             end = start + chunk_size + 1
#         else:
#             end = start + chunk_size
#         chunks.append(set(all_idxs[start:end]))
#         start = end

#     # 分配给每个用户
#     dict_users = {i: chunks[i] for i in range(num_client)}
#     data_radio = [len(dict_users[i])/num_items for i in range(num_client)]
#     return dict_users,data_radio