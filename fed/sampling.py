import numpy as np
from torchvision import datasets, transforms

def dataset_iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def dataset_contiguous(dataset, num_users):
    """
    Sample non-IID client data by splitting the dataset into contiguous chunks,
    and then randomly assign these chunks to each user.

    :param dataset: 数据集对象(需要支持 len())
    :param num_users: 用户数量
    :return: dict of image indices assigned to each user
    """
    num_items = len(dataset)
    chunk_size = num_items // num_users
    remainder = num_items % num_users
    # 创建所有样本索引
    all_idxs = np.arange(num_items)
    # 划分连续的 chunks
    chunks = []
    start = 0
    for i in range(num_users):
        if i < remainder:
            end = start + chunk_size + 1
        else:
            end = start + chunk_size
        chunks.append(set(all_idxs[start:end]))
        start = end
    # 随机打乱 chunks 的顺序
    np.random.shuffle(chunks)
    # 分配给每个用户
    dict_users = {i: chunks[i] for i in range(num_users)}

    return dict_users