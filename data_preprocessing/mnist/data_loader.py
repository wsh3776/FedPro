import random
import numpy as np
import bisect
import torch
from data_preprocessing.mnist.datasets import get_datasets
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset
from itertools import accumulate
import argparse


def parse_args():
    """
    除了fedavg_main.py中需要读入参数外，这个模块也要读入参数，可以把参数都写在命令行参数里面，
    parser会自动从sys.argv中去解析匹配到的参数
    """
    parser = argparse.ArgumentParser(description='*******data_loader*******')

    parser.add_argument('--client_num_in_total', type=int, default=200, metavar='NN',
                        help='number of clients in distributed cluster')

    parser.add_argument('--batch_size', type=int, default=32, metavar='BS',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--partition_alpha', type=float, default=0.28, metavar='RPN',
                        help='partition_alpha')

    args = parser.parse_known_args()[0]
    return args


def partition_data(partition_method="hetero"):
    args = parse_args()

    train_data, test_data = get_datasets()

    if partition_method == "homo":
        train_dataloader, test_dataloader = split_data_iid(train_data, test_data, num_clients=args.client_num_in_total,
                                                           batch_size=args.batch_size)
    elif partition_method == "hetero":
        # TODO: num_clients和alpha最好作为args参数传入
        # alpha越小,异质程度越高
        train_dataloader, test_dataloader = split_data_non_iid(train_data, test_data,
                                                               num_clients=args.client_num_in_total,
                                                               alpha=args.partition_alpha,
                                                               batch_size=args.batch_size)
    elif partition_method == "centralized":
        train_dataloader, test_dataloader = centralized_data(train_data, test_data, batch_size=args.batch_size)
    return train_dataloader, test_dataloader


def centralized_data(train_data, test_data, batch_size=32):
    """

    Args:
        train_data:
        test_data:
        batch_size:

    Returns:

    """
    train_dataloader, test_dataloader = [], []

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_dataloader.append(train_loader)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    test_dataloader.append(test_loader)

    return train_dataloader, test_dataloader


def split_data_iid(train_data, test_data, num_clients, batch_size):
    train_dataloader, test_dataloader = [], []

    # =============== train_data =====================
    # 随机打乱数据集的seed，划分成iid
    train_X = train_data.data.reshape((len(train_data.data), 1, 28, 28))
    train_X = train_X.numpy()
    train_Y = train_data.targets.numpy()
    train_X, train_Y = shuffle(train_X, train_Y, random_state=42)

    # array_split 划分数据集，不均匀不会报错
    train_X_list = np.array_split(train_X, num_clients)
    train_Y_list = np.array_split(train_Y, num_clients)

    for X, Y in zip(train_X_list, train_Y_list):
        X_train = torch.as_tensor(X, dtype=torch.float32)
        Y_train = torch.as_tensor(Y, dtype=torch.long)
        train_ids = TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_ids, batch_size=batch_size, shuffle=True)
        train_dataloader.append(train_loader)

    # =============== test_data =====================
    # 随机打乱数据集的seed，划分成iid
    test_X = test_data.data.reshape((len(test_data.data), 1, 28, 28))
    test_X = test_X.numpy()
    test_Y = test_data.targets.numpy()
    test_X, test_Y = shuffle(test_X, test_Y, random_state=42)

    # array_split 划分数据集，不均匀不会报错
    test_X_list = np.array_split(test_X, num_clients)
    test_Y_list = np.array_split(test_Y, num_clients)

    for X, Y in zip(test_X_list, test_Y_list):
        X_test = torch.as_tensor(X, dtype=torch.float32)
        Y_test = torch.as_tensor(Y, dtype=torch.long)
        test_ids = TensorDataset(X_test, Y_test)
        test_loader = torch.utils.data.DataLoader(dataset=test_ids, batch_size=batch_size, shuffle=True)
        test_dataloader.append(test_loader)

    # b = np.array([[1,2],[3,4],[5,6],[7,8]])
    # c = np.array([0, 1, 2, 3])
    return train_dataloader, test_dataloader


def split_data_non_iid(train_data, test_data, num_clients, alpha, batch_size):
    """
    使用狄利克雷分布划分MNIST数据集为non-iid数据集
    """
    train_dataloader, test_dataloader = [], []

    clients_train_data = data_split(train_data, num_clients, alpha)
    for client_train_data in clients_train_data:
        train_dataloader.append(data_to_dataloader(client_train_data, batch_size))

    clients_test_data = data_split(test_data, num_clients, alpha)
    for client_test_data in clients_test_data:
        test_dataloader.append(data_to_dataloader(client_test_data, batch_size))

    return train_dataloader, test_dataloader


# *****************************************************************************************
"""狄利克雷分布产生non-iid数据集"""


def dirichlet_partition(samples, num_clients, alpha):
    """
    O(n)
    """
    ret = {i: [] for i in range(num_clients)}
    random.shuffle(samples)
    # TODO: what does it mean
    prop = np.random.dirichlet(np.repeat(alpha, num_clients))
    prop = list(accumulate(prop))
    i = 0
    for idx in range(0, len(prop)):
        pre = i
        while i / (len(samples)) < prop[idx]:
            i += 1
        ret[idx] += samples[pre:i]
    # for i, sample in enumerate(samples):
    #     idx = bisect.bisect_left(prop, i/len(samples), 0, len(prop))
    #     ret[idx].append(sample)
    return ret


def dirichlet_partition_2(samples, num_clients, alpha):
    """
    O(nlogn)
    """
    ret = {i: [] for i in range(num_clients)}
    random.shuffle(samples)
    prop = np.random.dirichlet(np.repeat(alpha, num_clients))
    for i in range(1, len(prop)):
        prop[i] = prop[i - 1] + prop[i]
    for i, sample in enumerate(samples):
        idx = bisect.bisect_left(prop, i / len(samples), 0, len(prop))
        ret[idx].append(sample)
    return ret


def split_by_label(data):
    ret = {}
    for sample, label in data:  # sample shape: torch.Size([1, 28, 28])
        if label not in ret.keys():
            ret[label] = []
        ret[label].append((sample, label))
    # ret: {0: [(data_i, 0), (data_j, 0)], 1: [..., ..., ...], ..., 9: [..., ...]}
    return ret


def data_split(data, num_clients, alpha):
    # return a dict user2data
    label2data = split_by_label(data)
    user2data = {i: [] for i in range(num_clients)}

    while True:
        # 每个客户端至少分到10个数据
        flag = 0
        for i, data in user2data.items():
            if len(data) < 10:
                flag = 1
                break

        if flag == 0:
            break
        for label, samples in label2data.items():
            ret = dirichlet_partition(samples, num_clients, alpha)
            for user, sample in ret.items():
                user2data[user] += sample  # [(data_i, 5), (data_j, 5)] += [(data_k, 3), (data_t, 3), ...]

    return list(user2data.values())  # 得到每个客户端划分好后的数据集[client_1_data, ..., client_n_data]


def data_to_dataloader(data, batch_size):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)


# *****************************************************************************************

if __name__ == "__main__":
    partition_data(partition_method="hetero")
