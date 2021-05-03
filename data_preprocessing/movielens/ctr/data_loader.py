# 这里要写完整的路径
from data_preprocessing.movielens.ctr.datasets import get_ctr_movielens_datasets, get_negative_samples_per_user
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
from sklearn.utils import shuffle
import argparse
import random
from itertools import accumulate


def parse_args():
    """
    除了fedavg_main.py中需要读入参数外，这个模块也要读入参数，可以把参数都写在命令行参数里面，
    parser会自动从sys.argv中去解析匹配到的参数

    Returns: args

    """
    config = {
        "ratio_of_neg_to_pos": 2,
        "partition_alpha": 0.8,
        "proportion_of_test_datasets": 0.2,
    }

    parser = argparse.ArgumentParser(description='*******data_loader*******')

    parser.add_argument('--ratio_of_neg_to_pos', type=int, default=1, metavar='RPN',
                        help='the ratio of generating negative samples')

    parser.add_argument('--proportion_of_test_datasets', type=int, default=0.2, metavar='RPN',
                        help='the proportion of test datasets in total datasets')

    parser.add_argument('--partition_alpha', type=float, default=0.9, metavar='RPN',
                        help='partition_alpha')

    parser.set_defaults(**config)

    args = parser.parse_known_args()[0]
    return args


def get_train_test_dataset(args):
    users, movies, ratings, all_data = get_ctr_movielens_datasets()  # 导入的模块函数

    # 生成负样本
    df_negative_items = get_negative_samples_per_user(users, movies, ratings,
                                                      ratio_of_neg_to_pos=args.ratio_of_neg_to_pos)

    ratings = ratings.append(df_negative_items).reset_index(drop=True)
    ratings.drop(columns=['timestamp'], inplace=True)

    # 生成负样本后再进行下面操作
    # ----------- embedding 准备工作 ----------------
    # 我们需要对user_id, movie_id特征创建字典（方便后续embedding）
    user_id_origin = list(users['user_id'])
    user_id_dic = {}
    for user_id in user_id_origin:
        if user_id not in user_id_dic:
            user_id_dic[user_id] = len(user_id_dic)

    # 在movies表中建立dict比较快
    movie_id_origin = list(movies['movie_id'])
    movie_id_dic = {}
    for movie_id in movie_id_origin:
        if movie_id not in movie_id_dic:
            movie_id_dic[movie_id] = len(movie_id_dic)
    # movie_id_dic: {1:0, ..., 3952: 3882}

    # 把ratings中的特征转换为dict映射后的值
    # apply is very fast
    ratings['user_id'] = ratings['user_id'].apply(lambda x: user_id_dic[x])
    ratings['movie_id'] = ratings['movie_id'].apply(lambda x: movie_id_dic[x])

    # movielens点击率预测label设置
    # ratings.loc[ratings['rating'] == 0, 'rating'] = 0 # 否则设为0
    ratings.loc[ratings['rating'] >= 1, 'rating'] = 1  # 评过分的record设为1

    # 对year进行归一化（后来我改成了onehot）
    # data['year'] = data['year'].astype(int)
    # data['year'] = (data['year'] - data['year'].min()) / (data['year'].max() - data['year'].min())

    # 这里应该把user_id, movie_id等信息考虑进来
    # features = list(data.columns.drop(labels=['rating', 'user_id', 'movie_id']))
    # len(features)

    features = ['user_id', 'movie_id']
    labels = ['rating']

    # 打乱数据集
    ratings = shuffle(ratings, random_state=42)

    # TODO: 100万条数据我先取前面10000条，这样速度快一点
    X = ratings[features][:200000]
    X = X.values  # pandas -> numpy
    Y = ratings[labels][:200000]
    Y = Y.values.reshape(len(Y))  # pandas -> numpy
    # array([1, 1, 1, ..., 0, 1, 1])

    # 利用train_test_split将数据集随机划分为训练集和测试集 4:1 (这里有个随机种子seed)
    train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=args.proportion_of_test_datasets,
                                                                      random_state=42)
    return train_data, test_data, train_label, test_label


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, i):
        data = self.data[i]
        label = self.label[i]
        return data, label

    def __len__(self):
        return len(self.label)


def partition_data(partition_method="homo", client_num_in_total=None, batch_size=None):
    # TODO: add parse_args
    args = parse_args()

    train_data, test_data, train_label, test_label = get_train_test_dataset(args)

    if partition_method == "homo":
        train_dataloader, test_dataloader = split_data_iid(train_data, test_data, train_label, test_label,
                                                           num_clients=client_num_in_total,
                                                           batch_size=batch_size)
    elif partition_method == "hetero":
        # alpha越小,异质程度越高
        train_dataloader, test_dataloader = split_data_non_iid(train_data, test_data, train_label, test_label,
                                                               num_clients=client_num_in_total,
                                                               alpha=args.partition_alpha,
                                                               batch_size=batch_size)
    elif partition_method == "centralized":
        train_dataloader, test_dataloader = centralized_data(train_data, test_data, train_label, test_label,
                                                             batch_size=batch_size)
    return train_dataloader, test_dataloader


def split_data_iid(train_data, test_data, train_label, test_label, num_clients, batch_size):
    train_dataloader, test_dataloader = [], []

    # =============== train_data =====================
    # 随机打乱数据集的seed，划分成iid
    train_X = train_data
    train_Y = train_label
    train_X, train_Y = shuffle(train_X, train_Y, random_state=12)

    # array_split 划分数据集，不均匀不会报错
    train_X_list = np.array_split(train_X, num_clients)
    train_Y_list = np.array_split(train_Y, num_clients)

    for X, Y in zip(train_X_list, train_Y_list):
        X_train = torch.as_tensor(X, dtype=torch.float32)
        Y_train = torch.as_tensor(Y, dtype=torch.long)
        train_ids = MyDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_ids, batch_size=batch_size, shuffle=True)
        train_dataloader.append(train_loader)

    # =============== test_data =====================
    # 随机打乱数据集的seed，划分成iid
    test_X = test_data
    test_Y = test_label
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


def split_data_non_iid(train_data, test_data, train_label, test_label, num_clients, alpha, batch_size):
    """
    使用狄利克雷分布划分MNIST数据集为non-iid数据集
    """
    train_dataloader, test_dataloader = [], []

    # 把train_data, test_data, train_label, test_label合并下
    # X_train = torch.as_tensor(train_data, dtype=torch.float32)
    # Y_train = torch.as_tensor(train_label, dtype=torch.long)
    # train_ids = MyDataset(X_train, Y_train)

    train_ids = []
    for data, label in zip(train_data, train_label):
        train_ids.append((data, int(label)))

    clients_train_data = data_split(train_ids, num_clients, alpha)
    for client_train_data in clients_train_data:
        train_dataloader.append(data_to_dataloader(client_train_data, batch_size))

    test_ids = []
    for data, label in zip(test_data, test_label):
        test_ids.append((data, int(label)))

    clients_test_data = data_split(test_ids, num_clients, alpha)
    for client_test_data in clients_test_data:
        test_dataloader.append(data_to_dataloader(client_test_data, batch_size))

    return train_dataloader, test_dataloader


def centralized_data(train_data, test_data, train_label, test_label, batch_size):
    train_dataloader, test_dataloader = [], []

    # 处理训练集
    X_train = torch.as_tensor(train_data.astype(float), dtype=torch.float32)
    Y_train = torch.as_tensor(train_label, dtype=torch.long)
    train_ids = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(dataset=train_ids, batch_size=batch_size, shuffle=True)  # shuffle打乱TensorDataset
    train_dataloader.append(train_loader)

    # 处理测试集
    X_test = torch.as_tensor(test_data.astype(float), dtype=torch.float32)
    Y_test = torch.as_tensor(test_label, dtype=torch.long)
    test_ids = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(dataset=test_ids, batch_size=batch_size, shuffle=True)  # shuffle打乱TensorDataset
    test_dataloader.append(test_loader)

    return train_dataloader, test_dataloader


# *****************************************************************************************
"""狄利克雷分布产生non-iid数据集"""


def data_split(data, num_clients, alpha):
    # TODO: alpha不能过小，不然有些客户端会只有0个样本
    # 解决方法：每个客户端都事先加点样本，或者概率加上一个0.1后作归一化(zs)
    # return a dict user2data
    label2data = split_by_label(data)
    user2data = {i: [] for i in range(num_clients)}

    for label, samples in label2data.items():
        ret = dirichlet_partition(samples, num_clients, alpha)
        for user, sample in ret.items():
            user2data[user] += sample  # [(data_i, 5), (data_j, 5)] += [(data_k, 3), (data_t, 3), ...]

    return list(user2data.values())  # 得到每个客户端划分好后的数据集[client_1_data, ..., client_n_data]


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


def split_by_label(data):
    ret = {}
    for sample, label in data:  # sample shape: torch.Size([1, 28, 28])
        if label not in ret.keys():
            ret[label] = []
        ret[label].append((sample, label))
    # ret: {0: [(data_i, 0), (data_j, 0)], 1: [..., ..., ...], ..., 9: [..., ...]}
    return ret


def data_to_dataloader(data, batch_size):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    partition_data(partition_method="hetero")
