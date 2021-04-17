# 这里要写完整的路径
from data_preprocessing.ctr.movielens.datasets_onehot import users, movies, ratings, all_data
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# print(users)
# print(movies)
# print(ratings)

data = all_data
# 对year进行归一化
# data['year'] = data['year'].astype(int)
# data['year'] = (data['year'] - data['year'].min()) / (data['year'].max() - data['year'].min())
# data['age'] = (data['age'] - data['age'].min()) / (data['age'].max() - data['age'].min())

# TODO：选择特征有待考虑，另外year，以及age需要归一化吗，需要归一化吗？ zip，timestamp需要加入吗
# 这里应该把user_id, movie_id等信息考虑进来
features = list(data.columns.drop(labels=['rating', 'user_id', 'movie_id']))
# len(features)
labels = ['rating']

# 100万条数据我先取前面10000条
X = data[features][:10000]
X = X.values  # pandas -> numpy
Y = data[labels][:10000]
Y = Y.values.reshape(len(Y))  # pandas -> numpy
# array([1, 1, 1, ..., 0, 1, 1])

# 利用train_test_split将数据集随机划分为训练集和测试集 4:1 (这里有个随机种子seed)
train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=0.20, random_state=32)


def partition_data(partition_method="homo", batch_size=32):
    if partition_method == "homo":
        # 随机打乱pandas数据集
        data = data.sample(frac=1).reset_index(drop=True)
        print(users.head())
    elif partition_method == "hetero":
        print(movies.head())
    elif partition_method == "centralized":
        train_dataloader, test_dataloader = centralized_data(batch_size=batch_size)
    return train_dataloader, test_dataloader


# 统一划分数据集吧
def split_data_iid(data, N):
    """
    :param data: 需要划分的数据集
    :param N: 每个客户端数量都为N
    :return:
    """
    pass


def split_data_noniid(data):
    pass


def centralized_data(batch_size):
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
