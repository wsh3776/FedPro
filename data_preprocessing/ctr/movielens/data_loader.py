# 这里要写完整的路径
from data_preprocessing.ctr.movielens.datasets import get_ctr_movielens_datasets, get_negative_samples_per_user
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
from sklearn.utils import shuffle
import argparse


def parse_args():
    """
    除了fedavg_main.py中需要读入参数外，这个模块也要读入参数，可以把参数都写在命令行参数里面，
    parser会自动从sys.argv中去解析匹配到的参数

    Returns: args

    """
    parser = argparse.ArgumentParser(description='*******data_loader*******')

    parser.add_argument('--client_num_in_total', type=int, default=200, metavar='NN',
                        help='number of clients in distributed cluster')

    parser.add_argument('--batch_size', type=int, default=32, metavar='BS',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--ratio_of_neg_to_pos', type=int, default=1, metavar='RPN',
                        help='the ratio of generating negative samples')

    parser.add_argument('--proportion_of_test_datasets', type=int, default=0.2, metavar='RPN',
                        help='the proportion of test datasets in total datasets')

    args = parser.parse_known_args()[0]
    return args


def get_train_test_dataset(args):
    users, movies, ratings, all_data = get_ctr_movielens_datasets()  # 导入的模块函数

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
    X = ratings[features][:100000]
    X = X.values  # pandas -> numpy
    Y = ratings[labels][:100000]
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


def partition_data(partition_method="homo", batch_size=32):
    # TODO: add parse_args
    args = parse_args()

    train_data, test_data, train_label, test_label = get_train_test_dataset(args)

    if partition_method == "homo":
        train_dataloader, test_dataloader = split_data_iid(train_data, test_data, train_label, test_label,
                                                           num_clients=args.client_num_in_total,
                                                           batch_size=args.batch_size)
    elif partition_method == "hetero":
        pass
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
    train_X, train_Y = shuffle(train_X, train_Y, random_state=42)

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


def split_data_non_iid(data):
    pass


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
