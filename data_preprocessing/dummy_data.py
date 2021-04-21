import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def DummyData():
    # 这里的随机性已经被fedavg_main.py中的seed固定了
    X = np.random.randn(300, 1, 28, 28)  # ndarray
    Y = np.random.randint(10, size=300)  # label = [0,1,...,9]

    # as_tensor把numpy或者list转换为tensor对象
    # 14个客户端
    alist = [0, 8, 19, 40, 61, 85, 100, 121, 149, 168, 198, 231, 265, 280]
    blist = [8, 19, 40, 61, 85, 100, 121, 149, 168, 198, 231, 265, 280, 300]
    train_dataloader = []

    for a, b in zip(alist, blist):
        X_train = torch.as_tensor(X[a:b], dtype=torch.float32)
        Y_train = torch.as_tensor(Y[a:b], dtype=torch.long)
        train_ids = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(dataset=train_ids, batch_size=4, shuffle=True)  # shuffle打乱TensorDataset
        train_dataloader.append(train_loader)

    X = np.random.randn(280, 1, 28, 28)  # ndarray
    Y = np.random.randint(10, size=280)  # label = [0,1,...,9]
    alist = [0, 9, 18, 43, 61, 81, 120, 131, 159, 168, 188, 239, 255, 260]
    blist = [9, 18, 43, 61, 81, 120, 131, 159, 168, 188, 239, 255, 260, 280]
    test_dataloader = []

    for a, b in zip(alist, blist):
        X_test = torch.as_tensor(X[a:b], dtype=torch.float32)
        Y_test = torch.as_tensor(Y[a:b], dtype=torch.long)
        test_ids = TensorDataset(X_test, Y_test)
        test_loader = DataLoader(dataset=test_ids, batch_size=4, shuffle=True)  # shuffle打乱TensorDataset
        test_dataloader.append(test_loader)
    # 每个dataloader中的sampler.num_samples可以的到数量
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    DummyData()
