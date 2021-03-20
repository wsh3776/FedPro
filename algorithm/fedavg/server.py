import torch
import numpy as np
from algorithm.fedavg.client import Client
import copy

from data.demo_data import demoData
from models.fedavg.MNIST import MNIST


class Server():
    def __init__(self, args):
        self.model_name = args.model
        self.dataset = args.dataset
        self.client_num_in_total = args.client_num_in_total
        self.client_num_per_round = args.client_num_per_round
        self.num_rounds = args.num_rounds
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.seed = args.seed
        self.note = args.note
        self.eval_interval = args.eval_interval
        self.selected_clients_idx = []
        self.clients = []
        # use client_num_per_round models to train all data
        # 设置k个模型来训练每轮被选择的K个客户端
        self.surrogate = self.setup_surrogate()
        # Add datasets into each client: self.clients
        self.clients = self.setup_clients()
        self.model = self.select_model(self.model_name)  # 创建对应的模型
        self.global_params = self.model.state_dict()  # 设置为全局模型的参数
        self.updates = []

    def select_model(self, model_name):
        model = None
        if model_name == 'mnist':
            model = MNIST()
        return model

    # TODO: load data
    def get_dataloader(self):
        if self.dataset == "demo":
            train_dataloader, test_dataloader = demoData()
        return train_dataloader, test_dataloader

    def select_clients(self, round_th):
        np.random.seed(seed=self.seed + round_th)
        selected_clients_index = np.random.choice(self.client_num_in_total,
                                                  size=self.client_num_per_round,
                                                  replace=False)
        return selected_clients_index

    # TODO: setup all clients, user_id or just id? add DATALOADER
    def setup_clients(self):
        # setup all clients (actually we just want to store the data into client)
        train_dataloader, test_dataloader = self.get_dataloader()
        # 这里不需要传递epoch,lr,model_name，因为这些每个客户端是一样的，我只要在surrogate里设置就行了
        clients = [
            Client(user_id=i,
                   train_dataloader=train_dataloader[i],
                   test_dataloader=test_dataloader[i])
            for i in np.arange(self.client_num_in_total)]
        return clients

    def setup_surrogate(self):
        """
        Your can use only one model to train. The only thing you need to is update the datasets and parameters for this each client
        Also you can define num_per_round models and use multiprocessing to speed up training if you want.
        """
        # Client need to update the dataset and params
        surrogate = [Client(user_id=i, train_dataloader=None, test_dataloader=None,
                            model_name=self.model_name, epoch=self.epoch, lr=self.lr)
                     for i in range(self.client_num_per_round)]
        return surrogate

    def aggrerate(self):
        n = sum([n_k for (params, n_k) in self.updates])

        new_params = self.updates[0][0]
        for key in self.updates[0][0].keys():  # key: cov1.weight, cov1.bias...
            for i in range(len(self.updates)):
                client_params, n_k = self.updates[i]
                if i == 0:
                    new_params[key] = client_params[key] * n_k / n
                else:
                    new_params[key] += client_params[key] * n_k / n

        self.global_params = new_params

    def federate(self):
        print("Begin Federating!")
        print(f"Training among {self.client_num_in_total} clients!")

        for t in range(self.num_rounds):
            """
            server-client communication round t
            """
            selected_clients_index = self.select_clients(round_th=t)
            # print(selected_clients_index)
            self.updates = []  # 每轮通信清空这个updates
            for k in range(self.client_num_per_round):
                # 训练时只把参数发给被选中的客户端
                surrogate = self.surrogate[k]  # 放到第k个槽位上
                surrogate.update_local_dataset(self.clients[selected_clients_index[k]])  # update datasets and params
                surrogate.set_params(self.global_params)
                # 得到参数，这里的sample_loss其实没什么用，因为我们还没有更新全局模型
                local_params, train_data_num, sample_loss = surrogate.train()
                self.updates.append((local_params, train_data_num))

            # average params
            self.aggrerate()  # 更新self.global_params

            # 间隔多久用当前的全局模型参数做一次所有训练集和测试集的测试
            # 测试时要把参数发给所有的客户端
            if t % self.eval_interval == 0:
                print("-"*40, "\n")
                print(f"Round {t}")

                # eval on train data
                acc_list, loss_list = self.eval_model(dataset='train')
                avg_acc_all = self.avg_metric(acc_list)
                avg_loss_all = self.avg_metric(loss_list)
                print(f"[TRAIN] Avg acc: {avg_acc_all * 100:.3f}%, Avg loss: {avg_loss_all:.5f}")

                # eval on test data
                acc_list, loss_list = self.eval_model(dataset='test')
                avg_acc_all = self.avg_metric(acc_list)
                avg_loss_all = self.avg_metric(loss_list)
                print(f"[TEST] Avg acc: {avg_acc_all * 100:.3f}%, Avg loss: {avg_loss_all:.5f}")


        # update global model params

    def eval_model(self, dataset='test'):
        """
        用当前的全局模型评估所有客户端训练集or测试集上的准确率和损失值

        Args:
            dataset:  'train' or 'test'

        Returns:

        """
        # print(f"\n====Eval on all {dataset} dataset====")
        acc_list = []
        loss_list = []
        for k in range(self.client_num_in_total):
            surrogate = self.surrogate[k % self.client_num_per_round] # 放到槽位上去算
            surrogate.update_local_dataset(self.clients[k])  # update dataset and params
            surrogate.set_params(self.global_params)
            client_num, accuracy, avg_loss = surrogate.test(dataset=dataset)  # 在本地模型上进行测试
            acc_list.append((client_num, accuracy))
            loss_list.append((client_num, avg_loss))

        return acc_list, loss_list

    @staticmethod # https://blog.csdn.net/qq_28805371/article/details/103248194
    def avg_metric(metric_list):
        total_num = 0
        total_metric = 0
        for (client_num, metric) in metric_list:
            total_num += client_num
            total_metric += metric * client_num
        avg_metric = total_metric / total_num
        return avg_metric

