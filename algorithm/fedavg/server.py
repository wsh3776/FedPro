import numpy as np
from algorithm.fedavg.client import Client
from tqdm import tqdm
import wandb

from data_preprocessing.dummy_data import DummyData  # 这个其实应该放到
from data_preprocessing.mnist.data_loader import partition_data as partition_data_mnist
from data_preprocessing.ctr.movielens.data_loader import partition_data as partition_data_ctr_movielens
from models.fedavg.mnist import MNIST
from models.fedavg.lr import LogisticRegression


class Server():
    def __init__(self, args):
        self.model_name = args.model
        self.dataset = args.dataset
        self.client_num_in_total = args.client_num_in_total
        self.client_num_per_round = args.client_num_per_round
        self.num_rounds = args.num_rounds
        self.partition_method = args.partition_method
        self.lr = args.lr
        self.optimizer = args.client_optimizer
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.eval_interval = args.eval_interval
        self.seed = args.seed
        self.device = args.device
        self.selected_clients_idx = []
        self.clients = []
        self.model = self._select_model(self.model_name)  # 创建对应的模型
        # use client_num_per_round models to train all data
        # 设置k个模型来训练每轮被选择的K个客户端
        self.surrogate = self._setup_surrogate()  # 会用到self.model
        # Add datasets into each client: self.clients
        self.clients = self._setup_clients()
        self.global_params = self.model.state_dict()  # 设置为全局模型的参数
        self.updates = []

    def _select_model(self, model_name):
        model = None
        if model_name == 'cnn_mnist':
            model = MNIST()
        elif model_name == 'lr_ctr':
            model = LogisticRegression(input_dim=129, output_dim=5)
        return model

    # TODO: load data
    def get_dataloader(self):
        if self.dataset == "dummy":
            train_dataloader, test_dataloader = DummyData()
        elif self.dataset == "mnist":
            train_dataloader, test_dataloader = partition_data_mnist(self.partition_method, self.batch_size)
        elif self.dataset == "ctr_movielens":
            train_dataloader, test_dataloader = partition_data_ctr_movielens(self.partition_method, self.batch_size)
        # 如果想跑集中式，我只要把参数并在一起就行了
        # if self.centralized==True:
        return train_dataloader, test_dataloader

    def _select_clients(self, round_th):
        np.random.seed(seed=self.seed + round_th)
        selected_clients_index = np.random.choice(self.client_num_in_total,
                                                  size=self.client_num_per_round,
                                                  replace=False)
        return selected_clients_index

    # TODO: setup all clients, user_id or just id? add DATALOADER
    def _setup_clients(self):
        # setup all clients (actually we just want to store the data into client)
        train_dataloader, test_dataloader = self.get_dataloader()
        # 这里不需要传递epoch,lr,model_name，因为这些每个客户端是一样的，我只要在surrogate里设置就行了
        clients = [
            Client(user_id=i,
                   train_dataloader=train_dataloader[i],
                   test_dataloader=test_dataloader[i])
            for i in np.arange(self.client_num_in_total)]
        return clients

    def _setup_surrogate(self):
        """
        Your can use only one model to train. The only thing you need to is update the datasets and parameters for this each client
        Also you can define num_per_round models and use multiprocessing to speed up training if you want.
        """
        # Client need to update the dataset and params
        surrogate = [Client(user_id=i, train_dataloader=None, test_dataloader=None,
                            model=self.model, epoch=self.epoch, lr=self.lr, optimizer=self.optimizer,
                            device=self.device)
                     for i in range(self.client_num_per_round)]
        # print(surrogate[0].model)
        return surrogate

    def _aggregate(self):
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
        return self

    def _train_on_clients(self, round):
        selected_clients_index = self._select_clients(round_th=round)
        print("-" * 50)
        print(f"Round {round}")
        print("train local models:")
        # print(selected_clients_index)
        self.updates = []  # 每轮通信清空这个updates
        for k in tqdm(range(self.client_num_per_round)):
            # 训练时只把参数发给被选中的客户端
            surrogate = self.surrogate[k]  # 放到第k个槽位上
            surrogate.update_local_dataset(self.clients[selected_clients_index[k]])  # update datasets and params
            surrogate.set_params(self.global_params)
            # 得到参数，这里的sample_loss其实没什么用，因为我们还没有更新全局模型
            # 本地训练 local client train
            local_params, train_data_num, sample_loss \
                = surrogate.train()
            self.updates.append((local_params, train_data_num))
        return self  # 返回这个类

    def _test_global_model(self, round):
        print("evaluate global model:")

        # eval on train data
        acc_list, loss_list = self._eval_model(dataset='train')
        avg_acc_all = self.avg_metric(acc_list)
        avg_loss_all = self.avg_metric(loss_list)
        wandb.log({"Train/acc": avg_acc_all * 100, "round": round})
        wandb.log({"Train/loss": avg_loss_all * 100, "round": round})

        # eval on test data
        acc_list, loss_list = self._eval_model(dataset='test')
        avg_acc_all = self.avg_metric(acc_list)
        avg_loss_all = self.avg_metric(loss_list)
        wandb.log({"Test/acc": avg_acc_all * 100, "round": round})
        wandb.log({"Test/loss": avg_loss_all * 100, "round": round})

        print()
        print(f"[TRAIN] Avg acc: {avg_acc_all * 100:.3f}%, Avg loss: {avg_loss_all:.5f}")
        print(f"[TEST]  Avg acc: {avg_acc_all * 100:.3f}%, Avg loss: {avg_loss_all:.5f}")
        # print("-" * 50)
        print()

    def federate(self):
        print("Begin Federating!")
        print(f"Training among {self.client_num_in_total} clients! \n")

        # TODO: (代码整洁之道）这里train和test两个功能应该包裹成两个函数，不要写在一起
        for t in range(self.num_rounds):
            """
            server-client communication round t
            """
            self._train_on_clients(t)._aggregate()  # 训练 + 更新self.global_params

            # 间隔多久用当前的全局模型参数做一次所有训练集和测试集的测试
            # 测试时要把参数发给所有的客户端
            if t % self.eval_interval == 0:
                self._test_global_model(t)

        # update global model params

    def _eval_model(self, dataset: str = 'test'):
        """
        用当前的全局模型评估所有客户端训练集or测试集上的准确率和损失值

        Args:
            dataset:  'train' or 'test'

        Returns:

        """
        # print(f"\n====Eval on all {dataset} dataset====")
        acc_list = []
        loss_list = []
        for k in tqdm(range(self.client_num_in_total)):
            surrogate = self.surrogate[k % self.client_num_per_round]  # 放到槽位上去算
            surrogate.update_local_dataset(self.clients[k])  # update dataset and params
            surrogate.set_params(self.global_params)

            client_num, accuracy, avg_loss = surrogate.test(dataset=dataset)  # 在本地模型上进行测试
            acc_list.append((client_num, accuracy))
            loss_list.append((client_num, avg_loss))

        return acc_list, loss_list

    @staticmethod
    def avg_metric(metric_list):
        total_num = 0
        total_metric = 0
        for (client_num, metric) in metric_list:
            total_num += client_num
            total_metric += metric * client_num
        avg_metric = total_metric / total_num
        return avg_metric
