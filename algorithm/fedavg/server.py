import wandb
import copy
import numpy as np
from algorithm.fedavg.client import Client
from tqdm import tqdm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from data_preprocessing.dummy_data import DummyData
from data_preprocessing.mnist.data_loader import partition_data as partition_data_mnist
from data_preprocessing.movielens.ctr.data_loader import partition_data as partition_data_movielens
from models.fedavg.mnist.cnn import CNN
from models.fedavg.movielens.mlp import MLP


class Server:
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
        self.lr_decay = args.lr_decay
        self.decay_step = args.decay_step

        self.clients: list = None
        self.surrogates: list = None
        self.model = None
        self.global_params = None

    def federate(self):
        print("Begin Federating!")
        print(f"Training among {self.client_num_in_total} clients! \n")

        self.model = self._select_model(self.model_name)
        # get the initialized global model params
        self.global_params = copy.deepcopy(self.model.state_dict())
        self.clients = self._setup_clients()
        self.surrogates = self._setup_surrogate()

        # server-client communication
        for round_th in range(self.num_rounds):
            updates = self._train_on_clients(round_th)

            self._aggregate_and_update_global_params(updates)

            if round_th % self.eval_interval == 0:
                self._eval_global_model(round_th)

    @staticmethod
    def _select_model(model_name):
        model = None
        if model_name == 'cnn':
            model = CNN()
        elif model_name == 'mlp':
            # user_id, movie_id进行embedding的网络模型
            model = MLP(output_dim=2)
        return model

    # TODO: load data
    def get_dataloader(self):
        # 多个if-else最好加个注释，可以接收哪几个参数
        if self.dataset == "dummy":
            train_dataloader, test_dataloader = DummyData()
        elif self.dataset == "mnist":
            train_dataloader, test_dataloader = partition_data_mnist(self.partition_method)
        elif self.dataset == "movielens":
            train_dataloader, test_dataloader = partition_data_movielens(self.partition_method)
        # 如果想跑集中式，我只要把参数并在一起就行了
        # if self.centralized==True:
        return train_dataloader, test_dataloader

    def _select_clients(self, round_th):
        np.random.seed(seed=self.seed + round_th)
        selected_clients_index = np.random.choice(self.client_num_in_total,
                                                  size=self.client_num_per_round,
                                                  replace=False)
        return selected_clients_index

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
                            model=self.model, epoch=self.epoch, lr=self.lr, lr_decay=self.lr_decay,
                            decay_step=self.decay_step, optimizer=self.optimizer,
                            device=self.device)
                     for i in range(self.client_num_per_round)]
        # print(surrogate[0].model)
        return surrogate

    def _aggregate_and_update_global_params(self, updates):
        n = sum([n_k for (params, n_k) in updates])

        new_params = updates[0][0]
        for key in updates[0][0].keys():  # key: cov1.weight, cov1.bias...
            for i in range(len(updates)):
                client_params, n_k = updates[i]
                if i == 0:
                    new_params[key] = client_params[key] * n_k / n
                else:
                    new_params[key] += client_params[key] * n_k / n

        self.global_params = new_params
        return self

    def _train_on_clients(self, round_th, updates=[]):
        selected_clients_index = self._select_clients(round_th=round_th)
        print(selected_clients_index)
        print("-" * 50)
        print(f"Round {round_th}")
        print("train local models:")
        # print(selected_clients_index)
        for k in tqdm(range(self.client_num_per_round)):
            # 训练时只把参数发给被选中的客户端
            surrogate = self.surrogates[k]  # 放到第k个槽位上
            surrogate.update_local_dataset(self.clients[selected_clients_index[k]])  # update datasets and params
            surrogate.set_params(self.global_params)
            # 得到参数，这里的sample_loss其实没什么用，因为我们还没有更新全局模型
            # 本地训练 local client train
            local_params, train_data_num, sample_loss \
                = surrogate.train(round_th)
            # print(local_params['fc2.weight'].sum().item())
            updates.append((local_params, train_data_num))
        return updates

    def _eval_global_model(self, round):
        print("evaluate global model:")

        # eval on train data
        all_train_labels, all_train_predicted, acc_train_list, loss_train_list = self._eval_model(dataset='train')
        avg_acc_train_all = self.avg_metric(acc_train_list)
        avg_loss_train_all = self.avg_metric(loss_train_list)

        # TODO: 这部分算precision二分类和多分类的情况？取平均？
        if self.dataset == "ctr_movielens":
            train_precision = precision_score(all_train_labels, all_train_predicted)
            train_recall = recall_score(all_train_labels, all_train_predicted)
            train_f1 = f1_score(all_train_labels, all_train_predicted)
            train_auc = roc_auc_score(all_train_labels, all_train_predicted)

            wandb.log({"Train/precision": train_precision, "round": round})
            wandb.log({"Train/recall": train_recall, "round": round})
            wandb.log({"Train/f1": train_f1, "round": round})
            wandb.log({"Train/auc": train_auc, "round": round})

        wandb.log({"Train/acc": avg_acc_train_all, "round": round})
        wandb.log({"Train/loss": avg_loss_train_all, "round": round})

        # eval on test data
        all_test_labels, all_test_predicted, acc_test_list, loss_test_list = self._eval_model(dataset='test')
        avg_acc_test_all = self.avg_metric(acc_test_list)
        avg_loss_test_all = self.avg_metric(loss_test_list)

        if self.dataset == "ctr_movielens":
            test_precision = precision_score(all_test_labels, all_test_predicted)
            test_recall = recall_score(all_test_labels, all_test_predicted)
            test_f1 = f1_score(all_test_labels, all_test_predicted)
            test_auc = roc_auc_score(all_test_labels, all_test_predicted)

            wandb.log({"test/precision": test_precision, "round": round})
            wandb.log({"test/recall": test_recall, "round": round})
            wandb.log({"test/f1": test_f1, "round": round})
            wandb.log({"test/auc": test_auc, "round": round})

        wandb.log({"test/acc": avg_acc_test_all, "round": round})
        wandb.log({"test/loss": avg_loss_test_all, "round": round})

        print()
        print(f"[TRAIN] Avg acc: {avg_acc_train_all * 100:.3f}%, Avg loss: {avg_loss_train_all:.5f}")
        print(f"[TEST]  Avg acc: {avg_acc_test_all * 100:.3f}%, Avg loss: {avg_loss_test_all:.5f}")
        # print("-" * 50)
        print()

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
        all_labels = []
        all_predicted = []

        for k in tqdm(range(self.client_num_in_total)):
            surrogate = self.surrogates[k % self.client_num_per_round]  # 放到槽位上去算
            surrogate.update_local_dataset(self.clients[k])  # update dataset and params
            surrogate.set_params(self.global_params)

            # 把所有的predicted结果整合起来
            client_labels, client_predicted, client_num, accuracy, avg_loss = surrogate.test(
                dataset=dataset)  # 在本地模型上进行测试
            all_labels += client_labels
            all_predicted += client_predicted
            acc_list.append((client_num, accuracy))
            loss_list.append((client_num, avg_loss))

        return all_labels, all_predicted, acc_list, loss_list

    @staticmethod
    def avg_metric(metric_list):
        # 其实你把所有的结果加到一起再用sklearn去得到accuracy，precisison，recall等指标也是OK的
        total_num = 0
        total_metric = 0
        for (client_num, metric) in metric_list:
            total_num += client_num
            total_metric += metric * client_num
        avg_metric = total_metric / total_num
        return avg_metric
