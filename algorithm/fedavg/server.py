import wandb
import copy
import numpy as np
from algorithm.fedavg.client import Client
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from data_preprocessing.dummy_data import DummyData
from data_preprocessing.mnist.data_loader import partition_data as partition_data_mnist
from data_preprocessing.movielens.ctr.data_loader import partition_data as partition_data_movielens
from models.fedavg.mnist.cnn import CNN
from models.fedavg.movielens.mlp import MLP
from models.fedavg.movielens.fm import FM
from models.fedavg.movielens.lr import LR
from models.fedavg.movielens.widedeep import WideDeep


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
        self.early_stop = args.early_stop

        self.clients: list = None
        self.agents: list = None
        self.model = None
        self.global_params = None

    @staticmethod
    def _select_model(model_name):
        model = None
        if model_name == 'cnn':
            model = CNN()
        elif model_name == 'mlp':
            # user_id, movie_id进行embedding的网络模型
            model = MLP()
        elif model_name == 'widedeep':
            model = WideDeep()
        elif model_name == 'fm':
            model = FM(n=4, k=10)
        elif model_name == 'lr':
            # 针对ctr数据集的lr
            model = LR()
        return model

    # TODO: load data
    def get_dataloader(self):
        # 多个if-else最好加个注释，可以接收哪几个参数
        datasets = {'train': None, 'test': None}
        if self.dataset == "dummy":
            train_dataloader, test_dataloader = DummyData()
        elif self.dataset == "mnist":
            train_dataloader, test_dataloader = partition_data_mnist(self.partition_method,
                                                                     self.client_num_in_total,
                                                                     self.batch_size)
        elif self.dataset == "movielens":
            train_dataloader, test_dataloader = partition_data_movielens(self.partition_method,
                                                                         self.client_num_in_total,
                                                                         self.batch_size)
        datasets['train'], datasets['test'] = train_dataloader, test_dataloader
        return datasets

    def _select_clients(self, round_th):
        np.random.seed(seed=self.seed + round_th)
        selected_clients_index = np.random.choice(self.client_num_in_total,
                                                  size=self.client_num_per_round,
                                                  replace=False)
        return selected_clients_index

    def _setup_clients(self, datasets=None):
        # setup all clients (actually we just want to store the data into client)
        # 这里不需要传递epoch,lr,model_name，因为这些每个客户端是一样的，我只要在agent里设置就行了
        train_dataloader, test_dataloader = datasets['train'], datasets['test']
        clients = [
            Client(user_id=i,
                   train_dataloader=train_dataloader[i],
                   test_dataloader=test_dataloader[i])
            for i in np.arange(self.client_num_in_total)]
        return clients

    def _setup_agents(self):
        """
        Your can use only one model to train. The only thing you need to is update the datasets and parameters for this each client
        Also you can define num_per_round models and use multiprocessing to speed up training if you want.
        """
        # Client need to update the dataset and params
        agent = [Client(user_id=i, train_dataloader=None, test_dataloader=None,
                        model=self.model, epoch=self.epoch, lr=self.lr, lr_decay=self.lr_decay,
                        decay_step=self.decay_step, optimizer=self.optimizer,
                        device=self.device)
                 for i in range(self.client_num_per_round)]
        # print(agent[0].model)
        return agent

    def _aggregate_and_update_global_params(self, updates):
        n = sum([n_k for (params, n_k) in updates])

        new_params = updates[0][0]
        for key in updates[0][0].keys():  # key: cov1.weight, cov1.bias...
            for i in range(len(updates)):
                client_params, n_k = updates[i]
                if i == 0:
                    new_params[key] = (client_params[key] * n_k).true_divide(n)
                else:
                    new_params[key] += (client_params[key] * n_k).true_divide(n)

        self.global_params = new_params
        return self

    def _train_on_clients(self, round_th, updates=[]):
        selected_clients_index = self._select_clients(round_th=round_th)
        print("-" * 50)
        print(f"Round {round_th}")
        print("train local models:")
        # print(selected_clients_index)
        for k in tqdm(range(self.client_num_per_round)):
            # 训练时只把参数发给被选中的客户端
            agent = self.agents[k]  # 放到第k个槽位上
            agent.update_local_dataset(self.clients[selected_clients_index[k]])  # update datasets and params
            agent.set_params(self.global_params)
            # 本地训练 local client training
            local_params, train_data_num, sample_loss \
                = agent.train(round_th)
            # print(local_params['fc2.weight'].sum().item())
            updates.append((local_params, train_data_num))
        return updates

    def _eval_global_model(self, dataset: str = 'test'):
        """
        评估当前的全局模型在所有客户端训练集或测试集上性能
        """
        metrics = {
            'loss': 0,
            'total_sample_nums': 0,
            'labels_list': [],  # 所有客户端数据：[1,0,0,1...1,0,1,1]
            'predicted_list': [],
            'prob_list': [],
        }

        total_loss_per_client_list = []
        sample_nums_per_client_list = []

        for k in tqdm(range(self.client_num_in_total)):
            agent = self.agents[k % self.client_num_per_round]  # 放到槽位上去算
            agent.update_local_dataset(self.clients[k])  # update dataset and params
            agent.set_params(self.global_params)

            client_metrics = agent.test(dataset=dataset)

            metrics['labels_list'] += client_metrics['labels_list']
            metrics['predicted_list'] += client_metrics['predicted_list']
            metrics['prob_list'] += client_metrics['prob_list']

            total_loss_per_client_list.append(client_metrics['client_loss'] * client_metrics['client_num'])
            sample_nums_per_client_list.append(client_metrics['client_num'])

        metrics['total_sample_nums'] = sum(sample_nums_per_client_list)
        metrics['loss'] = sum(total_loss_per_client_list) / sum(sample_nums_per_client_list)

        return metrics

    def visualize(self, metrics: dict = None, info='test', round_th=1):
        labels_list = metrics['labels_list']
        predicted_list = metrics['predicted_list']
        prob_list = metrics['prob_list']
        loss = metrics['loss']

        if self.dataset == "movielens":
            precision = precision_score(labels_list, predicted_list)
            recall = recall_score(labels_list, predicted_list)
            f1 = f1_score(labels_list, predicted_list)
            auc = roc_auc_score(labels_list, predicted_list) # TODO

            wandb.log({f"{info.title()}/precision": precision, "round": round_th})
            wandb.log({f"{info.title()}/recall": recall, "round": round_th})
            wandb.log({f"{info.title()}/f1": f1, "round": round_th})
            wandb.log({f"{info.title()}/auc": auc, "round": round_th})

        accuracy = accuracy_score(labels_list, predicted_list)

        wandb.log({f"{info.title()}/auc": accuracy, "round": round_th})
        wandb.log({f"{info.title()}/loss": loss, "round": round_th})

        print(f"[{info.upper()}] Avg acc: {accuracy * 100:.3f}%, loss: {loss:.5f}")

        if info == 'test':
            print("first 30 Ground Truth: ", labels_list[:30])
            print("first 30 Prediction:   ", predicted_list[:30])

        return self

    @staticmethod
    def avg_metric(metric_list):
        # 其实你把所有的结果加到一起再用sklearn去得到accuracy，precision，recall等指标也是OK的
        total_num = 0
        total_metric = 0
        for (client_num, metric) in metric_list:
            total_num += client_num
            total_metric += metric * client_num
        avg_metric = total_metric / total_num
        return avg_metric

    def federate(self):
        """
        FedAvg Core Function
        """
        print("Begin Federating!")
        print(f"Training among {self.client_num_in_total} clients! \n")

        self.model = self._select_model(self.model_name)

        # get the initialized global model params
        self.global_params = copy.deepcopy(self.model.state_dict())

        datasets = self.get_dataloader()

        self.clients = self._setup_clients(datasets)

        self.agents = self._setup_agents()

        min_loss = 1000
        early_stop_cnt = 0

        # Server-Client communication
        for round_th in range(self.num_rounds):
            # (1)
            updates = self._train_on_clients(round_th)
            # (2)
            self._aggregate_and_update_global_params(updates)
            if round_th % self.eval_interval == 0:
                # (3)
                print("evaluate global model:")
                train_set_metrics = self._eval_global_model(dataset='train')
                test_set_metrics = self._eval_global_model(dataset='test')

                # (4)
                self.visualize(metrics=train_set_metrics, info='train', round_th=round_th)
                self.visualize(metrics=test_set_metrics, info='test', round_th=round_th)

                test_loss = test_set_metrics['loss']
                if min_loss > test_loss:
                    min_loss = test_loss
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += self.eval_interval

            # Stop training if your model stops improving for 'early_stop' rounds.
            if early_stop_cnt >= self.early_stop:
                break
