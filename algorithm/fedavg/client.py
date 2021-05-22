import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
import wandb
from base import Metrics


class Client:
    def __init__(self, user_id, train_dataloader=None, test_dataloader=None,
                 model=None, epoch=10, lr=0.01, lr_decay=0.998, decay_step=20, optimizer='sgd', device='cuda'):
        self.user_id = user_id
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model  # 创建本地模型
        self.epoch = epoch
        self.lr = lr
        self.lr_decay = lr_decay
        self.decay_step = decay_step
        self.optimizer = optimizer
        self.device = device

    def update_local_dataset(self, client):
        # 传进来一个被选择的模型client，用他的属性更新当前槽位surrogate的属性
        self.train_dataloader = client.train_dataloader
        self.test_dataloader = client.test_dataloader
        # print("update local dataset")

    def set_params(self, model_params):
        # load_state_dict is deepcopy
        self.model.load_state_dict(model_params)

    def get_params(self):
        # params = model.state_dict() is shadow copy
        return copy.deepcopy(self.model.cpu().state_dict())

    def train(self, round_th):
        """本地模型训练"""
        model = self.model
        model.to(device=self.device)
        model.train()  # 使用Dropout, BatchNorm

        # 把criterion放到model内比较好
        # criterion = nn.CrossEntropyLoss(reduction='mean').to(self.device)

        if self.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(),
                                  lr=self.lr * self.lr_decay ** (round_th / self.decay_step),
                                  momentum=0.9,
                                  weight_decay=3e-4)
            # optimizer.param_groups[0]["lr"]查看学习率大小
            # sgd要写学习率衰减，但是adam中不用
            # weight_decay就是正则化里的lambda
            # 权重衰减（L2正则化）的作用
        elif self.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0)

        batch_loss = []
        for epoch in range(self.epoch):
            for inputs, labels in self.train_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = model.cal_loss(outputs, labels)  # model内包含特定loss，如交叉熵，RMSE等
                batch_loss.append(loss)
                loss.backward()
                optimizer.step()
                # print(model.cpu().state_dict()['fc1.weight'].sum())
                # exit()

                # # https://docs.wandb.ai/library#logged-with-specific-calls
                # wandb.watch(model)

        # 这个客户端上一个样本的平均loss
        sample_loss = sum(batch_loss) / len(batch_loss)

        num_samples = self.train_dataloader.sampler.num_samples

        return self.get_params(), num_samples, sample_loss

    def test(self, dataset: str):
        """
        在本地模型上对train和test数据集进行测试,返回准确率 + loss
        :param dataset: string. Including "train"/"test"
        :return:
        """
        model = self.model
        model.eval()  # 关闭Dropout, 使用测试模式的BatchNorm
        model.to(self.device)

        # 测试的时候就不需要epoch了，只要算准确率和loss就行了
        if dataset == 'train':
            dataloader = self.train_dataloader
        elif dataset == 'test':
            dataloader = self.test_dataloader
        else:
            print("\nPlease input right dataset!!!")
            exit()

        client_metrics = Metrics()

        sample_nums_per_batch_list = []
        total_loss_per_batch_list = []

        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = model(images)
                loss = model.cal_loss(output, labels)  # average loss per sample for this batch
                _, predicted = torch.max(output.data, 1)

                sample_nums_per_batch_list.append(labels.size(0))
                total_loss_per_batch_list.append(loss * labels.size(0))

                client_metrics.labels_list += (list(labels.cpu().numpy()))
                client_metrics.predicted_list += (list(predicted.cpu().numpy()))
                client_metrics.prob_list += list(output[:, 1].cpu().numpy())

        client_metrics.num = sum(sample_nums_per_batch_list)
        client_metrics.loss = sum(total_loss_per_batch_list) / sum(sample_nums_per_batch_list)
        return client_metrics
