from models.fedavg.mnist import MNIST
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
import wandb


class Client():
    def __init__(self, user_id, train_dataloader=None, test_dataloader=None,
                 model=None, epoch=10, lr=0.01, device='cuda'):
        self.user_id = user_id
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model  # 创建本地模型
        self.epoch = epoch
        self.lr = lr
        self.device = device

    # def select_model(self, model_name):
    #     model = None
    #     if model_name == 'cnn':
    #         model = mnist()
    #     return model

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
        return self.model.cpu().state_dict()

    def train(self):
        """
        本地模型训练
        :return:
        """
        model = self.model
        model.to(device=self.device)
        model.train()  # 使用Dropout, BatchNorm

        criterion = nn.CrossEntropyLoss(reduction='mean')
        # optimizer = optim.SGD(model.parameters(), lr=self.lr)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)


        batch_loss = []
        for epoch in range(self.epoch):
            pbar = enumerate(tqdm(self.train_dataloader, desc="Training Bar"), 0)
            for i, (inputs, labels) in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # loss是对每个样本的loss做了平均
                batch_loss.append(loss)
                loss.backward()
                optimizer.step()

                # # https://docs.wandb.ai/library#logged-with-specific-calls
                # wandb.watch(model)

                # if i % 300 == 0:
                #     print(f"this is {i}th batch loss: {loss.item():.6f}")

        # 这个客户端上一个样本的平均loss
        sample_loss = sum(batch_loss) / len(batch_loss)

        # 返回训练好的参数和该客户端数据个数
        return self.get_params(), \
               self.train_dataloader.sampler.num_samples, \
               sample_loss

    def test(self, dataset: str):
        """
        在本地模型上进行测试,准确率 + loss
        Args:
            dataset: 'train', 'test'

        Returns:

        """
        model = self.model
        model.eval()  # 不使用Dropout, BatchNorm
        model.to(self.device)

        # 测试的时候就不需要epoch了，只要算准确率和loss就行了
        if dataset == 'train':
            dataloader = self.train_dataloader
        elif dataset == 'test':
            dataloader = self.test_dataloader
        else:
            print("\nPlease input right dataset!!!")
            exit()

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        num_correct = 0
        client_num = 0
        batch_loss = []

        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                client_num += labels.size(0)
                # print(labels) # tensor([6, 6, 1, 6])
                # print(predicted) # tensor([4, 6, 5, 4])
                # print((predicted == labels).sum()) # tensor(1)
                batch_loss.append(loss)
                num_correct += (predicted == labels).sum().item()

        return client_num, num_correct / client_num, sum(batch_loss) / len(batch_loss)
        # print('Accuracy of the network on the 10000 test images: %d %%' % (
        #         100 * correct / total))
