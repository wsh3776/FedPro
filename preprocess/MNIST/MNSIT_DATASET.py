import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 训练集
# 如果MNSIT数据集已经存在则会直接加载
trainset = torchvision.datasets.MNIST(root='../../data/', train=True, download=True, transform=transform)
# print(trainset.__dict__)

# trainset.data.shape
# torch.Size([60000, 28, 28])
# trainset.targets.shape
# torch.Size([60000])

# 测试集
testset = torchvision.datasets.MNIST(root='../../data/', train=False, download=True, transform=transform)
# print(testset.data.shape)
# torch.Size([10000, 28, 28])

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


def MNIST_Centralized(batch_size=10):
    """
    centralized: --client_num_in_total 1, --client_num_per_round 1
    """
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    # 为了和分布式统一，这里统一返回一个列表
    train_dataloader, test_dataloader = [], []
    train_dataloader.append(trainloader)
    test_dataloader.append(testloader)
    return train_dataloader, test_dataloader


# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def plotPics(data, h=4, w=4, offset=100, filename="out.jpg"):
#     fig, axes = plt.subplots(h, w, figsize=(h + 3, w + 3))
#     # 显示多张 子图
#     for i in range(h):
#         for j in range(w):
#             index = offset + (i + 1) * (j + 1) - 1  # 选定一张图片
#             image = data[index][0]  # 图片像素 [1,28,28]
#             label = data[index][1]  # 图片标签 [1]
#
#             axes[i][j].set_axis_off()  # 不显示坐标轴
#             axes[i][j].imshow(np.transpose(image, (1, 2, 0)), cmap='viridis',
#                               interpolation='nearest')  # transpose: (channel, width, height) -> (width, height, channel)
#             axes[i][j].set_title(f"Digit:{label}")
#
#     fig.tight_layout()  # 间距
#     plt.show()  # 显示图片
#     fig.savefig(filename)  # 保存图片
#
#
# # 显示MNIST手写数字图片
# # plotPics(trainset, 4, 4, offset=55)
#
#
# # 导入需要的包
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# # 定义模型
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=0)
#         self.conv2 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=5, stride=1, padding=0)
#         self.fc1 = nn.Linear(3872, 128)
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         # inputs : x
#         # [1, 28, 28]
#         x = F.relu(self.conv1(x))
#         # [12, 26, 26]
#         x = F.relu(self.conv2(x))
#         # [32, 22, 22]
#         x = F.max_pool2d(x, 2)
#         # [32, 11, 11]
#         x = torch.flatten(x, start_dim=1)
#         # [3872]
#         x = self.fc1(x)
#         # [128]
#         logits = self.fc2(x)
#         # [10]
#         return logits
#
#
# torch.manual_seed(2)  # 保证实验可重复
#
# device = torch.device('cpu')  # 在GPU上训练模型（在我本地算力不匹配）
#
# net = Net()  # 实例化CNN模型
# net.to(device)  # 将模型转移到device
#
# import torch.optim as optim
#
# # 损失函数
# criterion = nn.CrossEntropyLoss()  # 在分类问题上，Cross-entropy通常比MSELoss要好
# # 梯度优化器
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
# # 模型训练
# cnt = 0
# interval = 5000
# print(trainloader)
# # if __name__ == '__main__':
# for epoch in range(12):  # 训练6个epoch
#     running_loss = 0.0
#     # trainloader每次读取4张图片, 15000个batch
#     for i, data in enumerate(iterable=trainloader, start=0):
#         # inputs: [batch, 1, 28, 28], labels: [batch]
#         inputs, labels = data[0].to(device), data[1].to(device)
#         # 1.梯度清零
#         optimizer.zero_grad()
#         # 2.前向传播
#         outputs = net(inputs)  # outputs: [b, 10]
#         # 3.计算损失
#         loss = criterion(outputs, labels)
#         # 4.反向传播
#         loss.backward()
#         # 5.更新参数
#         optimizer.step()
#
#         running_loss += loss.item()
#
#         if i % interval == interval - 1:  # print every 5000 mini-batches
#             cnt += 1
#             print(cnt)
#
# print('Finished Training')
