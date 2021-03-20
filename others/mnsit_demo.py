import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 训练集
trainset = torchvision.datasets.MNIST(root='../data/', train=True, download=True, transform=transform)
print(trainset.__dict__)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=1)

# 测试集
testset = torchvision.datasets.MNIST(root='../data/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=1)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

print(trainloader.__dict__)