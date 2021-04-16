"""
Pytorch torchvision  MNIST datasets
"""
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = torchvision.datasets.MNIST(root='../../data/', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='../../data/', train=False, download=True, transform=transform)

if __name__ == "__main__":
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print(train_data.data.shape)
    print(train_data.targets.shape)
    print(test_data.data.shape)
    print(test_data.targets.shape)
