import torch
import torch.nn as nn
import torch.nn.functional as F

# 命名不规范，最好改掉，类名驼峰命名法
class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.cov1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.cov2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 1x28x28
        x = F.relu(self.cov1(x), inplace=True)
        # 32x24x24
        x = F.max_pool2d(x, 2, 2)
        # 32x12x12
        x = F.relu(self.cov2(x), inplace=True)
        # 64x8x8
        x = F.max_pool2d(x, 2, 2)
        # 64x4x4
        x = x.flatten(start_dim=1)
        # [1024]
        x = F.relu(self.fc1(x), inplace=True)
        # 512
        x = self.fc2(x)
        # 10
        return x


if __name__ == '__main__':
    model = MNIST()
    x = torch.rand((50, 1, 28, 28))
    output = model(x)
    print(f'{x.shape} -> {output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))