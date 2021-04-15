import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class CNNOriginalFedAvg(torch.nn.Module):
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 512)               1606144
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only mnist dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended mnist (FEMNIST)
        EMNIST: Extending mnist to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, data_size, num_classes):
        super(CNNOriginalFedAvg, self).__init__()
        input_channel = data_size[0]
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            Reshape()  # 自定义的类，可以作为一层
        )
        x = torch.zeros([4] + data_size)
        x = self.feature(x)

        num_features = x.shape[1]
        self.linear_1 = nn.Linear(num_features, 512)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


def main():
    # 根据你数据的大小，来调整
    model = CNNOriginalFedAvg([3, 30, 30], 10)
    print(model)
    x = torch.rand((4, 3, 30, 30))
    print(model(x).shape)


if __name__ == '__main__':
    main()
