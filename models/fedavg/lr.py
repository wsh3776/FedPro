import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: 最好改成含embedding的lr
# 现在用了两层的，严格来讲不是LR了
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 128)
        # 这里我把网络加深了一点
        self.linear2 = torch.nn.Linear(128, output_dim)
        # output_dim是2的时候用softmax（看成2分类任务）
        # output_dim是1的时候用sigmoid激活得到是1的概率

    def forward(self, x):
        # outputs = torch.sigmoid(self.linear(x))
        x = x.float()
        outputs = self.linear2(F.relu(self.linear(x)))
        # outputs = self.linear(x.flatten(start_dim = 1))
        return outputs
