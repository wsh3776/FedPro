import torch
import torch.nn as nn

class FM_Layer(nn.Module):
    def __init__(self, n=10, k=5):
        """
        :param n: 特征向量的维度
        :param k: 每个特征向量的包含k个描述的因子
        """
        super(FM_Layer, self).__init__()
        self.n = n
        self.k = k
        self.linear = nn.Linear(self.n, 1)   # 线性层

        self.V = nn.Parameter(torch.randn(self.n, self.k))   # 随机初始化隐向量矩阵V
        nn.init.uniform_(self.V, -0.1, 0.1)

    def fm_layer(self, x):
        linear_part = self.linear(x)
        interaction_part_1 = torch.mm(x, self.V)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        interaction_part_2 = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))
        # torch.sum的用法：https://zhuanlan.zhihu.com/p/268162211
        # linear_part: [batch, 1], torch.sum应该keepdim
        output = linear_part + 0.5 * torch.sum(interaction_part_2 - interaction_part_1, dim=1, keepdim=True)
        return output

    def forward(self, x):
        return self.fm_layer(x)

fm = FM_Layer(n=10, k=5)
x = torch.randn(4, 10) # [batch, n]
output = fm(x)
print(output) # [4, 1]

# # 这里我全连接层是n->1，所以我输出的是一个数，因为是用在二分类中的，即我最后再加个sigmoid来表示输出为1的概率
#
# 而如果要用多分类，应该有10个parameters吧???
#
# https://github.com/rixwew/pytorch-fm/blob/master/torchfm/model/fm.py
# 这里他好像是二分类，用的是sigmoid激活函数
#
# https://blog.csdn.net/baymax_007/article/details/83931698