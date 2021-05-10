import torch
import torch.nn as nn


# Factorization Machine Model
# 这里的FM是针对MovieLens的ID进行Embedding后的结果
class FM(nn.Module):
    def __init__(self, n=10, k=5):
        """
        :param n: 特征向量x的维度（这里n其实没用了）
        :param k: 每个特征向量x_i包含k个描述因子
        """
        super(FM, self).__init__()
        self.user_id_embed = nn.Embedding(6040, 128)
        self.movie_id_embed = nn.Embedding(3883, 128)

        self.linear = nn.Linear(128 * 2, 1)  # 线性层
        self.fm_layer = FactorizationMachineLayer(128 * 2, k)

        self.sig = nn.Sigmoid()
        self.criterion = nn.BCELoss(reduction="mean")

    def forward(self, x):
        user_embed = self.user_id_embed(x[:, 0].long())  # 必须是long类型
        movie_embed = self.movie_id_embed(x[:, 1].long())
        x = torch.cat((user_embed, movie_embed), axis=-1)

        # x: [batch_size, 128 * 2]
        logit = self.linear(x) + self.fm_layer(x)
        # logit: [batch_size, 1]
        output = self.sig(logit)
        return torch.cat((1 - output, output), dim=-1)

    def cal_loss(self, pred, target):
        """Calculate loss"""
        # 这里的pred指sigmoid后的值,即 1/(1+exp(-z))
        return self.criterion(pred[:, 1].squeeze(dim=-1), target.float())


class FactorizationMachineLayer(nn.Module):
    # O(kn^2) -> O(kn) time complexity
    def __init__(self, n, k):
        super().__init__()
        self.V = nn.Parameter(torch.randn(n, k))  # 随机初始化隐向量矩阵V
        nn.init.uniform_(self.V, -0.1, 0.1)

    def forward(self, x):
        # 这里具体写下公式就懂了
        temp = torch.mm(x, self.V)  # [b, n] * [n, v]
        sum_pow_interaction = torch.pow(temp, 2)
        pow_sum_interaction = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))
        # output: [batch_size, 1]
        output = 0.5 * torch.sum(sum_pow_interaction - pow_sum_interaction, dim=1, keepdim=True)
        return output


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = FM(n=10, k=5)
    # x = torch.randn(4, 2)  # [batch, n]
    x = torch.tensor([[122, 144], [534, 23], [19, 83], [73, 23]])
    target = torch.tensor([1, 0, 0, 1])
    output = model(x)
    # output 预测一个值并且sigmoid后的结果
    print(output)  # [[0.5228], [0.3266], [0.8038], [0.4608]]
    loss = model.cal_loss(output, target)
    print(f"loss: {loss:.6f}")  # (-log(0.5228)-log(1-0.3266)-log(1-0.8038)-log(0.4608)) / 4
