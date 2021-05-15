import torch
import torch.nn as nn
import torch.nn.functional as F


# CTR的LR模型
# 主要是用 用户ID和物品ID，将他们进行onehot

class LR(nn.Module):
    """
    user_id, movie_id进行embedding的MLP网络模型
    user共6040个，movie共3883个
    """

    def __init__(self):
        super(LR, self).__init__()

        # self.user_id_embed = nn.Embedding(6040, 128)
        # self.movie_id_embed = nn.Embedding(3883, 128)

        # self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(9923, 1)  # usr_id 和 movie_id onehot后的长度

        self.sig = nn.Sigmoid()
        # https://zhuanlan.zhihu.com/p/59800597
        self.criterion = nn.BCELoss(reduction="mean")

    def forward(self, x):
        device = x.device.type
        batch_data = None
        for id in x.cpu().numpy():
            us, mv = torch.zeros(6040), torch.zeros(3883)
            us[int(id[0])], mv[int(id[1])] = 1, 1
            data = torch.cat((us, mv), axis=-1).unsqueeze(dim=0)

            if batch_data is None:
                batch_data = data
            else:
                batch_data = torch.cat((batch_data, data), axis=0)

        x = batch_data.to(device)
        # x = x.float()
        # user_embed = self.user_id_embed(x[:, 0].long())  # 必须是long类型
        # movie_embed = self.movie_id_embed(x[:, 1].long())
        # x = torch.cat((user_embed, movie_embed), axis=-1)

        # 可以在这里进行onehot，节省时间
        logit = self.fc(x)
        output = self.sig(logit)
        return torch.cat((1 - output, output), dim=-1)

    def cal_loss(self, pred, target):
        """Calculate loss"""
        # 这里的pred指sigmoid后的值,即 1/(1+exp(-z))
        return self.criterion(pred[:, 1].squeeze(dim=-1), target.float())


class LR_Test(nn.Module):
    def __init__(self, features=41, output=5):
        super().__init__()
        self.features = features
        self.output = output
        self.fc = nn.Linear(features, self.output)

        # 手动全连接层
        self.w = nn.Parameter(torch.randn((features, output), requires_grad=True))
        # self.register_parameter('w', self.w)

        self.b = nn.Parameter(torch.randn((1, output), requires_grad=True))
        # self.register_parameter('b', self.b)

        print('查看模型参数：', self._parameters, end='\n')

    def forward(self, x):
        # x = self.fc(x)
        x = torch.matmul(x, self.w) + self.b
        return torch.sigmoid(x)


if __name__ == '__main__':
    x = torch.randn(4, 12)
    model = LR_Test(12, 1)
    print(model(x))
