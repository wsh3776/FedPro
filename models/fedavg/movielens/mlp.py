import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    user_id, movie_id进行embedding的MLP网络模型
    user共6040个，movie共3883个
    """

    def __init__(self):
        super(MLP, self).__init__()
        self.user_id_embed = nn.Embedding(6040, 128)
        self.movie_id_embed = nn.Embedding(3883, 128)
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.BatchNorm1d(128),
            # nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            # nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # https://zhuanlan.zhihu.com/p/59800597
        self.criterion = nn.BCELoss(reduction="mean")

    def forward(self, x):
        user_embed = self.user_id_embed(x[:, 0].long())
        movie_embed = self.movie_id_embed(x[:, 1].long())
        x = torch.cat((user_embed, movie_embed), axis=-1)
        x = self.fc(x)
        prob_pos = torch.sigmoid(x)
        prob_neg = 1 - prob_pos

        return torch.cat((prob_neg, prob_pos), dim=-1)

    def cal_loss(self, pred, target):
        """Calculate loss"""
        # 这里的pred指sigmoid后的值,即 1/(1+exp(-z))
        return self.criterion(pred[:, 1].squeeze(dim=-1), target.float())
