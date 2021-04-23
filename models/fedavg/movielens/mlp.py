import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    user_id, movie_id进行embedding的MLP网络模型
    user共6040个，movie共3883个
    """

    def __init__(self, output_dim):
        super(MLP, self).__init__()

        self.user_id_embed = nn.Embedding(6040, 128)
        self.movie_id_embed = nn.Embedding(3883, 128)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 2, 128)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, x):
        # x = x.float()
        user_embed = self.user_id_embed(x[:, 0].long())  # 必须是long类型
        movie_embed = self.movie_id_embed(x[:, 1].long())
        x = torch.cat((user_embed, movie_embed), axis=-1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc3(x))
        x = self.dropout(x)
        logits = self.fc4(x)
        return logits
