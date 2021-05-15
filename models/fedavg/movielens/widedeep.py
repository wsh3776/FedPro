# https://github.com/zhongqiangwu960812/AI-RecommenderSystem/blob/master/WideDeep/Wide%26Deep%20Model.ipynb
import torch
import torch.nn as nn


class WideDeep(nn.Module):
    def __init__(self):
        super(WideDeep, self).__init__()
        user_num, movie_num = 6040, 3883
        self.user_id_embed = nn.Embedding(user_num, 128)
        self.movie_id_embed = nn.Embedding(movie_num, 128)

        self.wide_linear = nn.Linear(user_num + movie_num, 1)

        self.deep_dnn = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        # https://zhuanlan.zhihu.com/p/59800597
        self.criterion = nn.BCELoss(reduction="mean")

    def forward(self, x):
        user_embed = self.user_id_embed(x[:, 0].long())
        movie_embed = self.movie_id_embed(x[:, 1].long())

        # deep 网络
        deep_input = torch.cat((user_embed, movie_embed), axis=-1)

        deep_out = self.deep_dnn(deep_input)

        # wide 网络
        device = x.device.type
        batch_data = None
        for input in x.cpu().numpy():
            us, mv = torch.zeros(6040), torch.zeros(3883)
            us[int(input[0])], mv[int(input[1])] = 1, 1
            data = torch.cat((us, mv), axis=-1).unsqueeze(dim=0)

            if batch_data is None:
                batch_data = data
            else:
                batch_data = torch.cat((batch_data, data), axis=0)

        wide_input = batch_data.to(device)
        wide_out = self.wide_linear(wide_input)

        # x = self.fc(x)
        prob_pos = torch.sigmoid(0.5 * (wide_out + deep_out))
        prob_neg = 1 - prob_pos

        return torch.cat((prob_neg, prob_pos), dim=-1)

    def cal_loss(self, pred, target):
        """Calculate loss"""
        # 这里的pred指sigmoid后的值,即 1/(1+exp(-z))
        return self.criterion(pred[:, 1].squeeze(dim=-1), target.float())


class TT(nn.Module):
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0.):
        super(TT, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns

        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        hidden_units.insert(0,
                            len(self.dense_feature_cols) + len(self.sparse_feature_cols) * self.sparse_feature_cols[0][
                                'embed_dim'])
        self.dnn_network = Dnn(hidden_units)
        self.linear = Linear(len(self.dense_feature_cols))
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        dense_input, sparse_inputs = x[:, :len(self.dense_feature_cols)], x[:, len(self.dense_feature_cols):]
        sparse_inputs = sparse_inputs.long()
        sparse_embeds = [self.embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in
                         range(sparse_inputs.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)

        dnn_input = torch.cat([sparse_embeds, dense_input], axis=-1)

        # Wide
        wide_out = self.linear(dense_input)

        # Deep
        deep_out = self.dnn_network(dnn_input)
        deep_out = self.final_linear(deep_out)

        # out
        outputs = F.sigmoid(0.5 * (wide_out + deep_out))

        return outputs


if __name__ == '__main__':
    x = torch.randint(3000, (10, 2))
    print(x)
    model = WideDeep()
    # torch.set_printoptions(profile="full")
    print(model(x))
    torch.set_printoptions(profile="default")
