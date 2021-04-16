from skimage.measure import label
import torchvision
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from tqdm import tqdm
import warnings
from torch import optim

from tqdm import tqdm
import random


def get_data(indexList, dataList, max_age):
    batch_data = [dataList[item][0] for item in indexList]
    label_list = [dataList[item][1] for item in indexList]

    batch_data = np.array(batch_data)
    batch_label = np.array(label_list)

    feature_norm = np.array([1, 1, max_age, 1, 1, 1, 1, 1, 1]).reshape(1, 9)

    return batch_data / feature_norm, batch_label


#### getBatch
def getBatchIndex(data, batchSize, shuffle=False):
    data_index = list(range(len(data)))

    if shuffle == True:
        random.shuffle(data_index)
    batchList = []
    tmp = []
    total_area = 0
    for i in data_index:
        tmp.append(i)
        total_area += data[i][-1]
        if len(tmp) == batchSize:
            batchList.append(tmp)
            tmp = []
            total_area = 0
    if len(tmp) > 0:
        batchList.append(tmp)
    return batchList


class Classify(nn.Module):
    """
    Encoder.
    """

    def __init__(self):
        super(Classify, self).__init__()

        self.movie_id_embed = nn.Embedding(4000, 128) # 这个就是你这个特征所有可能取值的个数，如果多几个的话，就不会得到训练
        self.gender_id_embed = nn.Embedding(2, 128)
        self.occupation_id_embed = nn.Embedding(30, 128)
        self.zip_id_embed = nn.Embedding(4000, 128)
        self.genres_id_embed = nn.Embedding(20, 128)  # 一部电影的类别：['drama', 'comedy', 'xxx']

        self.lr1 = nn.Linear(128 * 8 + 1, 1024)
        self.lr2 = nn.Linear(1024, 512)
        self.lr3 = nn.Linear(512, 128)
        self.lr4 = nn.Linear(128, 2)

    def forward(self, x):
        # movie_id = self.movie_id_embed(x[:, 0].long()) # 拿出Embedding矩阵的某一行 [32] -> [32, 128]
        gender_id = self.gender_id_embed(x[:, 0].long())
        age = x[:, 1].unsqueeze(-1) # [32] ->[32, 1]
        occ_id = self.occupation_id_embed(x[:, 2].long())
        # zip_id = self.zip_id_embed(x[:, 4].long())

        genres_id_0 = self.genres_id_embed(x[:, 3].long())
        genres_id_1 = self.genres_id_embed(x[:, 4].long())

        genres_id_2 = self.genres_id_embed(x[:, 5].long())
        genres_id_3 = self.genres_id_embed(x[:, 6].long())
        genres_id_4 = self.genres_id_embed(x[:, 7].long())
        genres_id_5 = self.genres_id_embed(x[:, 8].long())

        feature = torch.cat((gender_id, age, occ_id,
                             genres_id_0, genres_id_1, genres_id_2, genres_id_3, genres_id_4, genres_id_5), dim=-1) # 按照最后一个维度拼接起来

        out = F.relu(self.lr1(feature))
        out = F.relu(self.lr2(out))
        out = F.relu(self.lr3(out))
        out = self.lr4(out)
        """ x[:, 4]
           tensor([ 0.,  0., 12., 12.,  8.,  0.,  0., 10.,  0., 12., 10.,  0., 16.,  0.,
            5.,  0.,  5., 10.,  0.,  1.,  3.,  5.,  1., 13., 16., 10.,  8.,  6.,
            0.,  5., 16., 13.], device='cuda:0')
        """
        return out

f = open('movielens_row_data.csv').readlines()[1:]
user_id_dic = {}
movie_id_dic = {}
gender_dic = {}
occupation_dic = {}
zip_dic = {}
# title_dic = {}
genres_dic = {}
genres_dic['NULL'] = len(genres_dic)

# 对每一维特征创建字典
for item in tqdm(f):
    user_id_tmp = item.strip().split(',')[0]
    movie_id_tmp = item.strip().split(',')[1]
    rating_tmp = item.strip().split(',')[2]
    # times_tmp = item.strip().split(',')[3]
    gender_tmp = item.strip().split(',')[4]
    age_tmp = item.strip().split(',')[5]
    occupation_tmp = item.strip().split(',')[6]
    zip_tmp = item.strip().split(',')[7]
    # title_tmp = item.strip().split(',')[8]
    genres_tmp = item.strip().split(',')[-1].split('|')

    if user_id_tmp not in user_id_dic:
        user_id_dic[user_id_tmp] = len(user_id_dic)

    if movie_id_tmp not in movie_id_dic:
        movie_id_dic[movie_id_tmp] = len(movie_id_dic)

    if gender_tmp not in gender_dic:
        gender_dic[gender_tmp] = len(gender_dic)

    if occupation_tmp not in occupation_dic:
        occupation_dic[occupation_tmp] = len(occupation_dic)

    if zip_tmp not in zip_dic:
        zip_dic[zip_tmp] = len(zip_dic)

    # if title_tmp not in title_dic:
    #     title_dic[title_tmp] = len(title_dic)

    for g in genres_tmp:
        if g not in genres_dic:
            genres_dic[g] = len(genres_dic)

train_count = int(len(f) * 0.8)

train_data = []
test_data = []

# 打乱数据集
random.shuffle(f)

max_age = 0

for item in tqdm(f[:train_count]):
    # user_id_tmp = item.strip().split(',')[0]
    movie_id_tmp = item.strip().split(',')[1]
    rating_tmp = item.strip().split(',')[2]
    # times_tmp = item.strip().split(',')[3]
    gender_tmp = item.strip().split(',')[4]
    age_tmp = int(item.strip().split(',')[5])
    occupation_tmp = item.strip().split(',')[6]
    zip_tmp = item.strip().split(',')[7]
    # title_tmp = item.strip().split(',')[8]
    genres_tmp = item.strip().split(',')[-1].split('|')

    if age_tmp > max_age:
        max_age = age_tmp

    feature_tmp = [
        # movie_id_dic[movie_id_tmp],
        gender_dic[gender_tmp],
        age_tmp,
        occupation_dic[occupation_tmp]]

    gen_feature = []
    for g in genres_tmp:
        gen_feature.append(genres_dic[g])
    while len(gen_feature) < 6:
        gen_feature.append(genres_dic['NULL'])

    feature_tmp = feature_tmp + gen_feature

    # if len(feature_tmp) > 10:
    #     print(item)

    if int(rating_tmp) < 3:
        label_tmp = 0
    else:
        label_tmp = 1

    train_data.append([feature_tmp, label_tmp])

for item in tqdm(f[-(len(f) - train_count):]):
    # user_id_tmp = item.strip().split(',')[0]
    movie_id_tmp = item.strip().split(',')[1]
    rating_tmp = item.strip().split(',')[2]
    # times_tmp = item.strip().split(',')[3]
    gender_tmp = item.strip().split(',')[4]
    age_tmp = int(item.strip().split(',')[5])
    occupation_tmp = item.strip().split(',')[6]
    zip_tmp = item.strip().split(',')[7]
    # title_tmp = item.strip().split(',')[8]
    genres_tmp = item.strip().split(',')[-1].split('|')

    feature_tmp = [
        # movie_id_dic[movie_id_tmp],
        gender_dic[gender_tmp],
        age_tmp,
        occupation_dic[occupation_tmp]]

    gen_feature = []
    for g in genres_tmp:
        gen_feature.append(genres_dic[g])
    while len(gen_feature) < 6:
        gen_feature.append(genres_dic['NULL'])

    feature_tmp = feature_tmp + gen_feature

    if int(rating_tmp) < 3:
        label_tmp = 0
    else:
        label_tmp = 1

    test_data.append([feature_tmp, label_tmp])


def get_ace_metric(predict, label):
    predict = torch.argmax(predict, dim=-1).tolist()
    label = label.tolist()

    ratio_list = []

    for i in range(len(predict)):
        if predict[i] == label[i]:
            ratio_list.append(1)
        else:
            ratio_list.append(0)

    return sum(ratio_list) / len(ratio_list)
cly = Classify().cuda()
criterion = nn.CrossEntropyLoss().cuda()

BATCH_SIZE = 32
params = [p for p in cly.parameters() if p.requires_grad]

learningRate = 3e-4

best_val = 0
exit_count = 0
totalCount = 0

while True:
    optimizer = optim.Adam(params, lr=learningRate)
    cly.train()

    loss_list_train = []
    acc_train = []

    batch_index_tr = getBatchIndex(train_data, BATCH_SIZE, shuffle=True)
    random.shuffle(batch_index_tr)

    for index_tr in tqdm(batch_index_tr):
        batch_data_tr, batch_label_tr \
            = get_data(index_tr, train_data, max_age)

        batch_data_tr = (torch.tensor(batch_data_tr)).cuda().float()
        predict = cly(batch_data_tr)

        acc_tmp = get_ace_metric(predict, batch_label_tr)
        acc_train.append(acc_tmp)

        batch_label_tr = torch.tensor(batch_label_tr).cuda()

        loss = criterion(predict, batch_label_tr)

        loss_list_train.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = sum(loss_list_train) / len(loss_list_train)
    train_acc = (sum(acc_train) / len(acc_train)) * 100

    with torch.no_grad():
        cly.eval()

        loss_list_val = []
        acc_val = []

        batch_index_val = getBatchIndex(test_data, BATCH_SIZE, shuffle=True)
        random.shuffle(batch_index_val)

        for index_val in tqdm(batch_index_val):
            batch_data_val, batch_label_val \
                = get_data(index_val, test_data, max_age)

            batch_data_val = (torch.tensor(batch_data_val)).cuda().float()
            predict = cly(batch_data_val)

            acc_tmp = get_ace_metric(predict, batch_label_val)
            acc_val.append(acc_tmp)

            batch_label_val = torch.tensor(batch_label_val).cuda()

            loss = criterion(predict, batch_label_val)

            loss_list_val.append(loss.item())

        val_loss = sum(loss_list_val) / len(loss_list_val)
        val_acc = (sum(acc_val) / len(acc_val)) * 100

    if train_acc > best_val:
        # torch.save(cly.state_dict(), "model/cly.pkl")

        best_val = train_acc
        exit_count = 0
    else:
        if best_val > 0:
            exit_count += 1

        if exit_count > 0 and exit_count % 3 == 0:
            learningRate *= 0.5

    if exit_count == 10:
        exit()

    print("Epoch: ", totalCount,
          "\t 训练集:", "train_loss:", train_loss, "train_acc:", round(train_acc, 3),
          "\t　验证集", "val_loss:", val_loss, "val_acc:", round(val_acc, 3),
          "lr:", learningRate, "best:", best_val, "exit_count:",
          exit_count
          )

    totalCount += 1
