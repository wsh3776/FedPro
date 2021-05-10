"""
这个文件是处理movielens-1m数据集的（用于模拟联邦点击率预测ctr实验）
（这里我把评分1-5中的1-2转为未点击，3-5转为点击）

这个处理的数据，最后其实只用到了ratings和get_negative_samples_per_user方法
"""
import pandas as pd
from tqdm import tqdm
import os
import random
import numpy as np
from random import sample


def get_ctr_movielens_datasets():
    # path: https://ugirc.blog.csdn.net/article/details/115645345
    # os.path.dirname(__file__) 获得当前模块的绝对路径
    # 用os.path.join可以返回上一级..
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")) + '/data/MovieLens/1m'
    # 不建议用os.getcwd()

    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 100)

    # *******************************************
    # **************** users ********************
    # *******************************************

    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_table(f'{path}/users.dat', sep="::", header=None, names=unames, encoding='utf-8', engine="python")

    users = users.join(pd.get_dummies(users['gender'], prefix="gender"))
    users.drop(columns=['gender'], inplace=True)

    users = users.join(pd.get_dummies(users['age'], prefix="age"))
    users.drop(columns=['age'], inplace=True)

    # zip暂时不用
    users.drop(columns=['zip'], inplace=True)

    users = users.join(pd.get_dummies(users['occupation'], prefix="occupation"))
    users.drop(columns=['occupation'], inplace=True)

    # 性别'F','M'转为0,1
    # users['gender'] = users['gender'].apply(lambda x: 0 if x == 'F' else 1)

    # *******************************************
    # *************** movies ********************
    # *******************************************

    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table(f'{path}/movies.dat', sep='::', header=None, names=mnames, encoding='ISO-8859-1',
                           engine='python')

    # 从电影title中提取出电影的年份year
    movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=False)
    # 对year进行one_hot的话，会有81维
    movies = movies.join(pd.get_dummies(movies['year'], prefix="year"))
    movies.drop(columns=['year'], inplace=True)

    # 不用电影的title
    movies.drop(columns='title', inplace=True)

    # 将分割genres，转换为multi-hot
    genres_list = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
                   "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
                   "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

    # 增加多个列
    movies[genres_list] = 0

    # **** for loop is slow ****
    # # 创建一个tqdm对象
    # pbar = enumerate(tqdm(movies['genres'], desc="movies Processing Bar: ", ncols=100))
    # for i, genre in pbar:
    #     movies.loc[i, genre.split('|')] = 1

    # apply传入多个参数
    movies.apply(lambda x: split_genre(row=x, df=movies), axis=1)  # axis=1，每次得到一行数据

    movies.drop(columns='genres', inplace=True)

    # *******************************************
    # ************** ratings ********************
    # *******************************************

    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table(f'{path}/ratings.dat', sep='::', header=None, names=rnames, encoding='utf-8',
                            engine='python')

    all_data = pd.merge(pd.merge(ratings, users), movies)
    return users, movies, ratings, all_data


# 负采样 按照比例进行负采样


def get_negative_samples_per_user(users=None, movies=None, ratings=None, ratio_of_neg_to_pos=1):
    """
    用于生成负样本 numpy a little fast
    Args:
        ratings:
        movies:
        users:
        ratio_of_neg_to_pos: 正负样本的比例

    Returns: 所有负样本的构成的ratings表

    """
    items = list(movies['movie_id'])  # Note: movie_id 不是连续的，中间有缺失, 总共3883部，但movie_id最大3952
    total_negative_item_array = []

    # 建立一个字典 {用户：[看过的电影]}
    occurrence_matrix_dic = {}

    for line in ratings.itertuples():  # itertuples is very fast
        user_id = line[1]
        movie_id = line[2]
        # ratings = line[3]
        if user_id not in occurrence_matrix_dic:
            occurrence_matrix_dic[user_id] = [movie_id]
        else:
            occurrence_matrix_dic[user_id] += [movie_id]

    for user_id in tqdm(range(1, len(users) + 1), desc=f"Generate negative items (1:{ratio_of_neg_to_pos})"):
        positive_items = set(occurrence_matrix_dic[user_id])
        num_positive_items = len(positive_items)

        # 时间的瓶颈在于你要生成负样本的个数

        # 负样本的数量不能超过可选的数量
        num_negative_items = min(len(positive_items) * ratio_of_neg_to_pos, len(items) - num_positive_items)

        # print(num_negative_items)
        # 负样本候选列表
        items_remainder = [x for x in items if x not in positive_items]

        # cnt = 0
        # 先创建一个全为0的数组，用来添加选中的负样本
        negative_item_array = np.zeros((num_negative_items, 4))

        # 从items_remainder中随机负采样
        # https://blog.csdn.net/HappyRocking/article/details/84314313
        negative_item_list = sample(items_remainder, num_negative_items)

        for cnt, item in enumerate(negative_item_list, 0):
            negative_item_array[cnt][0] = user_id
            negative_item_array[cnt][1] = item
            negative_item_array[cnt][2] = 0

        total_negative_item_array.append(negative_item_array)

    # 把列表中的numpy数组拼接起来
    total_negative_items = np.concatenate(total_negative_item_array, axis=0)
    df_negative_items = pd.DataFrame(total_negative_items,
                                     columns=['user_id', 'movie_id', 'rating', 'timestamp'])

    return df_negative_items


# df_negative_items = get_negative_samples_per_user(ratio_of_neg_to_pos=1)
# ratings_with_negative_samples = ratings.append(df_negative_items).reset_index(drop=True)

def get_negative_samples_per_user_2(ratio_of_neg_to_pos=1):
    """
    用于生成负样本 ratings.loc very slow
    Args:
        ratio_of_neg_to_pos: 正负样本的比例

    Returns:

    """
    for user_id in tqdm(range(1, len(users) + 1), desc="Generate negative samples for each user"):
        items = list(movies['movie_id'])  # Note: movie_id 不是连续的，中间有缺失, 总共3883部，但movie_id最大3952
        positive_items = list(ratings[ratings.user_id == user_id]['movie_id'])
        num_negative_items = len(positive_items) * ratio_of_neg_to_pos

        # 负样本候选列表
        items_remainder = [x for x in items if x not in positive_items]

        cnt = 0
        id = len(ratings)
        while cnt <= num_negative_items:
            negative_item = random.choice(items_remainder)
            items_remainder.remove(negative_item)

            # 这行处理速度很慢，因为你是在Dataframe中添加一行
            ratings.loc[id] = [user_id, negative_item, 0.0, None]
            id += 1
            cnt += 1


def get_negative_samples_per_user_3(ratio_of_neg_to_pos=1):
    """
    用于生成负样本 numpy a little fast
    Args:
        ratio_of_neg_to_pos: 正负样本的比例

    Returns: 所有负样本的构成的ratings表

    """
    items = list(movies['movie_id'])  # Note: movie_id 不是连续的，中间有缺失, 总共3883部，但movie_id最大3952
    total_negative_item_array = []
    for user_id in tqdm(range(1, len(users) + 1), desc="Generate negative items"):
        positive_items = list(ratings[ratings.user_id == user_id]['movie_id'])
        num_positive_items = len(positive_items)
        # 负样本的数量不能超过可选的数量
        num_negative_items = min(len(positive_items) * ratio_of_neg_to_pos, len(items) - num_positive_items)

        # 负样本候选列表
        items_remainder = [x for x in items if x not in positive_items]

        cnt = 0
        # 先创建一个全为0的数组，用来添加选中的负样本
        negative_item_array = np.zeros((num_negative_items, 4))
        while cnt < num_negative_items:
            negative_item = random.choice(items_remainder)
            items_remainder.remove(negative_item)

            negative_item_array[cnt][0] = user_id
            negative_item_array[cnt][1] = negative_item
            negative_item_array[cnt][2] = 0
            # ratings.loc[id] = [user_id, negative_item, 0.0, None]
            cnt += 1
        total_negative_item_array.append(negative_item_array)

    # 把列表中的numpy数组拼接起来
    total_negative_items = np.concatenate(total_negative_item_array, axis=0)
    df_negative_items = pd.DataFrame(total_negative_items,
                                     columns=['user_id', 'movie_id', 'rating', 'timestamp'])

    return df_negative_items


# ratings.drop(columns='timestamp', inplace=True)

# 这里我把评分1-5中的1-2转为未点击，3-5转为点击
# ratings['rating'] = ratings['rating'] - 1
# ratings.loc[ratings['rating'] < 3, 'rating'] = 0
# ratings.loc[ratings['rating'] >= 3, 'rating'] = 1

# Note: 不要上来就把三个表格合并，要把三个表格先分别处理好，如onehot，不然后续处理都是100万条数据了
# 跨越三个表格分析数据并不是一件简单的事情，而将所有表格合并到单个表中会容易很多
# If you merge all three tables into one table, Data analysis will become easier

# **** apply is recommended ****
def split_genre(row=None, df=None):
    df.loc[row.name, row['genres'].split('|')] = 1


if __name__ == '__main__':
    users, movies, ratings, all_data = get_ctr_movielens_datasets()
    print(movies.head())
    # 把处理好的数据保存到csv文件里
    # 会保存在服务器上，本地不会及时出现
    # data.to_csv("movielens-movielens-data.csv", index=0)  # index=0不保存行索引, head=0不保存行索引
    print("Proprocessing End!")
