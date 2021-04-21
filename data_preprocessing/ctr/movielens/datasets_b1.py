"""
_b1表示备份文件1
这个文件是处理movielens-1m数据集的（用于模拟联邦点击率预测ctr实验）
（这里我把评分1-5中的1-2转为未点击，3-5转为点击）
在别的模块可以直接用from xxx/datasets import users, movies, ratings, all_data的方式导入
"""
import pandas as pd
from tqdm import tqdm
import os
import time, random

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

# 性别'F','M'转为0,1
users['gender'] = users['gender'].apply(lambda x : 0 if x == 'F' else 1)


# 把occupation转为具体名称
users.insert(4, 'occupation_detail', None)
# 对应0-20
occupation_details = ["other",
                      "academic/educator",
                      "artist",
                      "clerical/admin",
                      "college/grad student",
                      "customer service",
                      "doctor/health care",
                      "executive/managerial",
                      "farmer",
                      "homemaker",
                      "K-12 student",
                      "lawyer",
                      "programmer",
                      "retired",
                      "sales/marketing",
                      "scientist",
                      "self-employed",
                      "technician/engineer",
                      "tradesman/craftsman",
                      "unemployed",
                      "writer"]

for i in range(len(occupation_details)):
    users.loc[users['occupation'] == i, 'occupation_detail'] = occupation_details[i]
users.rename(columns={'occupation': 'occupation_idx', 'occupation_detail': 'occupation'}, inplace=True)

# 对occupation这一列进行onehot编码
users = users.join(pd.get_dummies(users['occupation']))

# *******************************************
# *************** movies ********************
# *******************************************

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table(f'{path}/movies.dat', sep='::', header=None, names=mnames, encoding='ISO-8859-1', engine='python')

# 从电影title中提取出电影的年份year
movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=False)

# 将分割genres，转换为onehot
genres_list = ["Action",
               "Adventure",
               "Animation",
               "Children's",
               "Comedy",
               "Crime",
               "Documentary",
               "Drama",
               "Fantasy",
               "Film-Noir",
               "Horror",
               "Musical",
               "Mystery",
               "Romance",
               "Sci-Fi",
               "Thriller",
               "War",
               "Western"]

# 增加多个列
movies[genres_list] = 0

# **** for loop is slow ****
# # 创建一个tqdm对象
# pbar = enumerate(tqdm(movies['genres'], desc="movies Processing Bar: ", ncols=100))
# for i, genre in pbar:
#     movies.loc[i, genre.split('|')] = 1


# **** apply is recommended ****
def split_genre(row):
    movies.loc[row.name, row['genres'].split('|')] = 1


movies.apply(split_genre, axis=1) # axis=1，每次得到一行数据

# *******************************************
# ************** ratings ********************
# *******************************************

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(f'{path}/ratings.dat', sep='::', header=None, names=rnames, encoding='utf-8', engine='python')

# 这里我把评分1-5中的1-2转为未点击，3-5转为点击
ratings['rating'] = ratings['rating'] - 1
# ratings.loc[ratings['rating'] < 3, 'rating'] = 0
# ratings.loc[ratings['rating'] >= 3, 'rating'] = 1

# Note: 不要上来就把三个表格合并，要把三个表格先分别处理好，如onehot，不然后续处理都是100万条数据了
# 跨越三个表格分析数据并不是一件简单的事情，而将所有表格合并到单个表中会容易很多
# If you merge all three tables into one table, Data analysis will become easier
all_data = pd.merge(pd.merge(ratings, users), movies)


if __name__ == '__main__':
    print(all_data.head())
    print()
    print(all_data.columns)

    # 把处理好的数据保存到csv文件里
    # 会保存在服务器上，本地不会及时出现
    # data.to_csv("movielens-movielens-data.csv", index=0)  # index=0不保存行索引, head=0不保存行索引
    print("Proprocessing End!")
