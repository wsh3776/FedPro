"""
这个文件是处理movielens-1m数据集的
在别的模块可以直接用from dataset-1m import users, movies, ratings, all_data的方式导入
"""
import pandas as pd
from tqdm import tqdm
import time, random
from pathlib import Path

"""这种path的写法似乎有点问题吧，在别的地方运行会出错
# Get absolute path.
p = Path()
# absolute path
ap = str(p.resolve()).replace('\\', '/')
path = ap + r'/data/MovieLens/1m'
print(ap)
"""
path = r'../../../data/MovieLens/1m'

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

# *******************************************
# **************** users ********************
# *******************************************

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table(f'{path}/users.dat', sep="::", header=None, names=unames, engine='python')

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
movies = pd.read_table(f'{path}/movies.dat', sep='::', header=None, names=mnames, engine='python')

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

# 创建一个tqdm对象
pbar = enumerate(tqdm(movies['genres'], desc="Processing Bar: ", ncols=100))
for i, genre in pbar:
    movies.loc[i, genre.split('|')] = 1

# *******************************************
# ************** ratings ********************
# *******************************************

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(f'{path}/ratings.dat', sep='::', header=None, names=rnames, engine='python')

# Note: 不要上来就把三个表格合并，要把三个表格先分别处理好，如onehot，不然后续处理都是100万条数据了
# 跨越三个表格分析数据并不是一件简单的事情，而将所有表格合并到单个表中会容易很多
# If you merge all three tables into one table, Data analysis will become easier
all_data = pd.merge(pd.merge(ratings, users), movies)

if __name__ == '__main__':
    print(all_data.head())
    print()
    print(all_data.columns)

    # 把处理好的数据保存到csv文件里
    # data.to_csv("movielens-1m-data.csv", index=0)  # index=0不保存行索引, head=0不保存行索引
    print("Proprocessing End!")
