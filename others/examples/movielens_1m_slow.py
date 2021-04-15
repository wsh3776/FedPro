"""
这个文件是处理movielens-1m数据集的第一个文件
这个文件运行完后会输出一个很大的csv文件，里面已经整合了user，movie和rating的所有信息
改进的方法：见data_preprocessing/movielens/movielens/datasets.py
"""
import pandas as pd

from pathlib import Path

"""这种path的写法似乎有点问题吧，在别的地方运行会出错
# Get absolute path.
p = Path()
# absolute path
ap = str(p.resolve()).replace('\\', '/')
path = ap + r'/data/MovieLens/movielens'
print(ap)
"""
path = r'../../../data/MovieLens/movielens'

# users information
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table(f'{path}/users.dat', sep="::", header=None, names=unames, engine='python')

# ratings information
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(f'{path}/ratings.dat', sep='::', header=None, names=rnames, engine='python')

# movies information
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table(f'{path}/movies.dat', sep='::', header=None, names=mnames, engine='python')

print(users.head())
print(ratings.head())
print(movies.head())

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

# 跨越三个表格分析数据并不是一件简单的事情，而将所有表格合并到单个表中会容易很多
# If you merge all three tables into one table, Data analysis will become easier
data = pd.merge(pd.merge(ratings, users), movies)

data.head()
"""
   user_id  movie_id  rating  timestamp gender  age  occupation    zip  \
0        1      1193       5  978300760      F    1          10  48067   
1        1        48       5  978824351      F    1          10  48067   
2        1       938       4  978301752      F    1          10  48067   
3        1      1207       4  978300719      F    1          10  48067   
4        1      1721       4  978300055      F    1          10  48067   
                                    title  \
0  One Flew Over the Cuckoo's Nest (1975)   
1                       Pocahontas (1995)   
2                             Gigi (1958)   
3            To Kill a Mockingbird (1962)   
4                          Titanic (1997)   
                                 genres  
0                                 Drama  
1  Animation|Children's|Musical|Romance  
2                               Musical  
3                                 Drama  
4                         Drama|Romance  

"""

# 同一部电影的movie_id是一样的，所以其实可以去掉电影名这一列
# data.drop(columns=['title'], inplace=True)


# 把occupation转为具体名称
data.insert(7, 'occupation_detail', None)

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
    data.loc[data['occupation'] == i, 'occupation_detail'] = occupation_details[i]

data.rename(columns={'occupation': 'occupation_idx', 'occupation_detail': 'occupation'})

# TODO: genres按|分割 （这一步有办法加速吗？暂时保存到csv文件下次直接读取）
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
data[genres_list] = 0

from tqdm import tqdm
import time, random

# 创建一个tqdm对象
pbar = enumerate(tqdm(data['genres'], desc="Processing Bar: ", ncols=100))
for i , genre in pbar:
    genre.split('|')
    data.loc[i,genre] = 1
pbar.close()


# 把处理好的数据保存到csv文件里
# 由于处理得时间较长，下次建议直接在下面csv的基础上进行操作使用
data.to_csv("movielens-movielens-data.csv", index=0)  # index=0不保存行索引, head=0不保存行索引

# print(data.head())
print("Proprocessing End!")

# TODO: 2.考虑把评分变成CTR 0,1

# TODO: 3. 加入tilte年份的因素
