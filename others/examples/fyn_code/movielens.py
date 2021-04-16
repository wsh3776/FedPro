"""
MovieLens 1M 数据集处理示范
参考《利用Python进行数据分析》P392
"""
import pandas as pd

path = r'../../../data/MovieLens/1m'

# users information
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table(f'{path}/users.dat', sep="::", header=None, names=unames, engine='python')

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(f'{path}/ratings.dat', sep='::', header=None, names=rnames, encoding='utf-8', engine='python')

# movies information
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table(f'{path}/movies.dat', sep='::', header=None, names=mnames, encoding='ISO-8859-1', engine='python')

print(users.head())
print(ratings.head())
print(movies.head())

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

# 跨越三个表格分析数据并不是一件简单的事情，而将所有表格合并到单个表中会容易很多
# If you merge all three tables into one table, Data analysis will become easier
data = pd.merge(pd.merge(ratings, users), movies)

data.to_csv("movielens_row_data.csv", index=0)

