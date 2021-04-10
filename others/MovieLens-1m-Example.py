"""
MovieLens 1M 数据集处理示范
参考《利用Python进行数据分析》P392
"""
import pandas as pd

path = r'../data/MovieLens/1m'

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

# sort by user_id
data.sort_values(by='user_id', ascending=True, ignore_index=True, inplace=True)

# 计算按性别分级的每部电影的平均电影评分
mean_ratings = data.pivot_table('rating', index=["title"], columns=["gender"], aggfunc='mean')
"""
gender                                             F         M
title                                                         
$1,000,000 Duck (1971)                      3.375000  2.761905
'Night Mother (1986)                        3.388889  3.352941
'Til There Was You (1997)                   2.675676  2.733333
'burbs, The (1989)                          2.793478  2.962085
...And Justice for All (1979)               3.828571  3.689024
"""

# 过滤掉评分个数不超过250的电影
ratings_by_title = data.groupby('title').size()
# 统计每个分组rating的平均值
data.groupby('title').mean()['rating']
data.groupby('title').mean()['rating'].describe()

# 筛选出评价个数超过250的所有电影的名称
active_titles = ratings_by_title.index[ratings_by_title > 250]
mean_ratings = mean_ratings.loc[active_titles]

# 得到女性观众top电影（已经过滤掉哪些评价少的了）
top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)
"""
gender                                                     F         M
title                                                                 
Close Shave, A (1995)                               4.644444  4.473795
Wrong Trousers, The (1993)                          4.588235  4.478261
Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)       4.572650  4.464589
Wallace & Gromit: The Best of Aardman Animation...  4.563107  4.385075
Schindler's List (1993)                             4.562602  4.491415
...
"""

# 测试评价分歧
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff', ascending=True)
"""
gender                                         F         M      diff
title                                                               
Dirty Dancing (1987)                    3.790378  2.959596 -0.830782
Jumpin' Jack Flash (1986)               3.254717  2.578358 -0.676359
Grease (1978)                           3.975265  3.367041 -0.608224
Little Women (1994)                     3.870588  3.321739 -0.548849
Steel Magnolias (1989)                  3.901734  3.365957 -0.535777
"""

# 获得观众中引起最大争议的电影
# Q:怎么衡量争议？A:用评分的方差或标准差来衡量
data.groupby('title')['rating'].std()[active_titles].sort_values(ascending=False)
