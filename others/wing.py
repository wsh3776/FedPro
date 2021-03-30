dict = {'Name': 'Zara', 'Age': 7}
dict2 = {'Sex': 'female', 'Age': 18}

dict.update(dict2)
print(dict)

import copy

A = [1, 2, 3]
print(A)  # [1, 2, 3]

B = copy.copy(A) # 浅拷贝（最外层"值"会拷贝，"引用"会拷贝）
B.append(5)
print(A)  # [1, 2, 3]
print(B)  # [1, 2, 3, 5]

# Python code to demonstrate copy operations

# importing "copy" for copy operations
import copy

# initializing list 1
li1 = [1, 2, [3, 5], 4]

# using copy to shallow copy
li2 = copy.deepcopy(li1)

# original elements of list
print("The original elements before shallow copying")
for i in range(0, len(li1)):
    print(li1[i], end=" ")

print("\r")

# adding and element to new list
li2[1] = 7

# checking if change is reflected
print("The original elements after shallow copying")
for i in range(0, len(li1)):
    print(li1[i], end=" ")
# # 条件断点调试
#
# import torch
#
# print(("cuda" if torch.cuda.is_available() else "cpu"))
# a = 3
# b = 0
# print(a, b)
# assert b < a
#
#
# class A():
#     def __init__(self):
#         self.k = 2
#
#     def increase(self):
#         self.k += 1
#         return self
#
#     def reduce(self):
#         self.k -= 1
#         return self
#
#
# kt = A()
#
# kt \
#     .increase() \
#     .reduce() \
#     .reduce()
#
# print(kt.k)
# print(type(kt.increase()))
#
# a = (1, 2, 3, 4, 5)
# b = [1, 2, 3, 4, 5]
# c = "Angel"
#
# la = map(int, a)
# lc = list(map(str, c))
#
# print(la)  # <map object at 0x000001F74C211248>
# print(list(a))  # python3读取map对象
# print(lc)
#
#
# def mul(x):
#     return x * x
#
#
# list1 = [1, 2, 3, 4, 5]
# res = map(lambda x: x * 3, list1)  # lambda x : x * 3整体是个临时函数名
# res = list(res)
# print(res)  # [3, 6, 9, 12, 15]
#
#
# def tuple(x, y, z):
#     return x, y, z
#
#
# # add函数有多个参数时要加入多个iterable
# list1 = [1, 2, 3]
# list2 = [0, 1, 4]
# list3 = [3, 2, 1]
# res = list(map(tuple, list1, list2, list3))
# print(res)  # [(1, 0, 3), (2, 1, 2), (3, 4, 1)]
#
# list1 = [1, 2, 3, 4, 5, 6]
# list2 = [8, 0, 5, 2, 4, 7]
# print(list(zip(list1, list2)))  # [(1, 8), (2, 0), (3, 5), (4, 2), (5, 4), (6, 7)]
#
# res = filter(lambda zp: zp[0] < zp[1], zip(list1, list2))
# print(list(res))  # [(1, 8), (3, 5), (6, 7)]
#
# print("fffffff")
# numbers = [0, 1, 2, -3, 5, -8, 13]
#
# # 提取奇数
# result = filter(lambda x: x % 2, numbers)
# print("Odd Numbers are :", list(result))
#
# import pandas as pd
# import numpy as np
#
# df = pd.DataFrame(np.arange(24).reshape(6, 4),
#                   columns=['year', 'state', 'pop', 'height'],
#                   index=['one', 'two', 'three', 'four', 'five', 'six'])
#
# print(df.isnull())
# print(df.sort_values(by='year', ascending=True))  # 按照某一列排序
# print(df['state'].sort_values())  # 如果只有一列就不用写by了
#
# # 提取偶数
# result = filter(lambda x: x % 2 == 0, numbers)
# print("Even Numbers are :", list(result))
#
# # 提取正数
# result = filter(lambda x: x > 0, numbers)
# print("Positive Numbers are :", list(result))
#
# import math
#
#
# def is_sqr(x):
#     return math.sqrt(x) % 1 == 0
#
#
# newlist = filter(is_sqr, range(1, 101))
# print(list(newlist))  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
#
# import pandas as pd
# import numpy as np
# from numpy import nan as NaN
#
# df1 = pd.DataFrame([[1, 2, 3], [NaN, NaN, 2], [NaN, NaN, NaN], [8, 8, NaN]])
# # importing pandas as pd
# import pandas as pd
#
# # Creating the dataframe
# df = pd.DataFrame({"A": [14, 4, 5, 4, 1],
#                    "B": [5, 2, 54, 3, 2],
#                    "C": [20, 20, 7, 3, 8],
#                    "D": [14, 3, 6, 2, 6]})
#
# # Print the dataframe
# print(df.mode())

import pandas as pd
import numpy as np

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

df = pd.DataFrame(matrix, columns=list('xyz'), index=list('abc'))
# apply会对行或列运行指定的 function
print(df.apply(lambda x: x))
print(df.apply(np.square))

import pandas as pd

df = pd.DataFrame([
    ['green', 'A'],
    ['red', 'B'],
    ['blue', 'A']])

df.columns = ['color', 'class']
print(df)
print(pd.get_dummies(df))
print(df.join(pd.get_dummies(df.color)))

import torch
import copy

m1 = torch.nn.Linear(in_features=5, out_features=1, bias=True)
m2 = torch.nn.Linear(in_features=5, out_features=1, bias=True)

# m1是引用指向某块内存空间
# 浅拷贝相当于拷贝一个引用，所以他们“引用”变量的id是不一样的，指向的内存空间是一样的
ck = copy.copy(m1)
print(id(m1) == id(ck))  # False

print(m1.weight)
# Parameter containing:
# tensor([[ 0.0171,  0.4382, -0.4297,  0.4098, -0.3954]], requires_grad=True)

# state_dict is shadow copy
p = m1.state_dict()
print(id(m1.state_dict()) == id(p))  # False

# 通过引用p去修改内存空间
p['weight'][0][0] = 8.8888
# 可以看到m1指向的内存空间也被修改了
print(m1.state_dict())
# OrderedDict([('weight', tensor([[ 8.8888,  0.4382, -0.4297,  0.4098, -0.3954]])), ('bias', tensor([0.3964]))])


# deepcopy
m2.load_state_dict(p)
m2.weight[0][0] = 2.0
print(p)
# OrderedDict([('weight', tensor([[ 8.8888,  0.4382, -0.4297,  0.4098, -0.3954]])), ('bias', tensor([0.3964]))])
print(m2.state_dict())
# OrderedDict([('weight', tensor([[ 2.0000,  0.4382, -0.4297,  0.4098, -0.3954]])), ('bias', tensor([0.3964]))])

# 因为我federated框架中本地模型参数确实是浅拷贝，但是我们没有去修改这个


# Python code to demonstrate copy operations

# importing "copy" for copy operations
import copy

# initializing list 1
li1 = [1, 2, [3, 5], 4]

# using deepcopy to deep copy
li2 = copy.deepcopy(li1)

# original elements of list
print("The original elements before deep copying")
for i in range(0, len(li1)):
    print(li1[i], end=" ")

print("\r")

# adding and element to new list
li2[2][0] = 7

# Change is reflected in l2
print("The new list of elements after deep copying ")
for i in range(0, len(li1)):
    print(li2[i], end=" ")

print("\r")

# Change is NOT reflected in original list
# as it is a deep copy
print("The original elements after deep copying")

for i in range(0, len(li1)):
    print(li1[i], end=" ")

print()

A = [1, 2, 3]
print(A)

B = copy.copy(A)
B.append(5)
print(A)
print(B)

print()
