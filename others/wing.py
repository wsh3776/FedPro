# *********tqdm常用参数*********
# - 第一个参数是可迭代对象,range(100), list, ndarray等
# - ncols: 显示总长度

from tqdm import tqdm
import time, random

# 创建tqdm对象
pbar = tqdm(range(100), ncols=100)

for i in pbar:
    # 设置tqdm的描述（推荐写法）
    pbar.set_description(f"Processing {i + 1}-th iteration")
    # 每次循环停顿一会
    time.sleep(random.random())

pbar.close() # 可以不加