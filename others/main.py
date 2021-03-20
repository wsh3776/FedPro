# 条件断点调试

import torch
print(("cuda" if torch.cuda.is_available() else "cpu"))
a = 3
b = 0
print(a, b)
assert b < a
print("Hello")