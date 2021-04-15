import torch
import torch.nn as nn

torch.manual_seed(42)  # 固定随机初始化参数的种子
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)  # 词表长度为10，每个词用一个3维向量表示
print(embedding.weight)
"""
Parameter containing:
tensor([[ 1.9269,  1.4873,  0.9007],
        [-2.1055,  0.6784, -1.2345],
        [-0.0431, -1.6047, -0.7521],
        [ 1.6487, -0.3925, -1.4036],
        [-0.7279, -0.5594, -2.3169],
        [-0.2168, -1.3847, -0.8712],
        [-0.2234,  1.7174,  0.3189],
        [-0.4245, -0.8286,  0.3309],
        [-1.5576,  0.9956, -0.8798],
        [-0.6011, -1.2742,  2.1228]], requires_grad=True)
"""

"""
0: "K-12 student"
1: "academic"
2: "artist"
3: "clerical"
4: "college/grad student"
- 5: "customer service"
- - 6: "doctor/health care"
7: "executive/managerial"
8: "farmer"
9: "homemaker"
"""

# 你可以通过传入一个数值为0~9的tensor去得到对应的词向量表示
# 注意：访问的下标 < num_embeddings
print(embedding(torch.tensor([1, 0])))  # 取出词表的下标为1的词向量，和下标为0的词向量
"""
tensor([[-2.1055,  0.6784, -1.2345],
        [ 1.9269,  1.4873,  0.9007]], grad_fn=<EmbeddingBackward>)
"""

# 输入两个句子 我们可以得到每个单词的词向量表达
# ["academic"  "clerical" "K-12 student"] <=> [1, 3, 0]
# ["K-12 student", "customer service", "academic"]
print(embedding(torch.tensor([[1, 3, 0], [0, 5, 1]])))
"""
tensor([[[-2.1055,  0.6784, -1.2345],
         [ 1.6487, -0.3925, -1.4036],
         [ 1.9269,  1.4873,  0.9007]],

        [[ 1.9269,  1.4873,  0.9007],
         [-0.2168, -1.3847, -0.8712],
         [-2.1055,  0.6784, -1.2345]]], grad_fn=<EmbeddingBackward>)
"""

# 下标不能越界，否则会报错
# print(embedding(torch.tensor([12, 3])))
