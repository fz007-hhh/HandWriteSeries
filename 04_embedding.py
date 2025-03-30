# 前面几节中，已经做到将一段文本转化为tokens（即词表索引数组）
# 但是每个token有大有小，并不平等，因此需要将这些token转换成统一维度的向量
# 这样一个过程可以通过embedding_table(向量表)转换实现，这个embedding_table初始是随机的，后面要通过训练获得
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1337) # 随机种子,使用相同的随机种子可以复现
size=10
n_embedding = 3 # embedding的维度
embedding_table=nn.Embedding(size,n_embedding) # size表示token数量，即有多少个元素需要进行embedding(表映射)

# 这里0~9就是token，因为之前我们是把文本处理成有序集合，token本身就是数组索引
idx=torch.arange(size)
print(embedding_table(idx))