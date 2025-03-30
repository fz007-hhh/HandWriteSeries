# 首先对文本进行去重，然后生成一个有序的集合
# 这里每一个去重后的元素都可以算作一个token，可以按单个字符去重，也可以按分词去重

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337) # 设置随机种子
file_name="./SanGuoYanYi.txt"

# 读取文件
with open(file_name,"r",encoding='utf-8') as f:
  text=f.read() 

# 按单个汉字或标点符号，将文本处理成有序、不重复的列表
chars=sorted(list(set(text)))
vocab_size=len(chars)

# 字符到索引之间的映射
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}

#将字符串转成索引列表
encode=lambda str1:[stoi[c] for c in str1]
decode=lambda list1:"".join([itos[element] for element in list1])

l1=encode("天下三分")
str1=decode(l1)

print(str1)