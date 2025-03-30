# 首先对文本进行去重，然后生成一个有序的集合
# 这里每一个去重后的元素都可以算作一个token，可以按单个字符去重，也可以按分词去重

import torch
import torch.nn as nn
from torch.nn import functional as F

# 超级参数
batch_size=8
block_size=5
device="cuda" if torch.cuda.is_available() else "cpu"

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

# 训练、验证分组
# 将字符转换为token索引向量
data=torch.tensor(encode(text),dtype=torch.long)
# 设置分割线，分成训练集和测试集
spilt=int(0.9*len(data))
train_data=data[:spilt]
test_data=data[spilt:]

print(f"文件{file_name}读取完成")

def get_batch(mode):
  data = train_data if mode=="train" else test_data
  # 生成batch_size个范围小于len(data)-block_size的随机数(起始点)
  # 这里第二个参数必须是tuple of ints，所以才这么写
  ix=torch.randint(len(data)-block_size,(batch_size,))
  x=torch.stack([data[i:i+block_size] for i in ix])
  # 准备一个比较草率的目标值，简单定义为原本字符串往右滑动一个字符的结果
  y=torch.stack([data[i+1:i+block_size+1] for i in ix])
  x,y=x.to(device),y.to(device)
  # token_list=x.tolist()
  # for str_list in token_list:
  #   print(decode(str_list))
  return x,y

print(get_batch("train"))