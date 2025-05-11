# 首先对文本进行去重，然后生成一个有序的集合
# 这里每一个去重后的元素都可以算作一个token，可以按单个字符去重，也可以按分词去重

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import random
import textwrap

# 超级参数
# 每个批次放入几条字符串
batch_size = 3
# 每个字符串的长度是多少
block_size = 16
n_embd = 3
wrap_width = 20
device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)  # 设置随机种子
file_name = "./SanGuoYanYi.txt"

# 读取文件
with open(file_name, "r", encoding="utf-8") as f:
    text = f.read()

# 按单个汉字或标点符号，将文本处理成有序、不重复的列表
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 字符到索引之间的映射
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 将字符串转成索引列表
encode = lambda str1: [stoi[c] for c in str1]
decode = lambda list1: "".join([itos[element] for element in list1])

# 训练、验证分组
# 将字符转换为token索引向量
data = torch.tensor(encode(text), dtype=torch.long)
# 设置分割线，分成训练集和测试集
spilt = int(0.9 * len(data))
train_data = data[:spilt]
test_data = data[spilt:]

print(f"文件{file_name}读取完成")


def get_batch(mode):
    data = train_data if mode == "train" else test_data
    # 生成batch_size个范围小于len(data)-block_size的随机数(起始点)
    # 这里第二个参数必须是tuple of ints，所以才这么写
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    # 准备一个比较草率的目标值，简单定义为原本字符串往右滑动一个字符的结果
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    # token_list=x.tolist()
    # for str_list in token_list:
    #   print(decode(str_list))
    return x, y


# ---傻瓜模型---
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个词嵌入和位置嵌入的表
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, device=device)
        # token_embd = token_embedding_table(x)
        self.position_embedding_table = nn.Embedding(block_size, n_embd, device=device)
        self.network = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # B= batch_size,T=block_size
        token_embd = self.token_embedding_table(idx)
        # 位置信息是直接用[0~n-1]进行embedding的
        position_idx = torch.arange(T, device=device)
        position_embd = self.position_embedding_table(position_idx)
        x = token_embd + position_embd  # (B, T, n_embd)

        logits = self.network(x)  # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # 摊平
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, token_sequ, max_new_tokens):
        """_summary_

        Args:
            token_sequ (_type_): 已知的上文
            max_new_tokens (_type_): 续写的长度
        """
        for _ in range(max_new_tokens):
            tokens_input = token_sequ[:, -block_size:]
            logits, loss = self.forward(tokens_input)
            # 对于每个样本，都只取最后一个字符的[vocab_size]
            logits = logits[:, -1, :]
            # 刚才只是正则化，可能出现负数，这里还需要进行一步归一化才行
            probs = F.softmax(logits, dim=-1)
            # multinomial按照probs中的概率进行采样，采的就是[vocab_size]，也即词表中的索引
            token_next = torch.multinomial(probs, num_samples=1).to(
                device=device
            )  # 概率分布向量  ---->  one-hot向量  ----->  整数token
            token_sequ = torch.cat((token_sequ, token_next), dim=1)
        new_tokens = token_sequ[:, -max_new_tokens:]
        return new_tokens


# print(get_batch("train"))

# --------运行----------
model = LanguageModel()
model = model.to(device)
max_new_tokens = 500
start_idx = random.randint(0, len(test_data) - block_size - max_new_tokens)

# 上文内容
context = torch.zeros((1, block_size), dtype=torch.long, device=device)
context[0, :] = test_data[start_idx : start_idx + block_size]
context_str = decode(context[0].tolist())
wrapped_context_str = textwrap.fill(context_str, width=wrap_width)

# 真实下文
real_next_tokens = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)
real_next_tokens[0, :] = test_data[
    start_idx + block_size : start_idx + block_size + max_new_tokens
]
real_next_tokens_str = decode(real_next_tokens[0].tolist())
wrapped_real_next_tokens_str = textwrap.fill(real_next_tokens_str, width=wrap_width)

# 生成下文
generated_tokens = model.generate(context, max_new_tokens)
generated_str = decode(generated_tokens[0].tolist())
wrapped_generated_str = textwrap.fill(generated_str, width=wrap_width)


print("上文内容：")
print(wrapped_context_str)
print("生成内容：")
print(wrapped_generated_str)
print("真实内容：")
print(wrapped_real_next_tokens_str)


x, y = get_batch("train")
model = model.to(device)
out = model(x)

print(out)
# 词嵌入表，这里词表中有多少词就有多少embedding向量
# print("-" * 16, "词表信息", "-" * 16)
# token_embedding_table = nn.Embedding(vocab_size, n_embd, device=device)
# token_embd = token_embedding_table(x)
# print(token_embd)
# x_list = x.tolist()
# for str_list in x_list:
#     decode_str = decode(str_list)
#     print(decode_str)

# print("-" * 16, "位置信息", "-" * 16)
# position_embedding_table = nn.Embedding(block_size, n_embd, device=device)
# # 创建位置信息张量，即[0 ~ block_size-1]
# position_idx = torch.arange(block_size).to(device)
# position_embd = position_embedding_table(position_idx)
# print(position_embd)
