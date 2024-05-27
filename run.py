import torch
import torch.nn as nn
import torch.functional as F
from Model import Model
from tokenizer import Tokenizer

tokener = Tokenizer(mode="char",encoding="utf-16").load("./chinese.json")
vocab_count = tokener.get_vocab_count()

model = Model(vocab_count,512)

# 模型训练

# 模型续写
text = "杨过，你在吗？"
indices = tokener.encode(text)

x = torch.tensor(indices,dtype=torch.int32)
y = model(x)

y = torch.softmax(y,dim=-1)
y = torch.argmax(y,dim=1)

text = tokener.decode(list(y.numpy()))

print(text)
