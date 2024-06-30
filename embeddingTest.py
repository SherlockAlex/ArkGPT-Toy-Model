from ArkGPT import ArkGPT
import torch
import torch.nn.functional as F
import torch.nn as nn

model = ArkGPT().load()
model("你好",1)
embedding = model.get_embedding()
# 57668 37046

you = embedding(torch.tensor([57668],dtype=torch.long))
me = embedding(torch.tensor([37046],dtype=torch.long))
print(F.cosine_similarity(you,me))
print(you)
print(me)
