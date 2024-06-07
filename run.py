import torch
import torch.nn as nn
import torch.functional as F
from Ark import Ark
import text

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if it's available.

dataset = text.get_text()
ark = Ark(d_model=1024,units=256,device=device)

# 打印模型参数规模
print(ark.head())
ark.load()
ark.train(
    dataset=dataset,
    epochs=20,
    learning_rate=0.0005,
    eval_iters=5,
    context_length=512,
    batch_size=4,
    save_model=True
)

text = "杨过和小龙女来到了草庙村，遇到了张小凡和陆雪琪等人，"
sentence = ark(text,context_length=1024,temperature=0.5)

while True:
    question = input(">>>")
    if question == "清空上下文":
        ark.forget()
        print("已经清空上下文")
        continue
    sentence = ark(question,context_length=1024,temperature=0.5)


