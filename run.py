import torch
import torch.nn as nn
import torch.functional as F
from Model import Ark
import text

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if it's available.

dataset = text.get_text()
ark = Ark(d_model=1024,units=256,device=device)

ark.load()
ark.train(dataset=dataset,epochs=100,learning_rate=0.0005,eval_iters=10)
ark.save()

text = "杨过和小龙女来到了草庙村，遇到了张小凡和陆雪琪等人"
sentence = ark(text)
print(sentence)

while True:
    question = input(">>>")
    if question == "清空上下文":
        ark.forget()
    sentence = ark(question)
    print(sentence)


