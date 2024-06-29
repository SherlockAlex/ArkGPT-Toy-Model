from ArkGPT import ArkGPT
import torch
import torch.nn.functional as F

model = ArkGPT().load()

while True:
    text = input(">>>")
    if text == "清空上下文":
        continue
    model(text,1024)
    pass