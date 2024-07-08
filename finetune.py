from ArkGPT import ArkGPT
from Model import MoELayer
import torch.nn as nn
import traindata

model = ArkGPT().load()

model.finetune(
    train_set=traindata.prompt_dataset("./语料/trainset.json"),
    valid_set=traindata.prompt_dataset("./语料/validset.json"),
    epochs=1000,
    learning_rate=0.001,
    eval_iters=5,
    save_model=True
)

print("模型微调成功")
while True:
    pass