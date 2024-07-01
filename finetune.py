from ArkGPT import ArkGPT
from Model import MoELayer
import torch.nn as nn
import traindata

model = ArkGPT(
    d_model=512,    # 词向量维度
    units=512,      # 神经网络层神经元个数
    num_block=12,   # 堆叠的块数
    use_gpu=True
)

model.finetune(
    train_set=traindata.dataset("./语料/dataset.json"),
    valid_set=traindata.dataset("./语料/test.json"),
    epochs=100,
    learning_rate=0.001,
    eval_iters=5,
    save_model=True
)

print("模型微调成功")
while True:
    pass