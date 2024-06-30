from ArkGPT import ArkGPT
import traindata

model = ArkGPT(
    d_model=512,    # 词向量维度
    units=512,      # 神经网络层神经元个数
    num_block=12,   # 堆叠的块数
    use_gpu=True
).load()
 
model.train(
    trainset=traindata.train_data(),
    validset=traindata.valid_data(),
    epochs=100,
    learning_rate=0.001,
    eval_iters=5,
    context_length=128,
    batch_size=16,
    save_model=True
)

text = "　　杨过和小龙女来到了草庙村，遇到了张小凡和陆雪琪等人，"
temperature = 0.5
sentence = model(text,context_length=1024,temperature=temperature)

while True:
    pass




