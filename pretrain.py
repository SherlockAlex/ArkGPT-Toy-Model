from ArkGPT import ArkGPT
import traindata

model = ArkGPT(
    d_model=512,    # 词向量维度
    units=512,      # 神经网络层神经元个数
    num_block=12,   # 堆叠的块数
    use_gpu=True
).load()
 
model.pretrain(
    #trainset=traindata.news_data(1949,1971),
    #validset=traindata.news_data(1971,1979),
    trainset=traindata.dialog_train(),
    validset=traindata.dialog_test(),
    #trainset=traindata.train_data(),
    #validset=traindata.valid_data(),
    epochs=100,
    learning_rate=0.0001,
    eval_iters=5,
    context_length=512,
    batch_size=4,
    save_model=True
)

print("训练完成")
while True:
    pass




