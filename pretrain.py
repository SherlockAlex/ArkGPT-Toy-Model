from ArkGPT import ArkGPT
import traindata

model = ArkGPT().load()

'''
model.pretrain(
    trainset=traindata.pretrain_train_data(),
    validset=traindata.pretrain_valid_data(),
    epochs=1000,
    learning_rate=0.001,
    eval_iters=5,
    context_length=128,
    batch_size=8,
    save_model=True
)
'''

model.finetune(
    train_set=traindata.prompt_dataset("./语料/wiki_train.json"),
    valid_set=traindata.prompt_dataset("./语料/wiki_valid.json"),
    epochs=1000,
    learning_rate=0.001,
    eval_iters=5,
    save_model=True,
    model_filename='./model-ckpt.pt'
)


print("训练完成")
while True:
    pass




