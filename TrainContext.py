from ContextManager import ContextManager
import traindata

model = ContextManager().load()
 
model.train(
    trainset=traindata.pretrain_train_data(),
    validset=traindata.pretrain_valid_data(),
    epochs=10000,
    learning_rate=0.0001,
    eval_iters=5,
    context_length=128,
    batch_size=2,
    save_model=True
)

print("训练完成")
while True:
    pass
