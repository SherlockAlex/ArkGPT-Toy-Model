import torch
from Ark import Ark
import traindata
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if it's available.

def main(argv):
    
    ark = Ark(
        d_model=512,    # 词向量维度
        units=512,      # 神经网络层神经元个数
        num_block=12,   # 堆叠的块数
        device=device   # CPU or GPU
    )
    
    if 'load' in argv:
        try:
            ark.load()
        except:
            pass

    if 'train' in argv:
        dataset = traindata.get_text()
        ark.train(
            dataset=dataset,
            epochs=100,
            learning_rate=0.001,
            eval_iters=5,
            context_length=128,
            batch_size=16,
            save_model=True
        )

    text = "　　杨过和小龙女来到了草庙村，遇到了张小凡和陆雪琪等人，"
    temperature = 0.5
    sentence = ark(text,context_length=1024,temperature=temperature)
    print('\n原文:'+sentence)

    while True:
        question = input(">>>")
        if question == "清空上下文":
            ark.forget()
            print("已经清空上下文")
            continue
        sentence = ark(question,context_length=1024,temperature=temperature)
        print('\n原文:'+sentence)
    
    pass

if __name__ == "__main__":
    main(sys.argv[1:])




