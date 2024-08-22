from ArkGPT import ArkGPT

model = ArkGPT().load("./model-ckpt-finetune.pt")

start = '\",\"target\":\"'

while True:
    text = '{\"input\":\"' + input(">>>") + start
    #text = input(">>>")
    if text == "清空上下文":
        model.forget()
        continue
    prompt = model(text,1024,0.5,True)
    pass