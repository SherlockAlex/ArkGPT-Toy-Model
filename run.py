from ArkGPT import ArkGPT

model = ArkGPT(use_gpu=True).load()

start = '\",\"target\":\"'

while True:
    text = '{\"input\":\"' + input(">>>") + start
    if text == "清空上下文":
        model.forget()
        continue
    prompt = model(text,512,0.1,True)
    pass