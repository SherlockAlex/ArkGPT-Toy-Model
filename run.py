from ArkGPT import ArkGPT

model = ArkGPT(use_gpu=True).load()

while True:
    text = input(">>>")
    if text == "清空上下文":
        model.forget()
        continue
    model(text,1024)
    pass