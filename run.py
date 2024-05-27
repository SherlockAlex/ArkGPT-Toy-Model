import torch
import torch.nn as nn
import torch.functional as F
from Model import Model
from tokenizer import Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if it's available.

tokener = Tokenizer(mode="char",encoding="utf-16").load("./chinese.json")
vocab_count = tokener.get_vocab_count()

model = Model(vocab_count,512)

# 模型训练

with open("./语料/小说/诛仙/001.txt",'r',encoding="utf-16") as file:
    text = file.read()
    text = text.replace("\n","")
    text = text.replace("　","")

tokens_data = tokener.encode(text)
tokens_data = torch.tensor(tokens_data,dtype=torch.int32,device=device)

split_idx = int(len(tokens_data) * 0.9)
train_data = tokens_data[:split_idx]
val_data = tokens_data[split_idx:]

def get_batch(split: str,context_length = 128,batch_size = 4):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y

@torch.no_grad
def estimate_loss(model:Model,eval_iters = 20):
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits,loss = model(x_batch,y_batch)
            model.forget()
            losses[k] = loss.item()
        out[split] = losses.mean()
        pass
    model.train()
    return out

def train(model:Model,leaning_rate = 0.001,epochs = 100,eval_iters = 20):
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=leaning_rate)
    tracked_losses = []
    for step in range(epochs):
        if step % eval_iters == 0 or step == epochs - 1:
            losses = estimate_loss(model=model,eval_iters=eval_iters)
            tracked_losses.append(losses)
            print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
                round(losses['valid'].item(), 3))
            pass
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        model.forget()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pass
    torch.save(model.state_dict(), 'model-ckpt.pt')
    pass

def load():
    model.load_state_dict(torch.load("./model-ckpt.pt"))
    model.eval()

train(model=model)

# 模型续写
text = "杨过，你在吗？"
indices = tokener.encode(text)

tokens = [] + indices

x = torch.tensor(indices,dtype=torch.int32)
y = model(x)

y = torch.softmax(y,dim=-1)
y = torch.argmax(y,dim=1)
token = list(y.numpy())[-1]
tokens.append(token)

for i in range(512):
    index = torch.tensor([token],dtype=torch.int32)
    y = model(x)

    y = torch.softmax(y,dim=-1)
    y = torch.argmax(y,dim=1)
    token = list(y.numpy())[-1]
    tokens.append(token)
    pass

text = tokener.decode(tokens)
print(text)
