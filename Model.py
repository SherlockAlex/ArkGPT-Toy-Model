import torch
import torch.nn as nn
from torch.nn import functional as F
import layers

class Forward(nn.Module):
    def __init__(self,in_features,out_features,device,*args, **kwargs) -> None:
        super(Forward,self).__init__(*args, **kwargs)
        
        self.normalize = layers.RMSNormalize().to(device)
        
        self.dense1 = nn.Linear(in_features,out_features).to(device)
        self.dropout1 = nn.Dropout(0.1).to(device)
        self.dense2 = nn.Linear(out_features,out_features).to(device)
        self.dropout2 = nn.Dropout(0.1).to(device)

    def forward(self,x):
        x = torch.sigmoid(self.dense1(x))
        x = self.dropout1(x)
        x = torch.sigmoid(self.dense2(x))
        x = self.dropout2(x)

        return x

        pass
        
    pass

class Block(nn.Module):
    def __init__(self,units, device,*args, **kwargs) -> None:
        super(Block,self).__init__(*args, **kwargs)
        self.memory = layers.LinearAttention(units,units,units,device).to(device)
        self.moe = layers.MoELayer(num_experts=8,in_features=units,out_features=units,device = device).to(device)
        self.feedforward = Forward(units,units,device).to(device)
        self.normalize = layers.RMSNormalize().to(device)

    def forward(self,x):
        input = x
        x = self.memory(self.normalize(x)) + x
        x = self.moe(self.normalize(x))
        x = self.feedforward(self.normalize(x)) + x
        return x + input
        
    def forget(self):
        self.memory.forget()
        pass

    pass

class Model(nn.Module):
    def __init__(self,vocab_count,d_model,units,device,*args, **kwargs) -> None:
        super(Model,self).__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_count,d_model).to(device)
        self.linear = nn.Linear(d_model,units,bias=False).to(device)
        self.memory_blocks = [Block(units,device).to(device) for i in range(16)]
        self.block = nn.Sequential(*(self.memory_blocks+[layers.RMSNormalize().to(device)]))
        self.normalize = layers.RMSNormalize()
        self.logits = nn.Linear(units,vocab_count,bias=False).to(device)
        self.temperature = 0.5
        pass

    def forward(self,x,labels = None):
        x = self.embedding(x)
        x = self.linear(x)
        x = self.block(x) + x
        x = self.normalize(x)
        logits = self.logits(x)/self.temperature
        if labels is not None:
            B,T,C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            labels_reshaped = labels.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=labels_reshaped)
            self.forget()
            return logits,loss
            pass
        return logits
    
    def forget(self):
        for x in self.memory_blocks:
            x.forget()

    def set_temperature(self,temperature):
        self.temperature = temperature

    pass

import tiktoken
class Ark():
    def __init__(self,d_model,units = 256,device = 'cpu'):
        self.tokener = tiktoken.get_encoding("cl100k_base")
        vocab_count = self.tokener.max_token_value + 1
        self.model = Model(vocab_count,d_model,units,device).to(device)
        self.device = device
        pass

    def encode(self,text:str):
        return self.tokener.encode(text)
        pass

    def decode(self,tokens:list):
        return self.tokener.decode(tokens)
        pass

    def __call__(self,text:str,context_length = 512,temperature = 0.5):
        if text == "":
            return

        self.model.set_temperature(temperature=temperature)

        indices = self.tokener.encode(text)

        tokens = []

        x = torch.tensor([indices],dtype=torch.long,device = self.device)
        y = self.model(x)

        y = torch.softmax(y,dim=-1)
        y = torch.multinomial(input=y[:,-1,:], num_samples=1)
        token = list(y[:,-1].cpu().numpy())[0]
        tokens.append(token)

        word = self.tokener.decode([token])
        print(text+word,end="",flush=True)


        for i in range(context_length):
            x = torch.tensor([[token]],dtype=torch.long,device=self.device)
            y = self.model(x)

            y = torch.softmax(y,dim=-1)
            y = torch.multinomial(input=y[:,-1,:], num_samples=1)
            token = list(y[:,-1].cpu().numpy())[0]
            tokens.append(token)

            word = self.tokener.decode([token])
            print(word,end="",flush=True)
            pass
        print('\n')
        sentence = self.tokener.decode(tokens)
        return sentence
        pass

    def forget(self):
        self.model.forget()
        pass

    def load(self,filename = "./model-ckpt.pt"):
        self.model.load_state_dict(torch.load(filename))
        print("模型加载成功")
        self.model.eval()

    def save(self,filename = "./model-ckpt.pt"):
        torch.save(self.model.state_dict(),filename)

    def train(self,dataset:str,context_length = 512,epochs = 1000,batch_size = 8,learning_rate = 0.001,eval_iters = 20):
        tokens_data = self.encode(dataset)
        tokens_data = torch.tensor(tokens_data,dtype=torch.long,device=self.device)

        data_count = len(tokens_data)

        split_idx = int(data_count * 0.9)
        train_data = tokens_data[:split_idx]
        val_data = tokens_data[split_idx:]

        def get_batch(split: str):
            data = train_data if split == 'train' else val_data
            idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
            x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(self.device)
            y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(self.device)
            return x, y
        
        @torch.no_grad
        def estimate_loss():
            out = {}
            self.model.eval()
    
            for split in ['train', 'valid']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    x_batch, y_batch = get_batch(split)
                    logits,loss = self.model(x_batch,y_batch)
                    losses[k] = loss.item()

                out[split] = losses.mean()
                pass
            self.model.train()
            return out
        
        
        print(f"模型正在使用{self.device}训练")
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=learning_rate)
        tracked_losses = []
        for step in range(epochs):
            if step % eval_iters == 0 or step == epochs - 1:
                losses = estimate_loss()
                tracked_losses.append(losses)
                print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
                    round(losses['valid'].item(), 3))
                pass
        
            xb, yb = get_batch('train')
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pass

        pass

    pass