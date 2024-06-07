import torch
from Model import Model
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
        return self.generate(text=text,context_length=context_length,temperature=temperature)
        pass

    def generate(self,text:str,context_length = 512,temperature = 0.5):
        if text == "":
            return

        indices = self.tokener.encode(text)

        tokens = []

        x = torch.tensor([indices],dtype=torch.long,device = self.device)
        y = self.model(x)

        y = torch.softmax(y/temperature,dim=-1)
        y = torch.multinomial(input=y[:,-1,:], num_samples=1)
        token = y[:,-1].cpu().tolist()[0]
        tokens.append(token)

        word = self.tokener.decode([token])
        print(text+word,end="",flush=True)

        decode_token = []

        for i in range(context_length):
            x = torch.tensor([[token]],dtype=torch.long,device=self.device)
            y = self.model(x)

            y = torch.softmax(y,dim=-1)
            y = torch.multinomial(input=y[:,-1,:], num_samples=1)
            token = y[:,-1].cpu().tolist()[0]
            tokens.append(token)

            decode_token.append(token)
            word = self.tokener.decode(decode_token)
            if not all(char == '�' for char in word):
                decode_token.clear()
                print(word,end="",flush=True)
                pass

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

    def train(self,dataset:str,context_length = 512,epochs = 1000,batch_size = 8,learning_rate = 0.001,eval_iters = 20,save_model = False,model_filename = "./model-ckpt.pt"):
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

        if save_model:
            self.save(model_filename)

        pass

    def head(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    pass