import torch
from Model import LinearGPT
import tiktoken
import random
import time

'''
方舟生成式预训练语言模型
'''

class ArkGPT():
    def __init__(self,d_model = 512,units = 512,num_block = 12,device = 'cpu'):
        self.tokener = tiktoken.get_encoding("cl100k_base")
        vocab_count = self.tokener.max_token_value + 1
        self.model = LinearGPT(vocab_count,d_model,units,num_block,device).to(device)
        self.device = device
        self.eps = 1e-8
        pass

    def encode(self,text:str):
        return self.tokener.encode(text)
        pass

    def decode(self,tokens:list):
        return self.tokener.decode(tokens)
        pass

    def __call__(self,text:str,context_length = 512,temperature = 0.5,print_char = True):
        return self.generate(text=text,context_length=context_length,temperature=temperature,print_char=print_char)
        pass

    def generate(self,text:str,context_length:int = 512,temperature:float = 0.5,print_char:bool = True):
        if text == "":
            return

        indices = self.tokener.encode(text)

        tokens = []
        decode_token = []

        def predict(index,begin_text:str=None):
            x = torch.tensor([index],dtype=torch.long,device = self.device)
            logits = self.model(x)
            # print(self.model.decoder.decoder_blocks[0].attention.cell.pos)
            probility = torch.softmax(logits/(temperature+self.eps),dim=-1)
            
            # 从几率最高的50个词语中，随机选择
            topk_prob,topk_idx = torch.topk(probility,k = 50,dim=-1)
            y = topk_idx.gather(dim=-1,index=torch.multinomial(topk_prob,num_samples=1))
        
            token = y[:,-1].cpu().tolist()[0]
            tokens.append(token)
        
            decode_token.append(token)
            word = self.tokener.decode(decode_token)
            if all(char == '�' for char in word):
                return token
            decode_token.clear()
            if not print_char:
                return token
            if begin_text is not None:
                print(begin_text+word,end="",flush=True)
            print(word,end="",flush=True)
            return token
            pass
        
        token = predict(indices,begin_text=text)

        for i in range(context_length):
            token = predict([token])
            pass

        if print_char:
            print('\n')
        sentence = self.tokener.decode(tokens)
        return sentence
        pass

    def forget(self):
        self.model.forget()
        pass

    def load(self,filename = "./model-ckpt.pt"):
        try:
            self.model.load_state_dict(torch.load(filename))
            print("模型加载成功")
            self.model.eval()
        except:
            print("加载模型失败，请确保模型文件路径正确")
        return self

    def save(self,filename = "./model-ckpt.pt"):
        torch.save(self.model.state_dict(),filename)
        pass

    def train(self,
              dataset:str,
              context_length = 512,
              epochs = 1000,
              batch_size = 8,
              learning_rate = 0.001,
              eval_iters = 20,
              save_model = False,
              random_context = False,
              model_filename = "./model-ckpt.pt"
              ):
        
        min_context_length = 128
        max_context_length = (min_context_length + 1) if context_length<=min_context_length else context_length

        split_idx = int(len(dataset) * 0.9)
        train_text = dataset[:split_idx]
        val_text = dataset[split_idx:]

        train_data = torch.tensor(self.encode(train_text),dtype=torch.long,device=self.device)
        val_data = torch.tensor(self.encode(val_text),dtype=torch.long,device=self.device)

        def get_batch(split: str,ctx_len:int):
            data = train_data if split == 'train' else val_data
            idxs = torch.randint(low=0, high=len(data) - ctx_len, size=(batch_size,))
            x = torch.stack([data[idx:idx + ctx_len] for idx in idxs]).to(self.device)
            y = torch.stack([data[idx + 1:idx + ctx_len + 1] for idx in idxs]).to(self.device)
            return x, y
        
        @torch.no_grad
        def estimate_loss(ctx_len:int):
            out = {}
            self.model.eval()
    
            for split in ['train', 'valid']:
                losses = torch.zeros(eval_iters)
                accuracies = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    x_batch, y_batch = get_batch(split,ctx_len)
                    loss,accuracy = self.model(x_batch,y_batch)
                    losses[k] = loss.item()
                    accuracies[k] = accuracy

                out[split] = losses.mean(),accuracies.mean()
                pass
            self.model.train()
            return out
        
        try:
            optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=learning_rate)
            tracked_losses = []
            print(f"启动{self.device}训练,训练参数量:{self.head()}")
            for step in range(epochs):
                lenght = random.randint(min_context_length,max_context_length) if random_context else context_length

                if step % eval_iters == 0 or step == epochs - 1:
                    out = estimate_loss(lenght)
                    tracked_losses.append(out)
                    train_loss,train_acc = out['train']
                    val_loss,val_acc = out['valid']
                    print(f"学习次数:{step}/{epochs},训练损失:{round(train_loss.item(), 3)},训练正确率:{round(train_acc.item(),3)},测试损失:{round(val_loss.item(), 3)},测试正确率:{round(val_acc.item(),3)},上下文长度:{lenght}")
                    pass
                
                xb, yb = get_batch('train',lenght)
                loss,accuracy = self.model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                time.sleep(1.0/30)
                pass

            if save_model:
                self.save(model_filename)
        except Exception as e:
            print(f"异常:\n{e}\n正保存模型，请稍后")
            self.save(model_filename)
            pass

        pass

    def head(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    pass