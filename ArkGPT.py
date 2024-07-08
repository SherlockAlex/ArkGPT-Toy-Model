import torch
from Model import LinearGPT
import tiktoken
import random
import time
import gc
import json
from torch.utils.data import Dataset, DataLoader

'''
方舟生成式预训练语言模型
'''

class CustomDataset(Dataset):
    def __init__(self, data, encode_func, device):
        self.data = data
        self.encode_func = encode_func
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        element = self.data[idx]
        indices = self.encode_func(str(element) + "<EOS>")
        x = torch.tensor(indices[:-1], dtype=torch.long, device=self.device)
        y = torch.tensor(indices[1:], dtype=torch.long, device=self.device)
        return x, y

class ArkGPT():
    def __init__(self,config_file = "./model_config.json"):
        with open(config_file,"r",encoding="utf-8") as f:
            self.config = json.load(f)
        
        device = 'cuda' if self.config["use_gpu"] and torch.cuda.is_available() else 'cpu'
        self.tokener = tiktoken.get_encoding("cl100k_base")
        vocab_count = self.tokener.max_token_value + 1
        self.model = LinearGPT(vocab_count,self.config["d_model"],self.config["units"],self.config["num_block"],device).to(device)
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

    def get_embedding(self):
        return self.model.embedding

    def to_end(self,s:str):
        end_str = '<EOS>'
        end_index = s.find(end_str)
        if end_index == -1:
            return False
        return s

        pass

    def generate(self,text:str,context_length:int = 512,temperature:float = 0.5,print_char:bool = True):
        
        if text == "":
            return

        indices = self.tokener.encode(text)
        decode_token = []

        def predict(index,begin_text:str=None):
            x = torch.tensor([index],dtype=torch.long,device = self.device)
            logits = self.model(x)
            probility = torch.softmax(logits/(temperature+self.eps),dim=-1)
            
            # 从几率最高的50个词语中，随机选择
            topk_prob,topk_idx = torch.topk(probility,k = 50,dim=-1)
            y = topk_idx.gather(dim=-1,index=torch.multinomial(topk_prob,num_samples=1))
        
            token = y[:,-1].cpu().tolist()[0]
            decode_token.append(token)

            word = self.tokener.decode(decode_token)

            if all(char == '�' for char in word):
                return token,""
            decode_token.clear()
            if not print_char:
                return token,word
            if begin_text is not None:
                print(begin_text,end="",flush=True)
            print(word,end="",flush=True)
            gc.collect()
            return token,word
            pass
        
        sentence = text
        token,word = predict(indices,text)

        x = indices
        prompt = False
        if context_length>1:
            for i in range(context_length):
                token,word = predict(x)
                x = [token]
                sentence = sentence + word
                prompt = self.to_end(sentence)
                if prompt != False:
                    break
                pass

        if print_char:
            print('\n')
        return prompt
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

    def pretrain(self,
              trainset:str,
              validset:str,
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
        
        train_text = trainset
        val_text = validset

        train_data = torch.tensor(self.encode(train_text),dtype=torch.long,device=self.device)
        val_data = torch.tensor(self.encode(val_text),dtype=torch.long,device=self.device)

        gc.collect()

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
                pass

            if save_model:
                self.save(model_filename)
        except Exception as e:
            print(f"异常:\n{e}\n正保存模型，请稍后")
            self.save(model_filename)
            pass

        pass

    def set_finetune_model(self,pretrain_model = "./model-ckpt.pt"):
        # 开始
        print("正在加载预训练模型...")
        self.load(filename=pretrain_model)
        layer_counter = 0
        for param in self.model.parameters():
            if layer_counter == 0 or layer_counter%2 == 0:
                param.requires_grad = False
                layer_counter = layer_counter + 1
        return self
        pass

    def finetune(self,
              train_set:list,
              valid_set:list,
              pretrain_model = "./model-ckpt.pt",
              epochs = 1000,
              learning_rate = 0.001,
              eval_iters = 20,
              save_model = False,
              model_filename = "./model-ckpt-finetune.pt"
              ):
        
        print("正在加载数据集...")
        trainset = []
        validset = []

        for element in train_set:
            indices = self.encode(str(element)+"<EOS>")
            x = torch.tensor(indices[:-1],dtype=torch.long)
            y = torch.tensor(indices[1:],dtype=torch.long)
            trainset.append((x,y))

        for element in valid_set:
            indices = self.encode(str(element)+"<EOS>")
            x = torch.tensor(indices[:-1],dtype=torch.long)
            y = torch.tensor(indices[1:],dtype=torch.long)
            validset.append((x,y))

        random.shuffle(trainset)
        random.shuffle(validset)

        # 开始

        def get_batch(split: str):
            data = trainset if split == 'train' else validset
            i = random.randint(0,len(data)-1)
            dataset = data[i]
            x = torch.stack([dataset[0]]).to(self.device)
            y = torch.stack([dataset[1]]).to(self.device)
            return x, y
        
        @torch.no_grad
        def estimate_loss():
            out = {}
            self.model.eval()
    
            for split in ['train', 'valid']:
                losses = torch.zeros(eval_iters)
                accuracies = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    x_batch, y_batch = get_batch(split)
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

                if step % eval_iters == 0 or step == epochs - 1:
                    out = estimate_loss()
                    tracked_losses.append(out)
                    train_loss,train_acc = out['train']
                    val_loss,val_acc = out['valid']
                    print(f"学习次数:{step}/{epochs},训练损失:{round(train_loss.item(), 3)},训练正确率:{round(train_acc.item(),3)},测试损失:{round(val_loss.item(), 3)},测试正确率:{round(val_acc.item(),3)}")
                    pass
                
                xb, yb = get_batch('train')
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