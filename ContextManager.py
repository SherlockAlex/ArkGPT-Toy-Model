import torch
import torch.nn as nn
from Model import EmbeddingModel
import json
import tiktoken
import random


class ContextManager():
    def __init__(self,config_file = "./context_manager.json") -> None:
        with open(config_file,"r",encoding="utf-8") as f:
            self.config = json.load(f)
        self.device = 'cuda' if self.config["use_gpu"] and torch.cuda.is_available() else 'cpu'
        self.tokener = tiktoken.get_encoding("cl100k_base")
        vocab_count = self.tokener.max_token_value + 1
        self.model = EmbeddingModel(
            vocab_count=vocab_count,
            d_model=self.config["d_model"],
            units=self.config["units"],
            num_block=self.config["num_block"],
            device=self.device
            )
        
        self.mask = {
            "PAD":44918
        }

        pass

    def encode(self,text:str):
        return self.tokener.encode(text)
        pass

    def decode(self,tokens:list):
        return self.tokener.decode(tokens)
        pass

    def __call__(self,text:str):
        return self.vectorization(text=text)
        pass

    def vectorization(self,text:str):
        if text == "":
            return
        indices = self.tokener.encode(text)
        x = torch.tensor([indices],dtype=torch.long,device = self.device)
        key = self.model(x)
        return key
        pass

    def create_database(self,text:str,context = 128):
        return [text[i:i+context] for i in range(0,len(text),context)]
        pass

    def search(self,text:str,contexts:list,nums = 4):
        if len(contexts)<=0:
            return
        query = self(text)
        sims = []
        for context in contexts:
            keys = self(context)
            sims.append(nn.functional.cosine_similarity(query,keys).cpu().tolist()[-1])
        sims = torch.tensor(sims,dtype=torch.float32)
        top_sims,top_idx = torch.topk(sims,nums)
        result = []
        print(top_idx)
        for i in top_idx.cpu().tolist():
            result.append(contexts[i])
        return result
        pass

    def load(self,filename = "./context-model-ckpt.pt"):
        try:
            self.model.load_state_dict(torch.load(filename))
            print("模型加载成功")
            self.model.eval()
        except:
            print("加载模型失败，请确保模型文件路径正确")
        return self

    def save(self,filename = "./context-model-ckpt.pt"):
        torch.save(self.model.state_dict(),filename)
        pass

    def train(self,
              trainset:str,
              validset:str,
              context_length = 512,
              epochs = 1000,
              batch_size = 8,
              learning_rate = 0.001,
              eval_iters = 20,
              save_model = False,
              random_context = False,
              model_filename = "./context-model-ckpt.pt"
              ):

        min_context_length = 128
        max_context_length = (min_context_length + 1) if context_length<=min_context_length else context_length
        
        train_text = trainset
        val_text = validset

        train_data = torch.tensor(self.encode(train_text),dtype=torch.long,device=self.device)
        val_data = torch.tensor(self.encode(val_text),dtype=torch.long,device=self.device)

        # 定义一个函数来随机遮罩序列中的token
        def mask_tokens(sequence, mask_id, mask_ratio=0.25):
            """
            随机遮罩序列中的token。
    
            参数:
            - sequence: 一个一维张量，表示token序列
            - mask_id: 用于遮罩的特定数字
            - mask_ratio: 被遮罩的token所占的比例
    
            返回:
            - 遮罩后的序列
            """
            # 确定要遮罩的token数量
            num_masked_tokens = int(mask_ratio * sequence.size(0))
    
            # 随机选择遮罩的位置
            mask_positions = torch.randperm(sequence.size(0))[:num_masked_tokens]
    
            # 遮罩这些位置的token
            masked_sequence = sequence.clone()  # 首先复制序列以避免修改原始数据
            masked_sequence[mask_positions] = mask_id
    
            return masked_sequence

        def get_batch(split: str,ctx_len:int):
            data = train_data if split == 'train' else val_data
            idxs = torch.randint(low=0, high=len(data) - ctx_len, size=(batch_size,))
            x = torch.stack([mask_tokens(data[idx:idx + ctx_len],mask_id=self.mask["PAD"]) for idx in idxs]).to(self.device)
            y = torch.stack([data[idx:idx + ctx_len] for idx in idxs]).to(self.device)
            #y = torch.stack([data[idx + 1:idx + ctx_len + 1] for idx in idxs]).to(self.device)
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

    def head(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    pass


