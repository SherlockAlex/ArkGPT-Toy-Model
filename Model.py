import torch
import torch.nn as nn
from torch.nn import functional as F

class LinearAttentionCell(nn.Module):
    def __init__(self,in_features,out_features,device) -> None:
        super(LinearAttentionCell,self).__init__()
        
        self.device = device

        self.in_features = in_features
        self.out_features = out_features
        
        self.rotatry = torch.tensor([1 if i % 2 == 0 else -1 for i in range(out_features)],device=device)
        self.swap_indices = torch.tensor([i+1 if i % 2 == 0 else i-1 for i in range(out_features)], dtype=torch.long,device=device)

        self.wq = nn.Linear(in_features=in_features,out_features=out_features,bias=False).to(device)
        self.wk = nn.Linear(in_features=in_features,out_features=out_features,bias=False).to(device)
        self.wv = nn.Linear(in_features=in_features,out_features=out_features,bias=False).to(device)

        self.memory = 0
        self.zeta = 0
        self.pos = 0

        self.dk = torch.sqrt(torch.tensor(1.0/out_features,dtype=torch.float32,device=device))

        indices = (torch.arange(self.out_features//2,dtype=torch.float32,device=self.device) + 1)/self.out_features
        self.thetas = 10000**(-2*indices)
        self.use_rope = False


    def get_cosine(self,pos):
        cosines = torch.cos(pos*self.thetas)
        cosines = cosines.repeat_interleave(2,dim=-1)
        return cosines
    
    def get_sine(self,pos):
        sine = torch.sin(pos*self.thetas)
        sine = sine.repeat_interleave(2,dim=-1)
        return sine

    def forward(self,x):
        
        # 自注意力机制
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)           

        if self.use_rope:
            cosines = self.get_cosine(self.pos)
            sines = self.get_sine(self.pos)

            B,T,C = q.shape
            q_x = q
            q_y = ((q*self.rotatry).squeeze().index_select(-1,self.swap_indices).unsqueeze(1))
            q_y = q_y.view(B,T,C)
            

            k_x = k
            k_y = ((k*self.rotatry).squeeze().index_select(-1,self.swap_indices).unsqueeze(1))
            k_y = k_y.view(B,T,C)
            
            q = q_x*cosines + q_y*sines
            k = k_x*cosines + k_y*sines

        q = torch.nn.functional.elu(q*self.dk) + 1
        k = torch.nn.functional.elu(k*self.dk) + 1

        delta_zeta = k.transpose(-2,-1)
        delta_memory = torch.matmul(delta_zeta,v)

        if self.pos == 0:
            with torch.no_grad():
                self.memory = 0*delta_memory
                self.zeta = 0*delta_zeta
        
        # 检索模型中记忆相关的部分（加上1e-5防止出现Nan的情况）
        scale = torch.matmul(q,delta_zeta) + torch.matmul(q,self.zeta) + 1e-5
        memory = torch.matmul(q,delta_memory) + torch.matmul(q,self.memory)

        y = (memory/scale)
        with torch.no_grad():

            self.memory = self.memory + delta_memory
            self.zeta = self.zeta + delta_zeta
            self.pos = self.pos + 1

        return y
    
    def is_use_rope(self,value):
        self.use_rope = value
    
    def forget(self):
        with torch.no_grad():
            self.memory = 0
            self.zeta = 0
            self.pos = 0
        pass
    
    pass


class LinearAttention(nn.Module):
    def __init__(self,d_model,in_features,out_features,device,*args, **kwargs) -> None:
        super(LinearAttention,self).__init__(*args, **kwargs)
        self.cell = LinearAttentionCell(in_features=in_features,out_features=d_model,device = device).to(device)
        self.projection = nn.Linear(in_features=d_model,out_features=out_features,bias=False,device=device)

    def forward(self,x):
        B,T,S = x.shape
        x = x.view(B,T,1,S)
        y = torch.stack([self.cell(x[:,i,:,:]) for i in range(T)],dim=0)
        y = torch.transpose(y,0,1)  # 第一指标与第二指标进行交换

        B,T,X,S = y.shape
        y = y.view(B,T,S)
        y = self.projection(y)

        return y

    def forget(self):
        self.cell.forget()

    pass

class FeedForward(nn.Module):
    def __init__(self,in_features,out_features,device,*args, **kwargs) -> None:
        super(FeedForward,self).__init__(*args, **kwargs)
        self.dense = nn.Linear(in_features,4*out_features).to(device)
        self.linear = nn.Linear(4*out_features,out_features).to(device)
        self.dropout = nn.Dropout(0.3).to(device)
        self.gelu = nn.GELU()
        pass

    def forward(self,x):
        x = self.gelu(self.dense(x))
        x = self.dropout(self.linear(x))
        return x

    pass

class MoELayer(nn.Module):
    def __init__(self,num_experts,in_features,out_features,device,*args, **kwargs) -> None:
        super(MoELayer,self).__init__(*args, **kwargs)
        self.head_size = in_features // num_experts
        self.experts = nn.ModuleList([FeedForward(in_features=self.head_size,out_features=out_features,device = device) for i in range(num_experts)])
        self.router = nn.Linear(in_features=in_features,out_features=num_experts).to(device)

    def forward(self,x):
        logits = self.router(x)
        router = torch.softmax(logits,dim=-1)

        inputs = torch.split(x,self.head_size,dim=-1)
        experts = torch.stack([expert(input) for input,expert in zip(inputs,self.experts)],dim=0)
        y = torch.einsum('btx,xbts->bts',router,experts)
        return y

        pass
    pass

class Block(nn.Module):
    def __init__(self,units, device,*args, **kwargs) -> None:
        super(Block,self).__init__(*args, **kwargs)
        self.attention = LinearAttention(64,units,units,device).to(device)
        self.feed_forward = FeedForward(in_features=units,out_features=units,device=device)
        self.norm_1 = nn.LayerNorm(normalized_shape=units).to(device)
        self.norm_2 = nn.LayerNorm(normalized_shape=units).to(device)
        self.norm_3 = nn.LayerNorm(normalized_shape=units).to(device)
        pass

    def forward(self,x):
        x = self.norm_1(x)
        x = self.norm_2(self.attention(x)) + x
        x = self.norm_3(self.feed_forward(x)) + x
        return x
        
    def forget(self):
        self.attention.forget()
        pass

    pass

class Decoder(nn.Module):
    def __init__(self,vocab_count,d_model,units,num_block,device,*args, **kwargs) -> None:
        super(Decoder,self).__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_count,d_model).to(device)
        self.linear = nn.Linear(d_model,units,bias=False).to(device)
        self.decoder_blocks = [Block(units,device).to(device) for i in range(num_block)]
        self.decoder_blocks[0].attention.cell.is_use_rope(True)
        self.decoder = nn.Sequential(*(self.decoder_blocks))
        self.norm_1 = nn.LayerNorm(normalized_shape=units)
        self.norm_2 = nn.LayerNorm(normalized_shape=units)
        self.moe = MoELayer(num_experts=8,in_features=units,out_features=units,device = device).to(device)
        self.logits = nn.Linear(units,vocab_count,bias=False).to(device)

        pass

    def forward(self,x,labels = None):
        x = self.embedding(x)
        x = self.linear(x)
        x = self.decoder(x)+x
        x = self.norm_1(x)
        x = self.moe(x) + x
        logits = self.logits(self.norm_2(x))
        if labels is not None:
            # 训练模式
            B,T,C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            labels_reshaped = labels.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=labels_reshaped)
            
            # 计算模型正确率
            predictions = torch.argmax(logits,dim=-1).view(B * T)
            correct = (predictions == labels_reshaped).sum().item()
            total = labels_reshaped.numel()
            accuracy = correct / total

            self.forget()
            return loss,accuracy
            pass
        return logits[:,-1,:]
    
    def forget(self):
        for x in self.decoder_blocks:
            x.forget()

    pass

