import torch
import torch.nn as nn
from torch.nn import functional as F

eps = 1e-8
class LinearAttentionCell(nn.Module):
    def __init__(self,in_features,out_features,device) -> None:
        super(LinearAttentionCell,self).__init__()
        # 定义模型及其运行设备:cpu|gpu
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        # 计算query、key、value
        self.wq = nn.Linear(in_features=in_features,out_features=out_features,bias=False).to(device)
        self.wk = nn.Linear(in_features=in_features,out_features=out_features,bias=False).to(device)
        self.wv = nn.Linear(in_features=in_features,out_features=out_features,bias=False).to(device)
        # 计算线性注意力分数余项
        self.wo = nn.Linear(in_features=out_features,out_features=out_features).to(device)
        # 记忆矩阵与缩放矩阵以及当前状态
        self.memory = 0
        self.zeta = 0
        # 细胞当前处理token的位置
        self.pos = 0
        # 计算缩放因子
        self.dk = torch.sqrt(torch.tensor(1.0/self.out_features,dtype=torch.float32,device=self.device))
        # 使用位置编码
        self.use_rope = False
        pass


    def get_cosine(self,theta:torch.Tensor):
        cosines = torch.cos(theta)
        cosines = cosines.repeat_interleave(2,dim=-1)
        return cosines
    
    def get_sine(self,theta:torch.Tensor):
        sine = torch.sin(theta)
        sine = sine.repeat_interleave(2,dim=-1)
        return sine

    def forward(self,x:torch.Tensor):
        
        # 自注意力机制
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        if self.use_rope:
            theta = self.pos*self.thetas
            cosines = self.get_cosine(theta)
            sines = self.get_sine(theta)

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
        # 更新模型的记忆
        delta_zeta = k.transpose(-2,-1)
        delta_memory = torch.matmul(delta_zeta,v)

        memory = self.memory + delta_memory
        zeta = self.zeta + delta_zeta
        # 检索模型中记忆相关的部分
        scale = torch.matmul(q,zeta) + eps     #使用无穷小量替代0，防止出现Nan的情况
        score = torch.matmul(q,memory) + torch.tanh(self.wo(v))
        attention = score/scale
        self.apply_update(memory=memory,zeta=zeta)
        return attention
    
    def apply_update(self,memory:torch.Tensor,zeta:torch.Tensor):
        with torch.no_grad():
            self.memory = memory
            self.zeta = zeta
            self.pos = self.pos + 1
        pass

    def is_use_rope(self,value):
        self.use_rope = value
        if self.use_rope:
            self.rotatry = torch.tensor([1 if i % 2 == 0 else -1 for i in range(self.out_features)],device=self.device)
            self.swap_indices = torch.tensor([i+1 if i % 2 == 0 else i-1 for i in range(self.out_features)], dtype=torch.long,device=self.device)
            indices = (torch.arange(self.out_features//2,dtype=torch.float32,device=self.device) + 1)/self.out_features
            self.thetas = 10000**(-2*indices)
    
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
        self.dropout = nn.Dropout(0.2)
        pass

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        B,T,S = x.shape
        x = x.view(B,T,1,S)
        y = torch.stack([self.cell(x[:,i,:,:]) for i in range(T)],dim=0)
        y = torch.transpose(y,0,1)  # 第一指标与第二指标进行交换
        B,T,X,S = y.shape
        y = y.view(B,T,S)
        y = self.dropout(self.projection(y))
        return y

    def forget(self):
        self.cell.forget()
        pass
    pass

class FeedForward(nn.Module):
    def __init__(self,in_features,out_features,device,*args, **kwargs) -> None:
        super(FeedForward,self).__init__(*args, **kwargs)
        self.dense = nn.Linear(in_features,4*out_features).to(device)
        self.linear = nn.Linear(4*out_features,out_features).to(device)
        self.dropout = nn.Dropout(0.3).to(device)
        self.gelu = nn.GELU()
        pass

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.dense(x))
        x = self.dropout(self.linear(x))
        return x

    pass

class MoELayer(nn.Module):
    # 专家层，强化局域信息
    def __init__(self,num_experts,in_features,out_features,device,*args, **kwargs) -> None:
        super(MoELayer,self).__init__(*args, **kwargs)
        self.head_size = in_features // num_experts
        self.experts = nn.ModuleList([FeedForward(in_features=self.head_size,out_features=out_features,device = device) for i in range(num_experts)])
        self.router = nn.Linear(in_features=in_features,out_features=num_experts).to(device)
        self.norm = nn.LayerNorm(normalized_shape=num_experts).to(device)
        pass

    def forward(self,x:torch.Tensor)->torch.Tensor:
        logits = self.router(x)
        router = torch.softmax(self.norm(logits),dim=-1)
        inputs = torch.split(x,self.head_size,dim=-1)
        experts = torch.stack([expert(input) for input,expert in zip(inputs,self.experts)],dim=0)
        y = torch.einsum('btx,xbts->bts',router,experts)
        return y
        pass

    pass

class DecoderBlock(nn.Module):
    def __init__(self,units, device,*args, **kwargs) -> None:
        super(DecoderBlock,self).__init__(*args, **kwargs)
        self.attention = LinearAttention(64,units,units,device).to(device)
        self.feed_forward = FeedForward(in_features=units,out_features=units,device=device)
        self.norm_1 = nn.LayerNorm(normalized_shape=units).to(device)
        self.norm_2 = nn.LayerNorm(normalized_shape=units).to(device)
        self.norm_3 = nn.LayerNorm(normalized_shape=units).to(device)
        pass

    def forward(self,x):
        input = x
        x = self.norm_1(x)
        x = self.norm_2(self.attention(x)+x)
        x = self.norm_3(self.feed_forward(x) + x) + input
        return x
        
    def forget(self):
        self.attention.forget()
        pass

    pass


class Decoder(nn.Module):
    '''
    Decoder层\n
    采用线性注意力机制实现\n
    可以手动清除模型记忆\n
    根据当前token输入以及记忆机制，预测下一token的输出\n
    '''
    def __init__(self,in_features,out_features,units = 512,num_block = 12,device = 'cpu',*args, **kwargs) -> None:
        super(Decoder,self).__init__(*args, **kwargs)
        self.linear = nn.Linear(in_features=in_features,out_features=units,bias=False).to(device)
        self.decoder_blocks = [DecoderBlock(units,device).to(device) for i in range(num_block)]
        self.decoder_blocks[0].attention.cell.is_use_rope(True)
        self.decoder = nn.Sequential(*(self.decoder_blocks))
        self.norm_1 = nn.LayerNorm(normalized_shape=units)
        self.norm_2 = nn.LayerNorm(normalized_shape=units)
        self.moe_2 = MoELayer(num_experts=8,in_features=units,out_features=units,device = device).to(device)
        self.logits = nn.Linear(units,out_features,bias=False).to(device)
        pass

    def forward(self,x):
        x = self.linear(x)
        x = self.norm_1(self.decoder(x) + x)
        x = self.norm_2(self.moe_2(x) + x)
        logits = self.logits(x)
        return logits
    
    def forget(self):
        for x in self.decoder_blocks:
            x.forget()
        pass
    pass

class DecoderLanguageModel(nn.Module):
    '''
    DecoderOnly架构语言模型\n
    采用线性注意力机制实现\n
    实现无限上下文计算\n
    '''
    def __init__(self,vocab_count,d_model,units = 512,num_block = 12,device = 'cpu', *args, **kwargs) -> None:
        super(DecoderLanguageModel,self).__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_count,d_model).to(device)
        self.decoder = Decoder(in_features=d_model,out_features=vocab_count,units=units,num_block=num_block,device=device,*args,**kwargs)
        pass

    def forward(self,x,labels=None):
        x = self.embedding(x)
        logits = self.decoder(x)
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
            # 清空模型的记忆
            self.forget()
            return loss,accuracy
            pass
        return logits[:,-1,:]
        pass

    def forget(self):
        self.decoder.forget()
    pass